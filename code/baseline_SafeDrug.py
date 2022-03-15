import dill
import numpy as np
import argparse
from collections import defaultdict
from sklearn.metrics import jaccard_score
from torch.optim import Adam
import os
import torch
import time
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, buildMPNN
import torch.nn.functional as F
import math
import torch.nn as nn
from torch.nn.parameter import Parameter


torch.manual_seed(1203)
np.random.seed(2048)
model_name = 'SafeDrug'
resume_path = ""

if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

def eval(model, data_eval, voc_size, epoch):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input):
            target_output, _ = model(input[:adm_idx+1])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prod
            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)
            
            # prediction med set
            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rtest step: {} / {}'.format(step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path='../data/ddi_A_final.pkl')

    llprint('\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'.format(
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
    ))

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt


data_path = '../data/records_final.pkl'
voc_path = '../data/voc_final.pkl'

ehr_adj_path = '../data/ehr_adj_final.pkl'
ddi_adj_path = '../data/ddi_A_final.pkl'
ddi_mask_path = '../data/ddi_mask_H.pkl'
molecule_path = '../data/idx2SMILES.pkl'
device = torch.device('cuda:0')

ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
ddi_mask_H = dill.load(open(ddi_mask_path, 'rb'))
data = dill.load(open(data_path, 'rb'))
molecule = dill.load(open(molecule_path, 'rb')) 

voc = dill.load(open(voc_path, 'rb'))
diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

split_point = int(len(data) * 2 / 3)
data_train = data[:split_point]
eval_len = int(len(data[split_point:]) / 2)
data_test = data[split_point:split_point + eval_len]
data_eval = data[split_point+eval_len:]

voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
MPNNSet, N_fingerprint, average_projection = buildMPNN(molecule, med_voc.idx2word, 2, device)


class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, mask):
        weight = torch.mul(self.weight, mask)
        output = torch.mm(input, weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

    
    
class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprint, dim, layer_hidden, device):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.device = device
        self.embed_fingerprint = nn.Embedding(N_fingerprint, dim).to(self.device)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim).to(self.device)
                                            for _ in range(layer_hidden)])
        self.layer_hidden = layer_hidden

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch proc essing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(self.device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.mm(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def forward(self, inputs):

        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        """MPNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            # fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.
            fingerprint_vectors = hs

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        return molecular_vectors
    
    
    
class SafeDrugModel(nn.Module):
    def __init__(self, vocab_size, ddi_adj, ddi_mask_H, MPNNSet, N_fingerprints, average_projection, emb_dim=256, device=torch.device('cpu:0')):
        super(SafeDrugModel, self).__init__()

        self.device = device

        # pre-embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(2)])
        self.dropout = nn.Dropout(p=0.5)
        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(2)])
        self.query = nn.Sequential(
                nn.ReLU(),
                nn.Linear(2 * emb_dim, emb_dim)
        )

        # bipartite local embedding
        self.bipartite_transform = nn.Sequential(
            nn.Linear(emb_dim, ddi_mask_H.shape[1])
        )
        self.bipartite_output = MaskLinear(ddi_mask_H.shape[1], vocab_size[2], False)
        
        # MPNN global embedding
        self.MPNN_molecule_Set = list(zip(*MPNNSet))

        self.MPNN_emb = MolecularGraphNeuralNetwork(N_fingerprints, emb_dim, layer_hidden=2, device=device).forward(self.MPNN_molecule_Set)
        self.MPNN_emb = torch.mm(average_projection.to(device=self.device), self.MPNN_emb.to(device=self.device))
        self.MPNN_emb.to(device=self.device)
        # self.MPNN_emb = torch.tensor(self.MPNN_emb, requires_grad=True)
        self.MPNN_output = nn.Linear(vocab_size[2], vocab_size[2])
        self.MPNN_layernorm = nn.LayerNorm(vocab_size[2])
        
        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)
        self.init_weights()

    def forward(self, input):

	    # patient health representation
        i1_seq = []
        i2_seq = []
        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        for adm in input:
            i1 = sum_embedding(self.dropout(self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
            i2 = sum_embedding(self.dropout(self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))
            i1_seq.append(i1)
            i2_seq.append(i2)
        i1_seq = torch.cat(i1_seq, dim=1) #(1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1) #(1,seq,dim)

        o1, h1 = self.encoders[0](
            i1_seq
        )
        o2, h2 = self.encoders[1](
            i2_seq
        )
        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(dim=0) # (seq, dim*2)
        query = self.query(patient_representations)[-1:, :] # (seq, dim)
        
	    # MPNN embedding
        MPNN_match = F.sigmoid(torch.mm(query, self.MPNN_emb.t()))
        MPNN_att = self.MPNN_layernorm(MPNN_match + self.MPNN_output(MPNN_match))
        
	    # local embedding
        bipartite_emb = self.bipartite_output(F.sigmoid(self.bipartite_transform(query)), self.tensor_ddi_mask_H.t())
        
        result = torch.mul(bipartite_emb, MPNN_att)
        
        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)

        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)


model = SafeDrugModel(voc_size, ddi_adj, ddi_mask_H, MPNNSet, N_fingerprint, average_projection, emb_dim=64, device=device)
model.to(device=device)
print('parameters', get_n_params(model))
optimizer = Adam(list(model.parameters()), lr=5e-4)


history = defaultdict(list)
best_epoch, best_ja = 0, 0


EPOCH = 10
for epoch in range(EPOCH):
    tic = time.time()
    print ('\nepoch {} --------------------------'.format(epoch + 1))

    model.train()
    for step, input in enumerate(data_train):

        loss = 0
        for idx, adm in enumerate(input):

            seq_input = input[:idx+1]
            loss_bce_target = np.zeros((1, voc_size[2]))
            loss_bce_target[:, adm[2]] = 1

            loss_multi_target = np.full((1, voc_size[2]), -1)
            for idx, item in enumerate(adm[2]):
                loss_multi_target[0][idx] = item

            result, loss_ddi = model(seq_input)

            loss_bce = F.binary_cross_entropy_with_logits(result, torch.FloatTensor(loss_bce_target).to(device))
            loss_multi = F.multilabel_margin_loss(F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device))

            result = F.sigmoid(result).detach().cpu().numpy()[0]
            result[result >= 0.5] = 1
            result[result < 0.5] = 0
            y_label = np.where(result == 1)[0]
            current_ddi_rate = ddi_rate_score([[y_label]], path='../data/ddi_A_final.pkl')

            loss = 0.9 * loss_bce + 0.01 * loss_multi
#             if current_ddi_rate <= 0.06:
#                 loss = 0.95 * loss_bce + 0.05 * loss_multi + loss_ddi
#             else:
#                 beta = min(0, 1 + (0.06 - current_ddi_rate) / 0.05)
#                 loss = beta * (0.95 * loss_bce + 0.05 * loss_multi) + (1 - beta) * loss_ddi

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        llprint('\rtraining step: {} / {}'.format(step, len(data_train)))

    print ()
    tic2 = time.time() 
    ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_eval, voc_size, epoch)
    print ('training time: {}, test time: {}'.format(time.time() - tic, time.time() - tic2))

    history['ja'].append(ja)
    history['ddi_rate'].append(ddi_rate)
    history['avg_p'].append(avg_p)
    history['avg_r'].append(avg_r)
    history['avg_f1'].append(avg_f1)
    history['prauc'].append(prauc)
    history['med'].append(avg_med)

    if epoch >= 5:
        print ('ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}'.format(
            np.mean(history['ddi_rate'][-5:]),
            np.mean(history['med'][-5:]),
            np.mean(history['ja'][-5:]),
            np.mean(history['avg_f1'][-5:]),
            np.mean(history['prauc'][-5:])
            ))

    torch.save(model.state_dict(), open(os.path.join('saved', model_name, \
        'Epoch_{}_TARGET_{:.2}_JA_{:.4}_DDI_{:.4}.model'.format(epoch, 0.06, ja, ddi_rate)), 'wb'))

    if epoch != 0 and best_ja < ja:
        best_epoch = epoch
        best_ja = ja

    print ('best_epoch: {}'.format(best_epoch))

dill.dump(history, open(os.path.join('saved', model_name, 'history_{}.pkl'.format(model_name)), 'wb'))

eval(model, data_test, voc_size, 0)
