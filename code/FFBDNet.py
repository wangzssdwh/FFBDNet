import torch
import argparse
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
import torch.nn as nn
from collections import defaultdict
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, buildMPNN

from layers import GraphConvolution


torch.manual_seed(1203)
np.random.seed(1203)

model_name = 'FFBDNet'
resume_name = ''
if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))
        
data_path = '../data/records_final.pkl'
voc_path = '../data/voc_final.pkl'

ehr_adj_path = '../data/ehr_adj_final.pkl'
ddi_adj_path = '../data/ddi_A_final.pkl'
molecule_path = '../data/idx2SMILES.pkl'
device = torch.device('cuda:0')

ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
data = dill.load(open(data_path, 'rb'))
voc = dill.load(open(voc_path, 'rb'))
molecule = dill.load(open(molecule_path, 'rb')) 

diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

split_point = int(len(data) * 2 / 3)
data_train = data[:split_point]
eval_len = int(len(data[split_point:]) / 2)
data_test = data[split_point:split_point + eval_len]
data_eval = data[split_point+eval_len:]
voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
MPNNSet, N_fingerprint, average_projection = buildMPNN(molecule, med_voc.idx2word, 2, device)


#Greedy MASK
MASK_RESH = 1000

for patient in data_train:
    record = []
    for adm in patient:
        MASK = []
        while True:
            DICT = {}
            #update DICT
            med_code_set = list(set(adm[2])-set(MASK))
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if ddi_adj[med_i, med_j] == 1 or ddi_adj[med_j, med_i] == 1:
                        DICT[med_i] = DICT.get(med_i,0) + 1
            
            #read DICT
            max_ddi_med = -1
            max_ddi_num = 0
            for med in DICT:
                if DICT[med] > max_ddi_num:
                    max_ddi_med = med
                    max_ddi_num = DICT[med]
                    
            #judge
            if max_ddi_num>=MASK_RESH:
                MASK.append(max_ddi_med)
            else:
                break
        adm.append(list(set(adm[2])-set(MASK)))


def eval(model, data_eval, voc_size, epoch):
    # evaluate
    print('')
    model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    case_study = defaultdict(dict)
    med_cnt = 0
    visit_cnt = 0
    for step, input in enumerate(data_eval):
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        for adm_idx, adm in enumerate(input):

            target_output1 = model(input[:adm_idx+1])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output1)
            y_pred_tmp = target_output1.copy()
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred.append(y_pred_tmp)
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)


        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        case_study[adm_ja] = {'ja': adm_ja, 'patient': input, 'y_label': y_pred_label}

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record)

    llprint('\tDDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f, AVG_MED: %.4f\n' % (
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
    ))

    dill.dump(case_study, open(os.path.join('saved', model_name, 'case_study.pkl'), 'wb'))

    # print('avg med', med_cnt / visit_cnt)

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

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
    
class FFBDNet(nn.Module):
    def __init__(self, vocab_size, ehr_adj, ddi_adj, MPNNSet, N_fingerprints, average_projection, emb_dim=4, device=torch.device('cpu:0')):
        super(FFBDNet, self).__init__()

        self.device = device
        self.emb_dim = emb_dim
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.voc_size = vocab_size
        
        # feature extra
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(3)])
        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.MPNN_molecule_Set = list(zip(*MPNNSet))
        self.MPNN_emb = MolecularGraphNeuralNetwork(N_fingerprints, emb_dim, layer_hidden=2, device=device).forward(self.MPNN_molecule_Set)
        self.MPNN_emb = torch.mm(average_projection.to(device=self.device), self.MPNN_emb.to(device=self.device))
        self.MPNN_emb.to(device=self.device)

        # attention moudle
        self.query = nn.ModuleList(
            [nn.Linear(2*emb_dim, emb_dim) for i in range(3)] )
        self.key = nn.ModuleList(
            [nn.Linear(emb_dim, emb_dim) for i in range(3)] )
        self.value = nn.ModuleList(
            [nn.Linear(emb_dim, emb_dim) for i in range(3)] )
        
        #fuse
        self.pat_fuse = nn.Linear(5*emb_dim, emb_dim)
        self.med_fuse = nn.Linear(4*emb_dim, emb_dim)
        self.fuse_weight = nn.Embedding(2, 1)
        
        # re-combination
        self.recomb = nn.Sequential(
            nn.Linear(emb_dim * 5+vocab_size[2], emb_dim * 4),
            nn.ReLU(),
            nn.Linear(emb_dim * 4, vocab_size[2])
        )
        self.recomd = nn.Sequential(
            nn.Linear(emb_dim * 5, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )
        self.docter_weight = nn.Embedding(2, vocab_size[2])
        self.init_weights()

    def forward(self, input):
        # patient representation
        patient = []
        patient_d = self.embeddings[0](torch.LongTensor(input[-1][0]).to(self.device)).sum(dim=0) # (dim,)
        patient_p = self.embeddings[1](torch.LongTensor(input[-1][1]).to(self.device)).sum(dim=0) # (dim,)
        patient.append(torch.cat([patient_d,patient_p])) # (1,2*dim)
        patient = torch.stack(patient)
        
        #history representation
        history = [ [] for x in range(3) ]
        for item in input[:-1]:
            history_d = self.embeddings[0](torch.LongTensor(item[0]).to(self.device)).sum(dim=0) # (dim,)
            history_p = self.embeddings[1](torch.LongTensor(item[1]).to(self.device)).sum(dim=0) # (dim,)
            history_m = self.embeddings[2](torch.LongTensor(item[2]).to(self.device)).sum(dim=0) # (dim,)
            history[0].append(history_d) # (1,dim)
            history[1].append(history_p) # (1,dim)
            history[2].append(history_m) # (1,dim)
        tmp = []
        for i in range(3):
            if len(history[i]) == 0:
                tmp.append(torch.zeros(self.emb_dim).to(self.device))
                continue
            history[i] = torch.stack(history[i]) # (adm-1,dim)
            query = self.query[i](patient) # (1,dim)
            key = self.key[i](history[i]) # (adm-1,dim)
            value = self.value[i](history[i]) # (adm-1,dim)
            attention_weight = F.softmax(torch.cosine_similarity(query,key,dim=1)).unsqueeze(1) # (1,adm-1)
            value = torch.sum(value * attention_weight,dim=0) # (,dim)
            tmp.append(value)
        history = torch.cat(tmp).unsqueeze(0) #(1,3*dim)
        
        #medication representation
        med_base = self.embeddings[2](torch.LongTensor([x for x in range(self.voc_size[2])]).to(self.device)) # (dim,)
        med_ehr = self.ehr_gcn()
        med_ddi = self.ddi_gcn()
        med_mole = self.MPNN_emb
        
        #fuse
        fuse_weight = self.fuse_weight(torch.tensor([0,1]).to(self.device))
        patient = torch.cat([patient,history],dim=1)
        patient_fuse = self.pat_fuse(patient.detach())
        medication = self.med_fuse(torch.cat([med_base,med_ehr,med_ddi,med_mole],dim=1)) #(med_num,dim)
        
        #classfier
        docter_weight = self.docter_weight(torch.tensor([0,1]).to(self.device))
        similar =  torch.cosine_similarity(patient_fuse,medication,dim=1).unsqueeze(0) 
        docter_direct = self.recomd(patient)
        docter_recomb = self.recomb(torch.cat([patient*fuse_weight[0],similar*fuse_weight[1]],dim=1))
        output = docter_direct*docter_weight[0] + docter_recomb*docter_weight[1]
        
        if self.training:
            neg_pred_prob = F.sigmoid(output)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()

            return output, batch_neg
        else:
            return output

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)


EPOCH = 40
LR = 0.0002
Neg_Loss = False
DDI_IN_MEM = True
TARGET_DDI = 0.05
T = 0.5
decay_weight = 0.85

model = FFBDNet(voc_size,ehr_adj,ddi_adj, MPNNSet, N_fingerprint, average_projection,emb_dim=32, device=device).to(device=device)
best_model = FFBDNet(voc_size,ehr_adj,ddi_adj, MPNNSet, N_fingerprint, average_projection,emb_dim=32, device=device).to(device=device)
best_epoch = 0
best_ja = 0
optimizer = Adam(model.parameters(), lr=1e-3)
print('parameters', get_n_params(model))


history = defaultdict(list)

EPOCH = 5
for epoch in range(EPOCH):
    loss_record1 = []
    start_time = time.time()
    model.train()
    prediction_loss_cnt = 0
    neg_loss_cnt = 0
    for step, input in enumerate(data_train):
        for idx, adm in enumerate(input):
            seq_input = input[:idx+1]
            loss1_target = np.zeros((1, voc_size[2]))
            loss1_target[:, adm[3]] = 1
            loss3_target = np.full((1, voc_size[2]), -1)
            for idx, item in enumerate(adm[3]):
                loss3_target[0][idx] = item

            target_output1, batch_neg_loss = model(seq_input)

            loss1 = F.binary_cross_entropy_with_logits(target_output1, torch.FloatTensor(loss1_target).to(device))
            loss3 = F.multilabel_margin_loss(F.sigmoid(target_output1), torch.LongTensor(loss3_target).to(device))
            
            loss = 0.9 * loss1 + 0.01 * loss3
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            loss_record1.append(loss.item())

        llprint('\rTrain--Epoch: %d, Step: %d/%d, L_p cnt: %d, L_neg cnt: %d' % (epoch, step, len(data_train), prediction_loss_cnt, neg_loss_cnt))
    # annealing

    ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_eval, voc_size, epoch)

    history['ja'].append(ja)
    history['ddi_rate'].append(ddi_rate)
    history['avg_p'].append(avg_p)
    history['avg_r'].append(avg_r)
    history['avg_f1'].append(avg_f1)
    history['prauc'].append(prauc)

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    llprint('\tEpoch: %d, Loss: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                        np.mean(loss_record1),
                                                                                        elapsed_time,
                                                                                        elapsed_time * (
                                                                                                    EPOCH - epoch - 1)/60))
    torch.save(model.state_dict(), open( os.path.join('saved', model_name, 'Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, ja, ddi_rate)), 'wb'))
    print('')
    if epoch != 0 and best_ja < ja:
        best_model.load_state_dict(model.state_dict())
        best_epoch = epoch
        best_ja = ja


dill.dump(history, open(os.path.join('saved', model_name, 'history.pkl'), 'wb'))

# test
torch.save(model.state_dict(), open(
    os.path.join('saved', model_name, 'final.model'), 'wb'))

print('best_epoch:', best_epoch)


eval(best_model, data_test, voc_size, 0)