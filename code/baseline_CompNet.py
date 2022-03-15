import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import dill
import collections
import random
from sklearn.model_selection import train_test_split
from RL_data_util_ehr import *
from GCN import *
import gc
from sklearn.metrics import roc_auc_score

torch.manual_seed(1203)
random.seed(1203)
np.random.seed(1203)

from util import get_n_params

device='cuda:0'

class CNN(nn.Module):
    def __init__(self,vocab_size,emb_size,num_channels,hidden_dim,dropout):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(embedding_dim=emb_size,num_embeddings=vocab_size)
        self.conv=nn.Sequential(
            nn.Conv1d(           #input shape (1, 28, 28)
                in_channels=emb_size,   #input height
                out_channels=num_channels, # n_filters
                kernel_size=3,   # filter size
                stride=2,        # filter movement/step
                                 # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),              # output shape (16, 28, 28)
            nn.Tanh(),
            #nn.MaxPool2d(kernel_size=2),# choose max value in 2x2 area, output shape (16, 14, 14)
            nn.Conv1d(  # input shape (1, 28, 28)
                in_channels=num_channels,  # input height
                out_channels=num_channels,  # n_filters
                kernel_size=3,  # filter size
                stride=2,  # filter movement/step
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),
            nn.Tanh(),
            # output shape (16, 28, 28)
            #nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
            nn.Conv1d(  # input shape (1, 28, 28)
                in_channels=num_channels,  # input height
                out_channels=num_channels,  # n_filters
                kernel_size=3,  # filter size
                stride=2,  # filter movement/step
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),
        )
        self.dropout=dropout
        self.out=nn.Linear(num_channels,hidden_dim,bias=True)
        nn.init.kaiming_normal_(self.out.weight)
    def forward(self,x):
        #print('x:',type(x))
        x_emb=self.embedding(x).unsqueeze(0).permute(0,2,1)
        #print('x_emb:',x_emb.shape) #[1, 100, 30]
        x = self.conv(x_emb)
        # average and remove the extra dimension
        remaining_size = x.size(dim=2)
        features = F.max_pool1d(x, remaining_size).squeeze(dim=2)
        features = F.dropout(features, p=self.dropout)
        output = self.out(features)
        return output


class DQN(nn.Module):
    def __init__(self,state_size,action_size):
        super(DQN, self).__init__()
        # 一个简单的三层的感知器网络用来根据状态做决策
        self.W=nn.Parameter(torch.FloatTensor(state_size,state_size))
        # 使用xaview_uniform_方法初始化权重
        nn.init.xavier_uniform_(self.W.data)  # (2,8285,16)
        self.U=nn.Parameter(torch.FloatTensor(state_size,state_size))
        #nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_uniform_(self.U.data)  # (95,2)

        self.fc1=nn.Linear(state_size,512)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2=nn.Linear(512,action_size)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.learn_step_counter=0 #用于判断何时更新target网络

    def forward(self,x_):
        x_t,h_t_1=x_[0],x_[1]
        #x_t=torch.FloatTensor(x_t.cpu())
        #print('x_t:',x_t.shape) #1, 100
        #print('h_t_1:',type(h_t_1))

        h_t_1=h_t_1.to(device)
        #h_t_1=torch.from_numpy(h_t_1)
        #print('h_t_l:',h_t_1.shape)
        #print('h_t_1:',h_t_1[0][:5])
        state=F.sigmoid(torch.mm(self.W,x_t.t())+torch.mm(self.U,h_t_1.t()))
        #print('state:',state.t()[0][:5])
        fc1=F.relu(self.fc1(state.t()))
        output=self.fc2(fc1)
        #print('state-h:{}'.format(state.t()[0][:5]))
        return state.t(),output


class Agent(nn.Module):
    def __init__(self,state_size,action_size,layer_sizes):
        super(Agent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        # self.memory = collections.deque(maxlen=3000)
        self.gamma = 0.9  # 计算未来奖励的折扣率
        self.epsilon = 0.9  # agent最初探索环境选择action的探索率
        self.epsilon_min = 0.05  # agent控制随机探索的阈值
        self.epsilon_decay = 0.995  # 随着agent做选择越来越好，降低探索率
        self.cnn_diagnosis = CNN(voc_size[0] + 2, EMB_SIZE, 128, EMB_SIZE, 0.5).to(device)
        self.cnn_procedure = CNN(voc_size[1] + 2, EMB_SIZE, 128, EMB_SIZE, 0.5).to(device)
        self.rgcn= RGCN(layer_sizes,voc_size[2]+1,dev=device).to(device)
        self.model=DQN(state_size,action_size).to(device)
        self.target_model =DQN(state_size,action_size).to(device)
        self.model_params = list(self.cnn_diagnosis.parameters())+list(self.cnn_procedure.parameters()) +list(self.rgcn.parameters())+ list(self.model.parameters())
        
        
        self.load_params()
        self.update_target_model()

    def load_params(self):
        if os.path.exists('MIMIC-III/agent.pkl'):
            #reload the params
            trainedModel=torch.load('MIMIC-III/agent.pkl')
            print('load trained model....')
            self.cnn_diagnosis.load_state_dict(trainedModel.cnn_diagnosis.state_dict())
            self.cnn_procedure.load_state_dict(trainedModel.cnn_procedure.state_dict())
            self.rgcn.load_state_dict(trainedModel.rgcn.state_dict())
            self.model.load_state_dict(trainedModel.model.state_dict())
            self.target_model.load_state_dict(trainedModel.target_model.state_dict())

    def reset(self,x):
        #得到每个电子病历数据的表示
        x0=torch.LongTensor(x[0]).to(device)
        x1=torch.LongTensor(x[1]).to(device)
        diagnosis_f = self.cnn_diagnosis(x0)
        procedure_f=self.cnn_procedure(x1)
        f=torch.cat((diagnosis_f,procedure_f),0) #按行拼接 f:(batch,t1+t2,hidden_size) 即f:(batch,diagnosis_maxlen+procedure_maxlen,100)
        #print('g:',f.shape)
        return f # f.shape:(2,100)

    def act(self,x,h,selectedAction):
        #根据state 选择action
        if np.random.rand()<self.epsilon:
            while True:
                action=random.randrange(self.action_size)
                if action not in selectedAction:
                    return action,h #直接使用上一步的隐状态作为当前的隐状态
        next_h,output=self.model((x,h))
        while True:
            with torch.no_grad():
                action = torch.max(output, 1)[1]
                if action not in selectedAction:
                    return action,next_h
                else:
                    output[0][action]=-999999

    def new_state(self,f,g):
        #print('g:',g[0][:5])
        # 这里并不是简单的相加或者拼接 而是使用了一种gate attention机制
        a = nn.functional.softmax(torch.mm(f, g.t()))
        f_ = torch.mm(a.t(),f)
        x = f_+ g
        return x #x shape(1,100)

    def step(self,action,selectedAction,y):
        # 首先判断该action是否为结束的标志：
        if action==action_size-1:
            #判断当前结束的步数是否对：
            if len(selectedAction)==len(y)-1:
                reward=2
                return reward,0
            else: #不该结束 结束了 有两种情况 超过结束个数或者不到结束个数
                reward=-2
                return reward,0
        else:
            #根据该action进行奖励
            if int(action) in y and action not in selectedAction :
                reward=1
            else:
                reward=-1

            #更新新的药物图谱
            adjacencies=getADJ(action,selectedAction,drug2id,ddi_df)
            adjacencies=get_torch_sparse_matrix(adjacencies,device)

            _, g = self.rgcn(adjacencies) # g shape(1,100)
            del adjacencies
            gc.collect()
            return reward,g

    def replay(self,BATCH_SIZE):
        print('learning_step_counter:{},self.epsilon:{}'.format(self.model.learn_step_counter,self.epsilon))
        # 没训练一次 都将模型learn_step_counter加一 并且判断是否需要更新target网络
        if self.model.learn_step_counter%TARGET_UPDATE_ITER==0:
            print('Update target model.')
            self.update_target_model()

            # 保存下训练过程中的损失函数值
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon * self.epsilon_decay

        BATCH_SIZE = min(len(memory)+len(memory_done),BATCH_SIZE)
        self.model.learn_step_counter= self.model.learn_step_counter +1
        batch_idx=np.random.choice(len(memory_done),BATCH_SIZE-len(memory))
        #print(batch_idx)
        b_x=[]
        b_h=[]
        b_action = []
        b_reward=[]
        b_next_x=[]
        b_next_h=[]
        for id in range(len(memory)):
            b_x.append(memory[id][0])
            b_h.append(memory[id][1])
            b_action.append(memory[id][2])
            b_reward.append(memory[id][3])
            b_next_x.append(memory[id][4])
            b_next_h.append(memory[id][5])
            memory_done.append([memory[id][0].detach(),memory[id][1].detach(),memory[id][2],memory[id][3],memory[id][4].detach(),memory[id][5].detach()])
        
        #将这些数据转变成tensor对象
        b_x=torch.cat(b_x,0).to(device)
        b_h = torch.cat([item.to(device) for item in b_h], 0)
        b_reward=torch.FloatTensor(b_reward).to(device)
        b_action=torch.LongTensor(b_action).to(device)
        b_next_x=torch.cat(b_next_x,0)
        b_next_h = torch.cat([item.to(device) for item in b_next_h], 0)

        b_action=b_action.unsqueeze(1).to(device)
        _,q_eval=self.model((b_x,b_h))
        q_eval=q_eval.gather(1,b_action)
        #print('q_eval:',q_eval[0][:5])
        _,q_next=self.target_model((b_next_x,b_next_h))
        q_next=q_next.detach()

        q_target=(b_reward+GAMMA*q_next.max(1)[0]).unsqueeze(1)
        
        return q_eval,q_target
        #print('q_target:', q_target[0][:5])
        loss=self.loss(q_eval,q_target)

        self.optimizier.zero_grad()
        loss.backward()
        # for param in self.model_params:
        #     param.grad.data.clamp_(-5,5)
        self.optimizier.step()

        return loss

    def update_target_model(self):
        #加载self.model的参数到target_model中
        self.target_model.load_state_dict(self.model.state_dict())


from util import llprint, multi_label_metric, ddi_rate_score, get_n_params
from collections import defaultdict
def eval(agent, X,Y, voc_size, epoch):
    # evaluate
    print('')
    agent.cnn_diagnosis.eval()
    agent.cnn_procedure.eval()
    agent.rgcn.eval()
    agent.model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    case_study = defaultdict(dict)
    med_cnt = 0
    visit_cnt = 0
    step_idx=0
    for x,y in zip(X,Y):
        step_idx += 1
        visit_cnt += 1
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        
        if len(y)==0:
            break
        sampleReward = 0
        y = set(y)  # 因为y中有重复的药物 因此这里转变为集合 去掉重复的药物
        # 得到初始的状态
        # 使用GCN对adjaccies进行处理 得到图表示
        # if test:
        _, g = agent.rgcn(init_adj)
        #print('g:',g.shape)
        f= agent.reset(x)
        selectedAction=[]
        h = np.zeros((1, state_size))  # （1,100）
        h=torch.FloatTensor(h)
        #按理说应该不知道len(y)
        for step in range(max_step):
            x = agent.new_state(f, g)  # x shape:(100,1)
            #print('step:{},x:{},h:{}'.format(step,x[0][:5],h[0][:5]))

            next_h, output =agent.model((x,h))
            #print('output:',output[0][:5])
            next_h=next_h.detach()
            output=output.detach()
            prob,action=torch.max(output,1)
            # print('output:{}'.format(output[0][:5]))
            # print('prob:{},action:{}'.format(prob,action))
            if action == action_size - 1 and step == 0:
                output[0][action_size-1]=-999999
            while True:
                action = torch.max(output, 1)[1]
                if int(action) not in selectedAction:
                    break
                else:
                    output[0][action] = -999999
            # 执行该action 得到reward 并更新状态
            reward, _= agent.step(action, selectedAction, y)
            if type(_) != int:  # 说明预测的不是结束符
                g = _
                # 将选择的action加入到selectedAction中
                selectedAction.append(int(action))
                sampleReward += int(reward)
                next_x = agent.new_state(f, g)  # 得到新时刻的输入xt
                # 用新时刻的状态替代原先的状态
                x = next_x
                h = next_h

            else:  # 预测出了结束符
                selectedAction.append(int(action))
                sampleReward += int(reward)
                break
                
        selectedAction = sorted(selectedAction)
        selectedAction = selectedAction[:-1] if selectedAction[-1]==drug2id["END"] else selectedAction
        y = sorted(y)
        y = y[:-1] if y[-1] == drug2id["END"] else y
        y_gt_tmp = np.zeros(voc_size[2])
        y_gt_tmp[y] = 1
        y_gt.append(y_gt_tmp)
        
        y_pred_tmp = np.zeros(voc_size[2])
        y_pred_tmp[selectedAction] = 1
        y_pred.append(y_pred_tmp)
        y_pred_prob.append(y_pred_tmp)
        med_cnt += len(selectedAction)
        smm_record.append([selectedAction])

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        case_study[adm_ja] = {'ja': adm_ja, 'patient': input, 'y_label': y_pred_label}

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step_idx, len(X)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record)

    llprint('\tDDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f, AVG_MED: %.4f\n' % (
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
    ))

#     dill.dump(case_study, open(os.path.join('saved', "compnet", 'case_study.pkl'), 'wb'))

    # print('avg med', med_cnt / visit_cnt)

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


class Evaluate(object):
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
    def evaluate(self):
        # 评估在数据集上的模型表现
        # 其他统计指标
        Jaccard_list = []
        Recall_list = []
        Reward_list = []
        Precision_list=[]
        F_list=[]
        D_DList=[]

        for x,y in zip(self.X,self.Y):
            if len(y)==0:
                break
            sampleReward = 0
            y = set(y)  # 因为y中有重复的药物 因此这里转变为集合 去掉重复的药物
            # 得到初始的状态
            # 使用GCN对adjaccies进行处理 得到图表示
            # if test:
            _, g = agent.rgcn(init_adj)
            #print('g:',g.shape)
            f= agent.reset(x)
            selectedAction=[]
            h = np.zeros((1, state_size))  # （1,100）
            h=torch.FloatTensor(h)
            #按理说应该不知道len(y)
            for step in range(max_step):
                x = agent.new_state(f, g)  # x shape:(100,1)
                #print('step:{},x:{},h:{}'.format(step,x[0][:5],h[0][:5]))

                next_h, output =agent.model((x,h))
                #print('output:',output[0][:5])
                next_h=next_h.detach()
                output=output.detach()
                prob,action=torch.max(output,1)
                # print('output:{}'.format(output[0][:5]))
                # print('prob:{},action:{}'.format(prob,action))
                if action == action_size - 1 and step == 0:
                    output[0][action_size-1]=-999999
                while True:
                    action = torch.max(output, 1)[1]
                    if int(action) not in selectedAction:
                        break
                    else:
                        output[0][action] = -999999
                # 执行该action 得到reward 并更新状态
                reward, _= agent.step(action, selectedAction, y)
                if type(_) != int:  # 说明预测的不是结束符
                    g = _
                    # 将选择的action加入到selectedAction中
                    selectedAction.append(int(action))
                    sampleReward += int(reward)
                    next_x = agent.new_state(f, g)  # 得到新时刻的输入xt
                    # 用新时刻的状态替代原先的状态
                    x = next_x
                    h = next_h

                else:  # 预测出了结束符
                    selectedAction.append(int(action))
                    sampleReward += int(reward)
                    break

            jaccard, recall, precision, f_measure = self.evaluate_sample(selectedAction, y)

            Jaccard_list.append(jaccard)
            Recall_list.append(recall)
            Reward_list.append(sampleReward)
            Precision_list.append(precision)
            F_list.append(f_measure)
            # 判断生成的药物中是否有DDI药物对
            d_d,ddRate =self.evaluate_ddi(y_pred=selectedAction)
            #print('d_d:',d_d)
            D_DList.append(ddRate)
        avg_jaccard = sum(Jaccard_list) * 1.0 / len(Jaccard_list)
        avg_recall = sum(Recall_list) * 1.0 / len(Recall_list)
        avg_reward = sum(Reward_list) * 1.0 / len(Reward_list)
        avg_precision=sum(Precision_list)*1.0/len(Precision_list)
        avg_f=sum(F_list)*1.0/len(F_list)
        avg_ddr=sum(D_DList)*1.0/len(D_DList)
        print('avg_jaccard:{},avg_recall:{},avg_precision:{},avg_f:{},avg_reward:{},avg_ddr:{}'.format(avg_jaccard, avg_recall,avg_precision,avg_f, avg_reward,avg_ddr))
        del Jaccard_list,Recall_list,Reward_list,Precision_list,F_list

        return avg_reward,avg_jaccard,avg_recall,avg_precision,avg_f,avg_ddr

    def evaluate_sample(self,y_pred,y_true): #针对单个样本的三个指标的评估结果
        print('y_pred:',y_pred)
        print('y_true:',y_true)
        jiao_1 = [item for item in y_pred if item in y_true]
        bing_1 = [item for item in y_pred] + [item for item in y_true]
        bing_1 = list(set(bing_1))
        # print('jiao:',jiao_1)
        # print('bing:',bing_1)
        recall = len(jiao_1) * 1.0 / len(y_true)
        precision = len(jiao_1) * 1.0 / len(y_pred)
        jaccard = len(jiao_1) * 1.0 / len(bing_1)

        if recall + precision == 0:
            f_measure = 0
        else:
            f_measure = 2 * recall * precision * 1.0 / (recall + precision)
        print('jaccard:%.3f,recall:%.3f,precision:%.3f,f_measure:%.3f' % (jaccard, recall, precision, f_measure))
        del jiao_1,bing_1
        return jaccard,recall,precision,f_measure

    def evaluate_ddi(self,y_pred):
        y_pred=list(set(y_pred))
        #根据药物id找到对应的药物名称
        # pred_drugs=[drug2id.get(id) for id in y_pred]
        #判断这些药物中是否存在着对抗的药物对

        #对生成的药物进行两两组合
        D_D=[]
        for i in range(len(y_pred) - 1):
            for j in range(i + 1, len(y_pred)):
                key1 = [y_pred[i],y_pred[j]]
                key2 = [y_pred[j],y_pred[i]]

                if key1 in ddi_df or key2 in ddi_df:
                    # 记录下来该DDI数据  以便论文中的case Study部分分析
                    D_D.append(key1)
        allNum=len(y_pred)*(len(y_pred)-1)/2
        if allNum>0:
            return D_D,len(D_D)*1.0/allNum
        else:
            return D_D,0

    def plot_result(self,total_reward,total_recall,total_jaccard):
        # 画图
        import matplotlib.pyplot as plt
        import matplotlib
        # 开始画图
        plt.figure()
        ax = plt.gca()

        epochs = np.arange(len(total_reward))

        plt.subplot(1, 2, 1)
        # 设置横坐标的显示刻度为50的倍数
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        plt.plot(epochs, total_reward, label='Reward')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')

        plt.subplot(1, 2, 2)
        plt.plot(epochs, total_recall, label='Recall', color='red')
        plt.plot(epochs, total_jaccard, label='Jaccard')
        # 设置横坐标的显示刻度为50的倍数
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')

        plt.show()  # 展示


def testModel():

    # 评估一下模型在测试集上的表现
    test_eva = Evaluate(X_test, Y_test)
    avg_reward, avg_jaccard, avg_recall, avg_precision, avg_f, ddiNum = test_eva.evaluate()
    # 将结果写入到文件中
    print('The result on test set.....')
    #print('avg_reward:{},\tavg_jaccard:{},\tavg_recall:{},\tavg_precision:{},\tavg_f:{},\tddiNum:{}'.format(
    #    avg_reward,
    #    avg_jaccard, avg_recall, avg_precision, avg_f, ddiNum))
    del avg_reward, avg_jaccard, ddiNum, avg_precision, avg_f
    gc.collect()


data_path = '../data/records_final.pkl'
voc_path = '../data/voc_final.pkl'

ehr_adj_path = '../data/ehr_adj_final.pkl'
ddi_adj_path = '../data/ddi_A_final.pkl'
device = torch.device('cuda:0')

ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
data = dill.load(open(data_path, 'rb'))
voc = dill.load(open(voc_path, 'rb'))
diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

split_point = int(len(data) * 2 / 3)
data_train = data[:split_point]
eval_len = int(len(data[split_point:]) / 2)
data_test = data[split_point:split_point + eval_len]
data_eval = data[split_point+eval_len:]


drug2id = voc['med_voc'].word2idx
drug2id["END"] = len(drug2id)

X,Y = [],[]
for patient in data:
    for adm in patient:
        X.append([adm[0],adm[1]])
        Y.append(adm[2]+[drug2id["END"]])
        
diagnosis_maxlen=max([len(line[0]) for line in X])
procedure_maxlen=max([len(line[1]) for line in X])
voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(drug2id))

for line in X:
    line[0] = line[0] + [voc_size[0] for _ in range(diagnosis_maxlen-len(line[0]))]
    line[1] = line[1] + [voc_size[1] for _ in range(procedure_maxlen-len(line[1]))]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(1/3), random_state=1)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=1)

ddi_df=[]
for i in range(ddi_adj.shape[0]):
    for j in range(ddi_adj.shape[0]):
        if ddi_adj[i,j]==1:
            ddi_df.append([i,j])
voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))


LR=1e-5
EMB_SIZE=100
Layer_sizes=[50,50]
state_size = EMB_SIZE
action_size = len(drug2id)
print('action_size:',action_size)
BATCH_SIZE = 64
EPISODES = 500  # 让agent玩游戏的次数
TARGET_UPDATE_ITER=10
GAMMA=1
max_step=15
min_jaccard = 0.0
Test=False


init_adj= init_ADJ(drug2id)
init_adj=get_torch_sparse_matrix(init_adj,device)


agent = Agent(state_size, action_size, layer_sizes=Layer_sizes)
print('parameters', get_n_params(agent))


# agent = Agent(state_size, action_size, layer_sizes=Layer_sizes)
criterion=nn.MSELoss()
optimizier=torch.optim.Adam(list(agent.parameters()),lr=LR,betas=(0.9,0.999),weight_decay=5.0)
memory_done = []

# 使用GCN对adjaccies进行处理 得到图表示
_, init_g = agent.rgcn(init_adj) #init_g shape:(1,100)
for e in range(EPISODES):
    print('epoch:%d'%e)
    epochLoss=[]
    #针对每个EHR
    batch_index = 0
    for x,y in zip(X_train,Y_train):
        batch_index += 1
        memory=[]
#         g = init_g.detach()
        _, g = agent.rgcn(init_adj)
        sampleReward=0
        y=set(y) #因为y中有重复的药物 因此这里转变为集合 去掉重复的药物
        if len(y)==0:
            break
        # 使用RNN得到EHR的表示f
        f=agent.reset(x) # shape (2,100)
        h=np.zeros((1,state_size)) #（1,100）
        h=torch.FloatTensor(h)
        selectedAction =[]
        # 得到初始的状态
        for step in range(len(y)):
            x=agent.new_state(f,g) # x shape:(100,1)
            #print('step:{},h:{}'.format(step, h[0][:5]))
            if step==len(y)-1: #说明是最后一步 不再让其预测 而是直接给出结束标志
                #这是因为模型很难选择出结束标志 模型得到的都是惩罚 所以它几乎不可能预测出end标志
                selectedAction.append(action_size-1)
                reward=1
                sampleReward+=int(reward) #直接给奖励值
                memory.append([x,h,action_size-1,reward,x.detach(),h.detach()]) #因为预测出了结束 即没有增加新的节点 因为下一个状态还是state
            else:
                #根据状态选择action
                action,next_h=agent.act(x,h,selectedAction)
                if action==action_size-1 and step==0:
                    #第一个就预测出了结束 则不能结束
                    while True:
                        action,next_h=agent.act(x,h,selectedAction)
                        if action!=action_size-1:
                            break
                #执行该action 得到reward 并更新状态
                reward,_=agent.step(action,selectedAction,y)
                if type(_)!=int: #说明预测的不是结束符
                    g=_
                    #将选择的action加入到selectedAction中
                    selectedAction.append(int(action))
                    sampleReward+=int(reward)
                    next_x=agent.new_state(f,g) #得到新时刻的输入xt
                    #将经验放入经验池
                    # 记忆这次transition
                    memory.append([x,h, action, reward, next_x.detach(),next_h.detach()])
                    #用新时刻的状态替代原先的状态
                    x=next_x
                    h=next_h

                else: #预测出了结束符
                    selectedAction.append(int(action))
                    sampleReward+=int(reward)
                    memory.append([x,h,action,reward,x.detach(),h.detach()])

        # 用之前的经验训练agent
        memory_done = memory_done[-1000:]
        print(len(memory))
        if True:
            q_eval,q_target = agent.replay(BATCH_SIZE)
            loss=criterion(q_eval,q_target)
            optimizier.zero_grad()
            loss.backward(retain_graph=True)
            optimizier.step()
            epochLoss.append(float(loss.item()))
        if batch_index%2000 == 0:
            torch.save(agent,"MIMIC-III/agent{}-{}.pkl".format(e,batch_index))
        gc.collect()
        print("batch:{}/{}".format(batch_index,len(X_train)))
        # params = agent.model.state_dict()
        # for k, v in params.items():
        #     print(k)  # 打印网络中的变量名
        #     print(v[:5])  # 打印conv1的weight
        #     # print(params['fc2.bias'])  # 打印conv1的bias
    if e%1==0:
        #每10轮评估一下模型在测试集上和验证集上的表现
        val_eva=Evaluate(X_val,Y_val)
        avg_reward, avg_jaccard, avg_recall,avg_precision,avg_f,ddiRate=val_eva.evaluate()
        #将结果写入到文件中
        # 写结果文件
        file = open('MIMIC-III/result.txt', 'a+')

        file.write('{},\t{},\t{},\t{},\t{},\t{}\t{},\t{}'.format(e,float(sum(epochLoss)*1.0/len(epochLoss)),avg_reward,avg_jaccard,avg_recall,avg_precision,avg_f,ddiRate))
        file.write('\n')
        if avg_jaccard>min_jaccard:
            min_jacard=avg_jaccard
            #保存当前模型
            torch.save(agent,'MIMIC-III/agent.pkl')
            #print(agent.model_params[1])
            #每当结果提升了以后保存模型 并且在得到测试集上的结果
            testModel()
        del avg_reward,avg_jaccard,ddiRate,epochLoss,avg_precision,avg_f
        gc.collect()
        file.close()


eval(agent,X_test,Y_test,voc_size,0)