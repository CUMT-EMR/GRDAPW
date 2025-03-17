# from xLSTM.model import xLSTM
from xLSTM import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from copy import deepcopy
import torch.nn.functional as F
import math
from torch import optim
from torch.utils.data import DataLoader,Dataset
from  sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("use device :" + str(device))
class DataHandler(Dataset):
    def __init__(self,data) -> None:
        super().__init__()        
        data = data.reset_index()
        all_data = data['all_data']
        all_label = data['all_label']
        
        self.all_data =  all_data
        self.target = all_label
    def __getitem__(self, index):
        return {"data":torch.tensor([self.all_data[index]],dtype=torch.float),
                "target":torch.tensor([self.target[index]],dtype=torch.float)}
    def __len__(self):
        return len(self.all_data)

batch_size = 16
d_model=512
seq_len =256
hidden_size = 256
dropout = 0.6
cfgslstm = xLSTMBlockStackConfig(
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend="cuda",
            num_heads=4,
            conv1d_kernel_size=4,
            bias_init="powerlaw_blockdependent",
        ),
        feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
    ),
    context_length=256,
    num_blocks=7,
    embedding_dim=256,
    slstm_at=[7],

)
cfgmlstm =xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            qkv_proj_blocksize=4,
            num_heads=4,
            conv1d_kernel_size=4,
        )
    ),
    context_length=256,
    num_blocks=7,
    embedding_dim=256,

)

class Slstmmodel(nn.Module):
    def __init__(self,output_dim,input_dim,seq_len ,hidden_size,num_layers,num_blocks,dropout,bidirectional,cfg) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.pos_embedding  = PositionalEncoding(d_model=input_dim,max_len=seq_len).to(device)
        self.linear1 = nn.Linear(1,hidden_size).to(device)
        self.xslstm =  xLSTMBlockStack(cfg).to(device)
        self.linear = nn.Linear(seq_len*hidden_size,1).to(device)
        self.sigmoid = nn.Sigmoid().to(device)
    def forward(self,input):
        input = self.pos_embedding(input.to(device)).transpose(1,2)
        input  = self.linear1(input).to(device)
        input = self.pos_embedding(input.to(device)).transpose(1,2)
        hidden_states=[]
        y = self.xslstm(input)
        return  self.linear(y.reshape(y.shape[0],-1))
class mlstmmodel(nn.Module):
    def __init__(self,output_dim,input_dim,seq_len,hidden_size,num_layers,num_blocks,dropout,bidirectional,cfg) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(1,hidden_size).to(device)
        self.pos_embedding  = PositionalEncoding(d_model=input_dim,max_len=seq_len).to(device)
        self.xmlstm =  xLSTMBlockStack(cfg).to(device)
        self.linear = nn.Linear(seq_len*hidden_size,1).to(device)
        self.sigmoid = nn.Sigmoid().to(device)
    def forward(self,input):
        input = self.pos_embedding(input.to(device)).transpose(1,2)
        input  = self.linear1(input)
        hidden_states=[]
        for i in range(self.num_blocks):
            if i<=2:
                hidden_states.append(self.hidden_size)
            elif i<=self.num_blocks-3:
                hidden_states.append(self.hidden_size*2)
            else:
                hidden_states.append(self.hidden_size)
        y= self.xmlstm(input)
        return  self.linear(y.reshape(y.shape[0],-1))
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            pe = pe.unsqueeze(0).transpose(0, 1)

            self.register_buffer('pe', pe)

    def forward(self, x):
            """
            x: [seq_len, batch_size, d_model]
            """
            x = x + self.pe[:x.size(0), :]
            return self.dropout(x)


criterion_slstm = nn.BCEWithLogitsLoss(reduction='sum').to(device)
criterion_mlstm = nn.BCEWithLogitsLoss(reduction='sum').to(device)

slstm = Slstmmodel(output_dim=1,input_dim=seq_len,seq_len=seq_len,hidden_size=hidden_size,num_layers=2,num_blocks=3,dropout=dropout,bidirectional=False,cfg = cfgslstm)
mlstm = mlstmmodel(output_dim=1,input_dim=seq_len,seq_len=seq_len,hidden_size=hidden_size,num_layers=2,num_blocks=3,dropout=dropout,bidirectional=False,cfg = cfgmlstm)
total_params = sum(p.numel() for p in slstm.parameters())
total_params += sum(p.numel() for p in slstm.buffers())
print(f'{total_params:,} total parameters.')
print(f'{total_params/(1024*1024):.2f}M total parameters.')
total_trainable_params = sum(
    p.numel() for p in slstm.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')
print(f'{total_trainable_params/(1024*1024):.2f}M training parameters.')

optimizer_slstm = torch.optim.Adam(slstm.parameters(), lr=0.006)
optimizer_mlstm = torch.optim.Adam(mlstm.parameters(), lr=0.006)
scheduler_slstm = ReduceLROnPlateau(optimizer_slstm, mode='min', factor=0.9, patience=2)
scheduler_mlstm = ReduceLROnPlateau(optimizer_mlstm, mode='min', factor=0.9, patience=2)
def train(trans_data_loader,test_data_loader,model_name="slstm"):
    slstm.train()
    mlstm.train()
    total_loss_slstm = 0
    total_loss_mlstm = 0
    items = 0
    for date_item in trans_data_loader:
        all_data = date_item['data'].to(device)
        all_target_list = date_item['target'].to(device).squeeze(1)
        if model_name=="slstm":
            slstm_outputs = slstm(all_data).to(device).squeeze(1)
            slstm_output = criterion_slstm(slstm_outputs,all_target_list).to(device)
            optimizer_slstm.zero_grad()
            slstm_output.backward()
            optimizer_slstm.step()
            items+=1
            total_loss_slstm+=slstm_output.item()
            
        elif model_name=="mlstm":
            mlstm_outputs = mlstm(all_data).to(device).squeeze(1)
            mlstm_output = criterion_mlstm(mlstm_outputs,all_target_list).to(device)
            optimizer_mlstm.zero_grad()
            mlstm_output.backward()
            optimizer_mlstm.step()
            items+=1
            total_loss_mlstm+=mlstm_output.item()
            
            
    if model_name=="slstm": 
        totle_slstm_test,acc_slstm_test = prediction_test(model=slstm,model_name="slstm",test_data_loader=trans_data_loader)
        totle_slstm_validation,acc_slstm_validation = prediction_test(model=slstm,model_name="slstm",test_data_loader=test_data_loader)
        return total_loss_slstm,totle_slstm_test,totle_slstm_validation,acc_slstm_test,acc_slstm_validation
    elif model_name=="mlstm":
        totle_mlstm_test,acc_mlstm_test = prediction_test(model=mlstm,model_name="mlstm",test_data_loader=trans_data_loader)
        totle_mlstm_validation,acc_mlstm_validation = prediction_test(model=mlstm,model_name="mlstm",test_data_loader=test_data_loader)
        return total_loss_mlstm,totle_mlstm_test,totle_mlstm_validation,acc_mlstm_test,acc_mlstm_validation   
    
def prediction_test(model,test_data_loader,model_name="slstm"):
    all_num = 0
    model.eval()
    totle_slstm = 0
    acc_slstm = 0
    totle_mlstm = 0
    acc_mlstm = 0
    for data_item in test_data_loader:
        data = data_item["data"].to(device)
        label_target = data_item["target"].to(device).squeeze(1)
        if model_name=="slstm":
            prediction_label_slstm = model(data).squeeze(1).to(device)
            slstm_loss = criterion_slstm(prediction_label_slstm,label_target).to(device)
            prediction_label_slstm = nn.Sigmoid()(prediction_label_slstm)
            totle_slstm+=slstm_loss.item()
        elif model_name=="mlstm":
            prediction_label_mlstm = model(data).squeeze(1).to(device)
            mlstm_loss = criterion_mlstm(prediction_label_mlstm,label_target).to(device)
            prediction_label_mlstm = nn.Sigmoid()(prediction_label_mlstm)
            totle_mlstm+=mlstm_loss.item()
        
        for i in range(len(label_target)):
            all_num+=1
            if model_name=="slstm":
                if label_target[i]==1:
                    if prediction_label_slstm[i]>0.5:
                        acc_slstm+=1
                else:
                    if prediction_label_slstm[i]<=0.5:
                        acc_slstm+=1
            elif model_name=="mlstm":
                if label_target[i]==1:
                    if prediction_label_mlstm[i]>0.5:
                        acc_mlstm+=1
                else:
                    if prediction_label_mlstm[i]<=0.5:
                        acc_mlstm+=1
        
    if model_name=="slstm":
        return totle_slstm,(acc_slstm/all_num)
    elif model_name=="mlstm":
        return totle_mlstm,(acc_mlstm/all_num)
def generate_pseudo_labels(slstm,mlstm,
                           X_unlabeled, unlabel_data,confidence_threshold=0.95):
    slstm.eval()
    mlstm.eval()
    new_label_data_slstm = []
    new_label_label_slstm = []
    new_label_data_mlstm = []
    new_label_label_mlstm = []
    new_unlabel_data = []
    new_unlabel_label = []
    acc_num_slstm = 0
    acc_num_mlstm = 0
    all_label_num_slstm=0
    all_label_num_mlstm=0
    acc_mlstm = 0
    acc_slstm = 0
    mlstm_condicate_number = []
    mlstm_condicate_label_number = []
    slstm_condicate_number = []
    slstm_condicate_label_number = []
    more_precent = False
    if int(len(unlabel_data)*0.3)>100:
        more_precent = True
    for date_item in X_unlabeled:
        all_data = date_item['data'].to(device)
        # print(all_data.shape)
        all_target_list = date_item['target'].to(device).squeeze(1)
        slstm_outputs = nn.Sigmoid()(slstm(all_data).to(device)).squeeze(1)
        mlstm_outputs = nn.Sigmoid()(mlstm(all_data).to(device)).squeeze(1)
        all_data = all_data.squeeze(1).cpu()
        # print("all_data"+str(all_data.shape))
        # print("all_target_list"+str(all_target_list.shape))
        all_target_list = all_target_list.cpu()
        for i in range(0,len(slstm_outputs)):
            slstm_label = False
            mlstm_label = False
            if slstm_outputs[i]>confidence_threshold or (1-slstm_outputs[i])>confidence_threshold:
                slstm_condicate_number.append(all_data[i].tolist())
                slstm_condicate_label_number.append(slstm_outputs[i])
                slstm_label = True
            if mlstm_outputs[i]>confidence_threshold or (1-mlstm_outputs[i])>confidence_threshold:
                mlstm_condicate_number.append(all_data[i].tolist())
                mlstm_condicate_label_number.append(mlstm_outputs[i])
                mlstm_label = True
            if  not mlstm_label and  not slstm_label :
                new_unlabel_data.append(all_data[i].tolist())
                new_unlabel_label.append(all_target_list[i].item())
    slstm_condicate_df = pd.DataFrame({"slstm_condicate_number":slstm_condicate_number,"slstm_condicate_label_number":slstm_condicate_label_number})
    mlstm_condicate_df = pd.DataFrame({"mlstm_condicate_number":mlstm_condicate_number,"mlstm_condicate_label_number":mlstm_condicate_label_number})
    slstm_condicate_df.sort_values(by="slstm_condicate_label_number",inplace=True)
    mlstm_condicate_df.sort_values(by="mlstm_condicate_label_number",inplace=True)
    pd.concat([slstm_condicate_df[:int(len(slstm_condicate_df)*0.3)]])
    if all_label_num_mlstm==0:
        acc_mlstm = 0
    else:
        acc_mlstm = acc_num_mlstm/all_label_num_mlstm
    if all_label_num_slstm==0:
        acc_slstm = 0
    else:
        acc_slstm = acc_num_slstm/all_label_num_slstm
    newlabeldata_mlstm  = pd.DataFrame({'all_data':new_label_data_mlstm,'all_label':new_label_label_mlstm})
    newlabeldata_slstm  = pd.DataFrame({'all_data':new_label_data_slstm,'all_label':new_label_label_slstm})
    newunlabeldata  = pd.DataFrame({'all_data':new_unlabel_data,'all_label':new_unlabel_label})
    return newlabeldata_slstm,newlabeldata_mlstm,newunlabeldata,acc_slstm,acc_mlstm
def split_data(labeled_data,unlabel_data):
    trans_data,test_validation = train_test_split(labeled_data,test_size=0.2,train_size=0.8,shuffle=False,random_state=1)
    test_data,validation_data = train_test_split(test_validation,test_size=0.5,train_size=0.5,shuffle=False,random_state=1)
    trans_dataloader = DataLoader(DataHandler(trans_data),batch_size=batch_size,shuffle=False)
    validation_dataloader = DataLoader(DataHandler(validation_data),batch_size=batch_size,shuffle=False)
    test_dataloader = DataLoader(DataHandler(test_validation),batch_size=batch_size,shuffle=False)
    unlabel_data_dataloader = None
    if unlabel_data is not None:
        unlabel_data_dataloader = DataLoader(DataHandler(unlabel_data),batch_size=batch_size,shuffle=False)
    return trans_dataloader,validation_dataloader,test_dataloader,unlabel_data_dataloader
def co_training():
    L1_trans_dataloader,L1_validation_dataloader,L1_test_dataloader,L2_trans_dataloader,L2_validation_dataloader,L2_test_dataloader,unlabel_data_dataloader,L1_data_labeled,L2_data_labeled,unlabel_data=main()
    loss_data_slstm=[]
    loss_data_mlstm=[]
    totle_loss_slstm_predction_test = []
    totle_loss_slstm_predction_validation = []
    totle_loss_mlstm_predction_test = []
    totle_loss_mlstm_predction_validation = []
    label_prection_acc_slstm_test = []
    label_prection_acc_slstm_validation = []
    label_prection_acc_mlstm_test = []
    label_prection_acc_mlstm_validation = []
    label_data_lenth_slstm = [len(L1_data_labeled)]
    label_data_lenth_mlstm = [len(L2_data_labeled)]
    unlable_data_lenth = [len(unlabel_data)]
    slstm_step = []
    mlstm_step = []
    label_acc_slstm = []
    label_acc_mlstm = []
    check_labeled_epoch = [0]
    Epoch = 1000
    for epoch in range(Epoch):
        total_loss_slstm,totle_slstm_test,totle_slstm_validation,acc_slstm_test,acc_slstm_validation = train(trans_data_loader=L1_trans_dataloader,test_data_loader=L1_test_dataloader,model_name="slstm")
        total_loss_mlstm,totle_mlstm_test,totle_mlstm_validation,acc_mlstm_test,acc_mlstm_validation = train(trans_data_loader=L2_trans_dataloader,test_data_loader=L2_test_dataloader,model_name="mlstm")
        totle_loss_slstm_predction_test.append(totle_slstm_test)
        totle_loss_slstm_predction_validation.append(totle_slstm_validation)
        totle_loss_mlstm_predction_test.append(totle_mlstm_test)
        totle_loss_mlstm_predction_validation.append(totle_mlstm_validation)
        label_prection_acc_slstm_test.append(acc_slstm_test)
        label_prection_acc_slstm_validation.append(acc_slstm_validation)
        label_prection_acc_mlstm_test.append(acc_mlstm_test)
        label_prection_acc_mlstm_validation.append(acc_mlstm_validation)
        if unlabel_data_dataloader is not None and acc_slstm_test>0.9 and acc_slstm_validation>0.9:
            # labeled,unlabel = generate_pseudo_labels(slstm,mlstm,unlabel_data)
            newlabeldata_slstm,newlabeldata_mlstm,newunlabeldata,acc_slstm,acc_mlstm = generate_pseudo_labels(slstm=slstm,mlstm=mlstm,X_unlabeled = unlabel_data_dataloader,unlabel_data = unlabel_data)
            label_acc_slstm.append(acc_slstm)
            label_acc_mlstm.append(acc_mlstm)
            check_labeled_epoch.append(epoch)
            unlable_data_lenth.append(len(newunlabeldata))
            if len(newunlabeldata)==0 :
                newunlabeldata  = None
            L2_data_labeled = pd.concat([L2_data_labeled, newlabeldata_slstm], ignore_index=True)
            L1_data_labeled = pd.concat([L1_data_labeled, newlabeldata_mlstm], ignore_index=True)
            label_data_lenth_slstm.append(len(L1_data_labeled))
            label_data_lenth_mlstm.append(len(L2_data_labeled))
            unlabel_data = newunlabeldata
            L1_trans_dataloader,L1_validation_dataloader,L1_test_dataloader,unlabel_data_dataloader = split_data(L1_data_labeled,unlabel_data)
            L2_trans_dataloader,L2_validation_dataloader,L2_test_dataloader,unlabel_data_dataloader = split_data(L2_data_labeled,unlabel_data)
        scheduler_loss_mlstm = deepcopy(total_loss_mlstm)
        scheduler_loss_slstm = deepcopy(total_loss_slstm)
        loss_data_slstm.append(total_loss_slstm)
        loss_data_mlstm.append(total_loss_mlstm)
        print("第%d个epoch slstm的学习率：%f" % (epoch, optimizer_slstm.param_groups[0]['lr']))
        slstm_step.append(optimizer_slstm.param_groups[0]['lr'])
        print("第%d个epoch mlstm的学习率：%f" % (epoch, optimizer_mlstm.param_groups[0]['lr']))
        mlstm_step.append(optimizer_mlstm.param_groups[0]['lr'])
        scheduler_slstm.step(scheduler_loss_slstm)
        scheduler_mlstm.step(scheduler_loss_mlstm)
        print('Epoch {}, slstm totle Loss {},mlstm totle Loss {}'.format(epoch, total_loss_slstm,total_loss_mlstm) )
        df = pd.DataFrame(loss_data_slstm, columns=['loss_data_slstm'])
        df['loss_data_mlstm'] = loss_data_mlstm
        df['step_data_mlstm'] = mlstm_step
        df['step_data_slstm'] = slstm_step
        df['totle_loss_mlstm_predction_test'] = totle_loss_mlstm_predction_test
        df['totle_loss_mlstm_predction_validation'] = totle_loss_mlstm_predction_validation
        df['totle_loss_slstm_predction_test'] = totle_loss_slstm_predction_test
        df['totle_loss_slstm_predction_validation'] = totle_loss_slstm_predction_validation
        df['label_prection_acc_slstm_test'] = label_prection_acc_slstm_test
        df['label_prection_acc_slstm_validation'] = label_prection_acc_slstm_validation
        df['label_prection_acc_mlstm_test'] = label_prection_acc_mlstm_test
        df['label_prection_acc_mlstm_validation'] = label_prection_acc_mlstm_validation
        df['epoch']=np.arange(0,len(loss_data_slstm))
        df.to_excel("s-mlstm_loss.xlsx", index=False)
        df = pd.DataFrame(label_acc_mlstm, columns=['label_acc_mlstm'])
        df['label_acc_slstm'] = label_acc_slstm
        df['epoch']=np.arange(0,len(label_acc_mlstm))
        df.to_excel("s-mlabel_acc.xlsx", index=False)
        lenchange = pd.DataFrame({"unlable_data_lenth":unlable_data_lenth,"label_data_lenth_mlstm":label_data_lenth_mlstm,"label_data_lenth_slstm":label_data_lenth_slstm,"check_labeled_epoch":check_labeled_epoch})
        lenchange.to_excel("lenchange_data.xlsx",index=False)
        
        
        
    torch.save(Slstmmodel,".\model\model_slstm.pt")
    torch.save(mlstmmodel,".\model\model_mlstm.pt")
    plt.plot(np.arange(0,Epoch),loss_data_slstm)
    df = pd.DataFrame(loss_data_slstm, columns=['loss_data_slstm'])
    df['loss_data_mlstm'] = loss_data_mlstm
    df['step_data_mlstm'] = mlstm_step
    df['step_data_slstm'] = slstm_step
    df['totle_loss_mlstm_predction_test'] = totle_loss_mlstm_predction_test
    df['totle_loss_mlstm_predction_validation'] = totle_loss_mlstm_predction_validation
    df['totle_loss_slstm_predction_test'] = totle_loss_slstm_predction_test
    df['totle_loss_slstm_predction_validation'] = totle_loss_slstm_predction_validation
    df['label_prection_acc_slstm_test'] = label_prection_acc_slstm_test
    df['label_prection_acc_slstm_validation'] = label_prection_acc_slstm_validation
    df['label_prection_acc_mlstm_test'] = label_prection_acc_mlstm_test
    df['label_prection_acc_mlstm_validation'] = label_prection_acc_mlstm_validation
    df['epoch']=np.arange(0,len(loss_data_slstm))
    df.to_excel("s-mlstm_loss.xlsx", index=False)
    df = pd.DataFrame(label_acc_mlstm, columns=['label_acc_mlstm'])
    df['label_acc_slstm'] = label_acc_slstm
    df['epoch']=np.arange(0,len(label_acc_mlstm))
    df.to_excel("s-mlabel_acc.xlsx", index=False)
    lenchange = pd.DataFrame({"unlable_data_lenth":unlable_data_lenth,"label_data_lenth_mlstm":label_data_lenth_mlstm,"label_data_lenth_slstm":label_data_lenth_slstm,"check_labeled_epoch":check_labeled_epoch})
    lenchange.to_excel("lenchange_data.xlsx",index=False)
    plt.show()
        
def main():
    input_file = "异常值数据集.xlsx"
    data_label = "T2"
    seq_len = 256
    df_anous = pd.read_excel(input_file, sheet_name='Processed Data')
    label_anos_data = df_anous[data_label+"_with_anomalies"]
    label_anos_label = df_anous[data_label+"_anomaly_label"]
    all_data = []
    all_label = []
    for i in range(len(label_anos_data)-seq_len-1):
        cur_data_group = label_anos_data[i:i+seq_len].tolist()
        group_label = label_anos_label[i+seq_len-1]
        all_label.append(group_label)
        all_data.append(cur_data_group)
    data_after = pd.DataFrame({'all_data':all_data,'all_label':all_label})
    labeled_data,unlabel_data = train_test_split(data_after,test_size=0.4,train_size=0.6,shuffle=False,random_state=1)
    L1_data_labeled,L2_data_labeled = train_test_split(labeled_data,test_size=0.5,shuffle=False,random_state=42)
    L1_trans_dataloader,L1_validation_dataloader,L1_test_dataloader,unlabel_data_dataloader = split_data(L1_data_labeled,unlabel_data)
    L2_trans_dataloader,L2_validation_dataloader,L2_test_dataloader,unlabel_data_dataloader = split_data(L2_data_labeled,unlabel_data)
    return L1_trans_dataloader,L1_validation_dataloader,L1_test_dataloader,L2_trans_dataloader,L2_validation_dataloader,L2_test_dataloader,unlabel_data_dataloader,L1_data_labeled,L2_data_labeled,unlabel_data
if __name__=='__main__':
  
    co_training()
