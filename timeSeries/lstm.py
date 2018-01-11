#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.autograd import *
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def ToVariable(x):  
    tmp = torch.FloatTensor(x)  
    return Variable(tmp)

def get_train_dat(seq,k):  
    dat = []  
    L = len(seq)  
    for i in range(L-k-1):  
        indat = seq[i:i+k]  
        outdat = seq[i+1:i+k+1]  
        dat.append((indat,outdat))  
    return dat  

class LSTMpred(nn.Module):  
  
    def __init__(self,input_size,hidden_dim):  
        super(LSTMpred,self).__init__()  
        self.input_dim = input_size  
        self.hidden_dim = hidden_dim  
        self.lstm = nn.LSTM(input_size,hidden_dim)  
        self.hidden2out = nn.Linear(hidden_dim,1)  
        self.hidden = self.init_hidden()  
  
    def init_hidden(self):  
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),  
                Variable(torch.zeros(1, 1, self.hidden_dim)))  
  
    def forward(self,seq):  
        lstm_out, self.hidden = self.lstm(  
            seq.view(len(seq), 1, -1), self.hidden)  
        outdat = self.hidden2out(lstm_out.view(len(seq),-1))  
        return outdat  



model = LSTMpred(1, 3)
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
data = np.load('/data1/qtang/tianchi/data/processed/ts/125403_70.npy')
dat = get_train_dat(data,12)
for epoch in range(10):  
    print ('epoch:',epoch)  
    for seq, outs in dat[:int(0.7*len(dat))]:  
        seq = ToVariable(seq)  
        outs = ToVariable(outs)  
        #outs = torch.from_numpy(np.array([outs]))  
  
        optimizer.zero_grad()  
  
        model.hidden = model.init_hidden()  
  
        modout = model(seq)  
  
        loss = loss_function(modout, outs)  
        loss.backward()  
        optimizer.step()  
  
predDat = [] 
realDat = []
for seq, trueVal in dat[int(0.7*len(dat)):]:  
    seq = ToVariable(seq)  
    trueVal = ToVariable(trueVal)  
    predDat.append(model(seq)[-1].data.numpy()[0])
    realDat.append(trueVal[-1].data.numpy()[0])
print (predDat)
print ('#########################')
print (realDat)
