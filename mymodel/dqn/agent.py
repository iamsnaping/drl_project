import random

import numpy as np
from torch import nn
import torch
from mymodel.dqn.dqn_net import *


class Replaymemory:
    def __init__(self,n_s,n_a):
        self.n_s=n_s
        self.n_a=n_a
        self.MEMORY_SIZE=1000
        self.BATCH_SIZE=64
        self.all_s=np.empty(shape=(self.MEMORY_SIZE,n_s),dtype=np.float32)
        self.all_r=np.empty(shape=self.MEMORY_SIZE)
        self.all_a=np.random.randint(low=0,high=n_a,size=self.MEMORY_SIZE,dtype=np.uint8)
        self.all_done=np.random.randint(0,2,self.MEMORY_SIZE,dtype=np.uint8)
        self.all_s_=np.empty(shape=(self.MEMORY_SIZE,n_s),dtype=np.float32)
        self.t_memo=0
        self.t_max=0

    def add_memo(self,s,r,a,done,s_):
        self.all_s[self.t_memo]=s
        self.all_r[self.t_memo]=r
        self.all_a[self.t_memo]=a
        self.all_done[self.t_memo]=done
        self.all_s_[self.t_memo]=s_
        self.t_max = max(self.t_max, self.t_memo + 1)
        self.t_memo= (self.t_memo+1)%self.MEMORY_SIZE

    def sample(self):
        if self.t_max>=self.BATCH_SIZE:
            idxes=random.sample(range(self.t_max),self.BATCH_SIZE)
        else:
            idxes=range(0,self.t_max)
        batch_s=[]
        batch_a=[]
        batch_r=[]
        batch_done=[]
        batch_s_=[]
        for idx in idxes:
            batch_s.append(self.all_s[idx])
            batch_a.append(self.all_a[idx])
            batch_r.append(self.all_r[idx])
            batch_done.append(self.all_done[idx])
            batch_s_.append(self.all_s_[idx])

        _batch_s=torch.as_tensor(np.asarray(batch_s),dtype=torch.float32)
        _batch_a=torch.as_tensor(np.asarray(batch_a),dtype=torch.int64).unsqueeze(-1)
        _batch_r=torch.as_tensor(np.asarray(batch_r),dtype=torch.float32).unsqueeze(-1)
        _batch_done=torch.as_tensor(np.asarray(batch_done),dtype=torch.int64).unsqueeze(-1)
        _batch_s_=torch.as_tensor(np.asarray(batch_s_),dtype=torch.float32)
        return _batch_s,_batch_a,_batch_r,_batch_done,_batch_s_



class Agent:
    def __init__(self,n_input,n_output):
        self.n_input=n_input
        self.n_output=n_output

        self.GAMMA=0.99
        self.lr=1e-3
        # self.memo=Replaymemory(n_input,n_output)

        self.online_net=DQNNET(15,20,8,6,9,26,7,5)
        self.target_net=DQNNET(15,20,8,6,9,26,7,5)

        self.optimizer=torch.optim.Adam(self.online_net.parameters(),lr=self.lr)