from typing import Any
import torch
from torch import nn
import os
import sys
from tqdm import tqdm
import collections
sys.path.append('/home/wu_tian_ci/drl_project')
from torch.utils.data import Dataset
import numpy as np
from mymodel.ddpg.ddpg_base_net import *
import matplotlib.pyplot as plt
import matplotlib
from dqnbasenet import *

os.chdir(sys.path[0])


from mymodel.ddpg.ddpg_pretrain_critic import *
from mymodel.ddpg.ddpg_pretrain_policy import*
def weight_init(m):
    random.seed(1114)
    np.random.seed(1114)
    torch.manual_seed(1114)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
    # torch.manual_seed(None)
    # random.seed(None)
    # np.random.seed(None)




class Agent(object):
    def __init__(self,device,embed_n=32,output_n=13,num_layer=4,rnn_layer=2) -> None:
        self.target=DQNNet(device=device,embed_n=embed_n,output_n=output_n,num_layer=num_layer,rnn_layer=rnn_layer)
        self.online=DQNNet(device=device,embed_n=embed_n,output_n=output_n,num_layer=num_layer,rnn_layer=rnn_layer)
        self.target.apply(weight_init)
        self.online.apply(weight_init)
        self.device=device
    
    def load(self,loadPath):
        self.target.load_state_dict(torch.load(loadPath))
        self.online.load_state_dict(torch.load(loadPath))
        self.target.to(self.device)
        self.online.to(self.device)

    def update(self):
        self.target.load_state_dict(self.online.state_dict())
    

    def act(self,click,eye,indxes):
        # print(click.shape,eye.shape,indxes.shape)
        maxAction=torch.argmax(self.online(click,eye,indxes),dim=-1)
        action=maxAction.squeeze().cpu().detach().numpy()
        return action
    

class RNNAgent(object):

    def __init__(self,device,embed_n=64,rnn_layer=10) -> None:
        # device,embed_n=64,rnn_layer=10
        self.target=DQNRNNNet(embed_n=embed_n,rnn_layer=rnn_layer,device=device)
        self.online=DQNRNNNet(embed_n=embed_n,rnn_layer=rnn_layer,device=device)
        self.target.apply(weight_init).to(device)
        self.online.apply(weight_init).to(device)
        self.device=device

    
    def load(self,loadPath):
        self.target.load_state_dict(torch.load(loadPath))
        self.online.load_state_dict(torch.load(loadPath))
        self.target.to(self.device)
        self.online.to(self.device)

    def update(self):
        self.target.load_state_dict(self.online.state_dict())
    

    def act(self,click,eye,clickP,lengths):
        # print(click.shape,eye.shape,indxes.shape)
        maxAction=torch.argmax(self.online(click,eye,clickP,lengths),dim=-1)
        action=maxAction.squeeze().cpu().detach().numpy()
        return action

        

    
class REMAgent(object):
    def __init__(self,device,embed_n=64,rnn_layer=10,networksNum=5) -> None:
        # device,embed_n=64,rnn_layer=10

        self.target=REMNet(embed_n=embed_n,rnn_layer=rnn_layer,device=device,remBlocskNum=networksNum)
        self.online=REMNet(embed_n=embed_n,rnn_layer=rnn_layer,device=device,remBlocskNum=networksNum)
        self.target.apply(weight_init).to(device)
        self.online.apply(weight_init).to(device)
        self.device=device

    
    def load(self,loadPath):
        self.target.load_state_dict(torch.load(loadPath))
        self.online.load_state_dict(torch.load(loadPath))
        self.target.to(self.device)
        self.online.to(self.device)

    def update(self):
        self.target.load_state_dict(self.online.state_dict())
    

    def act(self,click,eye,clickP,lengths,person):
        actions=self.online(click,eye,clickP,lengths,person)
        actions=sum(actions)/len(actions)
        maxAction=torch.argmax(actions,dim=-1)
        action=maxAction.squeeze().cpu().detach().numpy()
        return action
    
    def eval(self):
        self.target.eval()
        self.online.eval()
    
    def train(self):
        self.target.train()
        self.online.train()

     
class REMAgent2(object):
    def __init__(self,device,embed_n=64,rnn_layer=10,networksNum=5) -> None:
        # device,embed_n=64,rnn_layer=10

        self.target=REMNet2(embed_n=embed_n,rnn_layer=rnn_layer,device=device,remBlocskNum=networksNum)
        self.online=REMNet2(embed_n=embed_n,rnn_layer=rnn_layer,device=device,remBlocskNum=networksNum)
        self.target.apply(weight_init).to(device)
        self.online.apply(weight_init).to(device)
        self.device=device

    
    def load(self,loadPath):
        self.target.load_state_dict(torch.load(loadPath))
        self.online.load_state_dict(torch.load(loadPath))
        self.target.to(self.device)
        self.online.to(self.device)

    def update(self):
        self.target.load_state_dict(self.online.state_dict())
    

    def act(self,click,eye,clickP,lengths,person,scene):
        actions=self.online(click,eye,clickP,lengths,person,scene)
        actions=sum(actions)/len(actions)
        maxAction=torch.argmax(actions,dim=-1)
        action=maxAction.squeeze().cpu().detach().numpy()
        return action
    
    def eval(self):
        self.target.eval()
        self.online.eval()
    
    def train(self):
        self.target.train()
        self.online.train()

 