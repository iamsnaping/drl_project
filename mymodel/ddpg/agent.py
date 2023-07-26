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
from mymodel.ddpg.ddpg_dataloader import *
import argparse
from mymodel.ddpg.ddpg_base_net import *

os.chdir(sys.path[0])

random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)


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


class ReplayBuffer2():
    def __init__(self, buffer_maxlen):
        self.buffer = []
        self.buffer_nums=0
        self.max_len=buffer_maxlen
        self.capacity=0

    def push(self, data):
        if self.capacity<self.max_len:
            self.buffer.append(data)
            self.capacity+=1
        else:
            self.buffer[self.buffer_nums]=data
            self.buffer_nums=(self.buffer_nums+1)%self.max_len

    def sample(self, batch_size):
        # state,,near_state,,last_goal_list_,last_action,action,isend,reward
        state_list=[]
        near_state_list=[]
        last_state_list=[]
        last_action_list=[]
        action_list=[]
        isend_list=[]
        reward_list=[]
        ep_list=[]
        goal_list=[]
        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            state,near_state,last_goal_list_,last_action,action,isend_,reward,goal,ep,goal_= experience
            # state, action, reward, next_state, done
            state_list.append([state])
            near_state_list.append([near_state])
            last_state_list.append([last_goal_list_])
            last_action_list.append([last_action])
            action_list.append([action])
            isend_list.append([isend_])
            reward_list.append([reward])
            ep_list.append([ep])
            goal_list.append[goal_]
            # print(last_state_list)
        return torch.as_tensor(state_list,dtype=torch.float32), \
               torch.as_tensor(near_state_list,dtype=torch.float32),\
               torch.as_tensor(last_state_list,dtype=torch.float32),\
               torch.as_tensor(np.array(last_action_list),dtype=torch.float32), \
               torch.as_tensor(np.array(action_list),dtype=torch.float32),\
               torch.as_tensor(isend_list,dtype=torch.float32).unsqueeze(-1),\
               torch.as_tensor(reward_list,dtype=torch.float32).unsqueeze(-1),\
               torch.as_tensor(ep_list,dtype=torch.float32).unsqueeze(-1),\
               torch.as_tensor(goal_list,dtype=torch.float32).unsqueeze(-1)
    def __len__(self):
        return len(self.buffer)



class ReplayBuffer():
    def __init__(self, buffer_maxlen):
        self.buffer = []
        self.buffer_nums=0
        self.max_len=buffer_maxlen
        self.capacity=0

    def push(self, data):
        if self.capacity<self.max_len:
            self.buffer.append(data)
            self.capacity+=1
        else:
            self.buffer[self.buffer_nums]=data
            self.buffer_nums=(self.buffer_nums+1)%self.max_len

    def sample(self, batch_size):
        # state,,near_state,,last_goal_list_,last_action,action,isend,reward
        state_list=[]
        near_state_list=[]
        last_state_list=[]
        last_action_list=[]
        action_list=[]
        isend_list=[]
        reward_list=[]
        ep_list=[]
        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            state,near_state,last_goal_list_,last_action,action,isend_,reward,goal,ep= experience
            # state, action, reward, next_state, done
            state_list.append([state])
            near_state_list.append([near_state])
            last_state_list.append([last_goal_list_])
            last_action_list.append([last_action])
            action_list.append([action])
            isend_list.append([isend_])
            reward_list.append([reward])
            ep_list.append([ep])
            # print(last_state_list)
        return torch.as_tensor(state_list,dtype=torch.float32), \
               torch.as_tensor(near_state_list,dtype=torch.float32),\
               torch.as_tensor(last_state_list,dtype=torch.float32),\
               torch.as_tensor(np.array(last_action_list),dtype=torch.float32), \
               torch.as_tensor(np.array(action_list),dtype=torch.float32),\
               torch.as_tensor(isend_list,dtype=torch.float32).unsqueeze(-1),\
               torch.as_tensor(reward_list,dtype=torch.float32).unsqueeze(-1),\
               torch.as_tensor(ep_list,dtype=torch.float32).unsqueeze(-1)
    def __len__(self):
        return len(self.buffer)




class ActorNet(PolicyBaseNet):
    def __init__(self,action_n,state_n,n_state_n,mouse_n ,num_layer,output_n):
        super(ActorNet, self).__init__(action_n,state_n,n_state_n,mouse_n ,num_layer,output_n)


    def act(self, last_goal,state,near_state,last_action,epochs):
        action = self(last_goal,state,near_state,last_action)
        noise = torch.tensor(np.random.normal(loc=0, scale=(5/(epochs+1))**2),
                             dtype=torch.float32)
        if self.is_noise:
            action = action+noise
        return action.squeeze().cpu().detach().numpy()

class ActorNet2(PolicyBaseNet2):
    def __init__(self,action_n,state_n,n_state_n,mouse_n ,num_layer,output_n):
        super(ActorNet2, self).__init__(action_n,state_n,n_state_n,mouse_n ,num_layer,output_n)

    def act(self, last_goal,state,near_state,last_action,epochs):
        action,goal = self(last_goal,state,near_state,last_action)
        action=torch.clamp(action,max=1,min=-1)
        noise = torch.tensor(np.random.normal(loc=0, scale=(5/(epochs+1))**2),
                             dtype=torch.float32)
        if self.is_noise:
            action = last_action+action + noise
        goal=goal.squeeze().cpu().detach().numpy()
        # goal[0]=goal[0]*1920+1920
        # goal[1]=goal[1]*540+540
        return action.squeeze().cpu().detach().numpy()+last_action,goal


class CriticNet(BaseNet):
    def __init__(self,in_n,ou_n,layer_n,meta_n):
        super(CriticNet, self).__init__(in_n,ou_n,layer_n,meta_n)



class Critic_Net:
    def __init__(self,lr, tao,mode):
        # self.target = CriticNet(40,1,10,2)
        # self.online = CriticNet(40,1,10,2)
        if mode==1:
            self.target=ActorNet(2,30,6,3,5,1)
            self.online=ActorNet(2,30,6,3,5,1)
        else:
            self.target=ActorNet(2,30,6,5,5,1)
            self.online=ActorNet(2,30,6,5,5,1)
        self.trainer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.tao = tao

    def update_params(self):
        for param_target, param in zip(self.target.parameters(), self.online.parameters()):
            param_target.data.copy_(param_target * (1 - self.tao) + param * self.tao)

    def load_dic(self,lpath):
        self.target.load_state_dict(torch.load(lpath))
        self.online.load_state_dict(torch.load(lpath))

class Actor_Net:
    def __init__(self, lr, tao, GAMMA,mode=1):
        if mode==2:
            self.target = ActorNet2(6,20,10,2,7,2)
            self.online = ActorNet2(6,20,10,2,7,2)
        else:
            self.target = ActorNet(6,20,10,2,7,2)
            self.online = ActorNet(6,20,10,2,7,2)
        self.online.is_noise = True
        self.target.is_noise = False
        self.trainer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.tao = tao
        self.GAMMA = GAMMA

    def update_params(self):
        for param_target, param in zip(self.target.parameters(), self.online.parameters()):
            param_target.data.copy_(param_target * (1 - self.tao) + param * self.tao)


class Agent:
    def __init__(self,buffer_maxlen=50000, lr=0.03, GAMMA=0.9,tao=0.9,mode=1):
        if mode==1:
            self.memo = ReplayBuffer(buffer_maxlen)
        else:
            self.memo=ReplayBuffer2(buffer_maxlen)
        self.actor_net = Actor_Net(lr,tao,GAMMA,mode)
        self.critic_net = Critic_Net(lr,tao,mode)
        self.critic_net.online.apply(weight_init)
        self.critic_net.target.apply(weight_init)
