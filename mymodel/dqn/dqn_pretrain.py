import random

import os
import sys
import warnings
# warnings.filterwarnings('error')

sys.path.append('/home/wu_tian_ci/drl_project')
sys.path.append('/home/wu_tian_ci/drl_project/mymodel/pretrain_model')
sys.path.append('/home/wu_tian_ci/drl_project/mymodel/pretrain_model/mlpdrl.py')
import numpy as np
import torch
from torch import nn
from mymodel.dqn.agent import *
from mymodel.dqn.dqn_pretrain_data_loader import *





def weight_init(m):
    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def get_gaussian(x):
    return 1/(np.sqrt(2)*np.pi)*np.exp(-((x)**10)/2)


def get_scores(x,y,z,ans):
    x=(x.cpu().detach().numpy()-ans[:,:,0].reshape(-1,1,1).cpu().detach().numpy())**2
    y=(y.cpu().detach().numpy()-ans[:,:,1].reshape(-1,1,1).cpu().detach().numpy())**2
    z=(z.cpu().detach().numpy()-ans[:,:,2].reshape(-1,1,1).cpu().detach().numpy())**2
    x_scores=get_gaussian(x)
    y_scores=get_gaussian(y)
    z_scores=get_gaussian(z)
    # print(f'sum {x_scores.sum(),y_scores.sum(),z_scores.sum()} ans {ans[0][0]}')
    return 1.5-x,1.5-y,1-z

def train(net,epochs,train_iter,test_iter,lr,device):
    loss=nn.MSELoss()
    trainer = torch.optim.Adam(lr=lr, params=agent.online_net.parameters())
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(trainer, T_max=epochs)
    agent.online_net.to(device)
    agent.target_net.to(device)
    # trainer = torch.optim.RMSprop(agent.online_net.parameters(), lr=0.07, momentum=0.9, weight_decay=1e-8)
    for episode_i in range(epochs):
        episode_reward = 0.
        # state nextstate goal nextgoal flag
        scores=[0.,0.,0.,0.,0]
        for v,w,x,y,z in train_iter:
            v=v.to(device)
            w=w.to(device)
            x=x.to(device)
            y=y.to(device)
            z=z.to(device)
            x_value,y_value,z_value = net.target_net(v,x)
            max_x_value ,x_idx= x_value.max(dim=-1, keepdim=True)
            max_y_value,y_idx = y_value.max(dim=-1, keepdim=True)
            max_z_value,z_idx = z_value.max(dim=-1, keepdim=True)
            x_r,y_r,z_r=get_scores(x_idx,y_idx,z_idx,y)
            x_r=torch.tensor(x_r,dtype=torch.float32).to(device)
            y_r=torch.tensor(y_r,dtype=torch.float32).to(device)
            z_r=torch.tensor(z_r,dtype=torch.float32).to(device)
            x_scores = (1 - z) * net.GAMMA * max_x_value + x_r
            y_scores = (1 - z) * net.GAMMA * max_y_value + y_r
            z_scores = (1 - z) * net.GAMMA * max_z_value + z_r
            xq_values,yq_values,zq_values = net.online_net(w,x)
            xidxs=torch.argmax(xq_values,dim=-1,keepdim=True)
            yidxs=torch.argmax(yq_values,dim=-1,keepdim=True)
            zidxs=torch.argmax(zq_values,dim=-1,keepdim=True)
            x_q_values = torch.gather(input=xq_values, dim=-1, index=xidxs)
            y_q_values = torch.gather(input=yq_values, dim=-1, index=yidxs)
            z_q_values = torch.gather(input=zq_values, dim=-1, index=zidxs)

            # loss = nn.functional.smooth_l1_loss(y, a_q_values)

            # print(y,a_q_values)
            # print(y,a_q_values)
            l=loss(x_scores,x_q_values)*10+loss(y_scores,y_q_values)*0.5+loss(z_scores,z_q_values)*0.5
            scores[0]+=loss(xidxs,y[:,:,0].unsqueeze(-1)).cpu().detach().numpy()
            scores[1]+=loss(yidxs,y[:,:,1].unsqueeze(-1)).cpu().detach().numpy()
            scores[2]+=loss(zidxs,y[:,:,2].unsqueeze(-1)).cpu().detach().numpy()
            scores[3]+=l
            scores[4]+=1
            trainer.zero_grad()
            # agent.optimizer.zero_grad()
            l.backward()
            # loss.backward()
            trainer.step()
            for param_target, param in zip(agent.target_net.parameters(), agent.online_net.parameters()):
                param_target.data.copy_(param_target * (1 - 0.99) + param * 0.99)
            # agent.optimizer.step()
        schedule.step()
        l_r=trainer.param_groups[0]['lr']
        trainer.param_groups[0]['lr']
        print(f' eposide {episode_i} scores x:{scores[0]/scores[4]} y:{scores[1]/scores[4]} z:{scores[2]/scores[4]} l:{scores[3]/scores[4]} l_r:{l_r}')


if __name__=='__main__':
    agent=Agent(10,10)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    agent.online_net.apply(weight_init)
    agent.target_net.apply(weight_init)
    train_od = dataset_loader('/home/wu_tian_ci/eyedata/mixed/pretrain2/s1/train')
    train_iter=torch.utils.data.DataLoader(dataset=train_od,shuffle=False,batch_size=512,num_workers=12)
    test_od = dataset_loader('/home/wu_tian_ci/eyedata/mixed/pretrain2/s1/test')
    test_iter=torch.utils.data.DataLoader(dataset=test_od,shuffle=False,batch_size=512,num_workers=12)
    train(agent,1000,train_iter,test_iter,0.1,device)


