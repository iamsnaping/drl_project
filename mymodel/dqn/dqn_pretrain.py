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
from dqndataloader import *
from dqnbasenet import *
from dqnagent import Agent
from tqdm import tqdm

import dqnutils as UTIL




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
    # torch.manual_seed(None)
    random.seed(None)
    np.random.seed(None)




def train(agent,epochs,train_iter,test_iter,lr,device,storePath,supp,load=None):

    loss=nn.MSELoss()
    # agent.online.to(device)
    agent.target.to(device)
    print(device)
    trainer = torch.optim.RMSprop(agent.target.parameters(), lr=lr, momentum=0.9, weight_decay=1e-8)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(trainer, T_max=int(epochs*1.1))
    if load is not None:
        agent.target.load_state_dict(torch.load(load))
    if not os.path.exists(storePath):
        os.makedirs(storePath)
    infoPath=os.path.join(storePath,str(epochs)+'info.txt')
    with open(infoPath,'w') as f:
        f.write('epoch: '+str(epochs)+' lr: '+str(lr)+' supp: '+supp+'\n')
    modelInfo=os.path.join(storePath,str(epochs)+'modelInfo.txt')
    with open(modelInfo,'w') as f:
        f.write('epoch: '+str(epochs)+' lr: '+str(lr)+' supp: '+supp+'\n')
    modelPath=os.path.join(storePath,str(epochs)+'pretrain.pt')
    bestLoss=np.inf
    for episode_i in tqdm(range(epochs)):
        trainLoss=[]
        testLoss=[]
        # state nextstate goal nextgoal flag
        agent.target.train()
        for click,eye,goal,action,seq,mask,reward,nclick,neye,nseq in train_iter:
            click=click.to(device)
            eye=eye.to(device)
            goal=goal.to(device)
            action=action.to(device)
            mask=mask.to(device)
            reward=reward.to(device)
            nclick=nclick.to(device)
            neye=neye.to(device)
            seq=seq.to(device)
            nseq=nseq.to(device)

            values = agent.target(click,eye,seq)
            nvalues=agent.target(nclick,neye,nseq)
            max_v_value=torch.gather(input=values,dim=-1,index=action)
            max_nv_value,y_idx = nvalues.max(dim=-1, keepdim=True)
            y=max_nv_value*mask+reward
            l=loss(max_v_value,y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            trainLoss.append(l.cpu().detach().numpy())
            # agent.optimizer.step()
        schedule.step()
        agent.target.eval()
        with torch.no_grad():
            for click,eye,goal,action,seq,mask,reward,nclick,neye,nseq in test_iter:
                click=click.to(device)
                eye=eye.to(device)
                goal=goal.to(device)
                action=action.to(device)
                mask=mask.to(device)
                reward=reward.to(device)
                nclick=nclick.to(device)
                neye=neye.to(device)
                seq=seq.to(device)
                nseq=nseq.to(device)
                values = agent.target(click,eye,seq)
                nvalues=agent.target(nclick,neye,nseq)
                max_v_value = torch.gather(input=values,dim=-1,index=action)
                max_nv_value,y_idx = nvalues.max(dim=-1, keepdim=True)
                y=max_nv_value*mask+reward
                l=loss(max_v_value,y)
                testLoss.append(l.cpu().detach().numpy())
        # test
        
        if bestLoss>np.mean(testLoss):
            bestLoss=np.mean(testLoss)
            torch.save(agent.target.state_dict(),modelPath)
            with open(modelInfo,'a') as f:
                f.write('episode_i: '+str(episode_i+1)+' test ave loss: '+str(np.mean(testLoss))+' test max loss:'+str(np.max(testLoss))\
                    +' test min loss: '+str(np.min(testLoss))+'\n' )
                f.write('train ave loss: '+str(np.mean(trainLoss))+' train max loss:'+str(np.max(trainLoss))\
                    +' train min loss: '+str(np.min(trainLoss))+'\n' )
        with open(infoPath,'a') as f:
            f.write('episode_i: '+str(episode_i+1)+' test ave loss: '+str(np.mean(testLoss))+' test max loss:'+str(np.max(testLoss))\
                    +' test min loss: '+str(np.min(testLoss))+'\n' )
            f.write('train ave loss: '+str(np.mean(trainLoss))+' train max loss:'+str(np.max(trainLoss))\
                    +' train min loss: '+str(np.min(trainLoss))+'\n' )


if __name__=='__main__':
    train_dataset=DQNDataLoader('/home/wu_tian_ci/eyedata/dqn/1/s1/train1.json')
    test_dataset=DQNDataLoader('/home/wu_tian_ci/eyedata/dqn/1/s1/test1.json')
    train_iter=torch.utils.data.DataLoader(dataset=train_dataset,shuffle=True,batch_size=16384,num_workers=16,pin_memory=True)
    test_iter=torch.utils.data.DataLoader(dataset=test_dataset,shuffle=True,batch_size= 8192,num_workers=16,pin_memory=True)
    device = torch.device("cuda:2" if torch.cuda.is_available() else 'cpu')
    import time
    # device=torch.device('cpu')
    agent=Agent(device=device)
    ans=UTIL.getTimeStamp()
    # ans='20231022203322'
    storePath=os.path.join('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain/',ans)
    train(agent=agent,epochs=350,train_iter=train_iter,test_iter=test_iter,lr=0.07,device=device,storePath=storePath,supp='20231021500')
    loadPath=os.path.join(storePath,'350pretrain.pt')
    # storePath='/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain/2'
    train_dataset=DQNDataLoader('/home/wu_tian_ci/eyedata/dqn/1/s1/train1.json')
    test_dataset=DQNDataLoader('/home/wu_tian_ci/eyedata/dqn/1/s1/test1.json')
    train_iter=torch.utils.data.DataLoader(dataset=train_dataset,shuffle=True,batch_size=16384,num_workers=16,pin_memory=True)
    test_iter=torch.utils.data.DataLoader(dataset=test_dataset,shuffle=True,batch_size= 8192,num_workers=16,pin_memory=True)
    print(f'load path {loadPath}')
    # loadPath=''
    train(agent=agent,epochs=200,train_iter=train_iter,test_iter=test_iter,lr=0.01,device=device,storePath=storePath,supp='20231021300',load=loadPath)


