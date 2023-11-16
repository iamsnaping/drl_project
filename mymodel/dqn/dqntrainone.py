import torch
from torch import nn
import os
import sys
from tqdm import tqdm
from torch.nn import functional as F
sys.path.append('/home/wu_tian_ci/drl_project')
from torch.utils.data import Dataset
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import argparse

from copy import deepcopy as dp

import time

from dqnenv import *

from dqnutils import *
import dqnutils as UTIL
import dqnbasenet
import dqnagent
import json
from dqnagent import *



class DataRecorder(object):
    def __init__(self):
        self.object_list=[]
        self.seq_num={}
        self.max_len=0
        self.res_list=[]
        self.restore_path=''
    
    # click,eye,goal,action,seq,mask,reward,nclick,neye,nseq
    def add(self,click,eye,goal,action,seq,reward):
        if click[0]==0 and click[1]==0:
            return
        click.extend([goal])
        click.append(seq)
        hash_list=str(click)
        # print(self.seq_num.get(hash_list))
        # print(self.seq_num.get(hash_list) is None)
        if self.seq_num.get(hash_list) is None:
            self.seq_num[hash_list]=self.max_len
            self.res_list.append([[eye,int(action),float(reward)]])
            self.max_len+=1
        else:
            idx=self.seq_num.get(hash_list)
            self.res_list[idx].append([eye,int(action),float(reward)])
    
    def print_res(self):
        for item in self.seq_num.items():
            print(item[0],self.res_list[item[1]])


    def restore_data(self):
        f=open(self.restore_path,'w')
        json_dict={}
        # print(f'maxlen {self.max_len}')
        for item in self.seq_num.items():
            # print(item[0],item[1])
            json_dict[item[0]]=str(self.res_list[item[1]])
        json.dump(json_dict,f)
    
    def __del__(self):
        self.restore_data()

class ValueRecoder:
    def __init__(self):
        self.nums=0
        self.data_dic={}
        self.restore_path=''


    def add(self,value):
        self.data_dic[str(self.nums)]=value
        self.nums+=1

    def restore_data(self):
        f=open(self.restore_path,'w')
        json.dump(self.data_dic,f,indent=2)
    
    def __del__(self):
        self.restore_data()



class NumRecorder(object):

    def __init__(self,nums) -> None:
        self.nums=nums
        self.recoder=[0 for i in range(nums)]
    
    def add(self,index):
        self.recoder[index]+=1

    def clearOne(self,index):
        self.recoder[index]=0

    def ind(self,index):
        return self.recoder[index]

    def clear(self):
        self.recoder=[0 for i in range(self.nums)]

        
def train_epoch(agent:Agent, lr, epochs, batch_size,device,mode,envP,store_path=None,json_path=None,value_path=None,critic_loss_path=None):
    random.seed(None)
    np.random.seed(None)
    agent.online.to(device)
    agent.target.to(device)
    EPOSILON_DECAY=epochs
    EPOSILON_START=0.1
    EPOSILON_END=0.02
    # ebuffer=ExampleBuffer(2**17)
    trainer = torch.optim.Adam(lr=lr, params=agent.online.parameters())
    loss=nn.MSELoss()
    agentBuffer=ReplayBuffer(2**17,device)
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    m_store_path_a=os.path.join(store_path,'dqnnetoffline.pt')
    reward_path=os.path.join(store_path,'reward.txt')
    updataInfo=os.path.join(store_path,'updateInfo.txt')
    testInfo=os.path.join(store_path,'testInfo.txt')
    updateTestInfo=os.path.join(store_path,'updateTestInfo.txt')
    f=open(reward_path,'w')
    f.write(envP+'\n')
    f.close()
    f=open(updataInfo,'w')
    f.write(envP+'\n')
    f.close()
    f=open(testInfo,'w')
    f.write(envP+'\n')
    f.close()
    f=open(updateTestInfo,'w')
    f.write(envP+'\n')
    f.close()
    t_reward=[]
    t_end_reward=[]
    t_reward_len=-1
    best_scores=-np.inf
    is_training=False
    # json_path='/home/wu_tian_ci/drl_project/mymodel/ddpg/pretrain_data/json_path'
    # envs 
    envPath=os.path.join('/home/wu_tian_ci/eyedata/seperate/',envP,'1')
    env=DQNEnv(envPath)
    trajectory=DQNTrajectory()
    env.load_dict()
    env.get_file()
    env.state_mode=mode
    numRecorder=0
    for K in tqdm(range(epochs)):
        rewards=[]
        endAveReward=[]
        endIn=0
        endOut=0
        totalIn=0
        totalOut=0
        traLenOverOne=0
        noActionNum=0
        noActionNumWithThreshold=0
        lastWithNoAction=0
        inNoAct=0
        outNoAct=0
        totalErrors=0
        traLenOverThree=0
        endInNoAct=0
        endOutNoAct=0
        for steps in range(500):
            ans=env.act()
            if isinstance(ans,bool):
                env.load_dict()
                env.get_file()
                numRecorder=0
                continue
            else:
                eye=torch.tensor([ans[0]],dtype=torch.int64).reshape((1,1,3)).to(device)
                click=torch.tensor([ans[1]],dtype=torch.int64).reshape((1,1,3)).to(device)
                index=[numRecorder for j in range(3)]
                index=torch.tensor([index],dtype=torch.float32).reshape(1,3,1).to(device)
                action=agent.act(click,eye,index)
                doneFlag=0
                if ans[3]==1:
                    doneFlag=0
                else:
                    doneFlag=1
                eposilon=np.interp(K,[0,EPOSILON_DECAY],[EPOSILON_START,EPOSILON_END])
                prob=random.random()
                if prob>eposilon:
                    trajectory.push(ans[1],ans[0],ans[2],action,doneFlag*UTIL.GAMMA)
                else:
                    trajectory.push(ans[1],ans[0],ans[2],random.randint(0,12),doneFlag*UTIL.GAMMA)
                numRecorder+=1
                if ans[3]==1:
                    numRecorder=0
                    trajectory.getNewTras()
                    traInfo=trajectory.getInfo()
                    if traInfo[0]>1:
                        traLenOverOne+=traInfo[0]
                        noActionNum+=traInfo[1]
                        noActionNumWithThreshold+=traInfo[2]
                    lastWithNoAction+=traInfo[3]
                    totalIn+=traInfo[4]
                    totalOut+=traInfo[5]
                    endIn+=traInfo[6]
                    endOut+=int(traInfo[6])^1
                    rewards.append(traInfo[7])
                    endAveReward.append(traInfo[8])
                    inNoAct+=traInfo[9]
                    outNoAct+=traInfo[10]
                    if traInfo[0]>=3:
                        traLenOverThree+=traInfo[0]
                        totalErrors+=traInfo[11]
                    TDZero=trajectory.getComTraZero()
                    if traInfo[12]!=-1:
                        endInNoAct+=traInfo[12]
                        endOutNoAct+=int(traInfo[12])^1
                    for tra in TDZero:
                        agentBuffer.push(tra)
                    trajectory.clear()
            dbatch_size=int((1+(agentBuffer.getRatio()))*batch_size)
            if (agentBuffer.holding>=dbatch_size):
                if not is_training:
                    with open(reward_path,'a') as f:
                        f.write('begin to train\n')
                    is_training=True
                click,eye,action,nextClick,nextEye,mask,reward,seq,nseq = agentBuffer.sample(dbatch_size)
                with torch.no_grad():
                    onlineValues=agent.online(nextClick,nextEye,nseq)
                    yAction=torch.argmax(onlineValues,dim=-1,keepdim=True)
                    targetValues=agent.target(nextClick,nextEye,nseq)
                    y=targetValues.gather(dim=-1,index=yAction)*mask+reward
                values=agent.online(click,eye,seq).gather(dim=-1,index=action)
                l = loss(values, y)
                trainer.zero_grad()
                l.backward()
                trainer.step()
                if  steps%10==0 and steps!=0:
                    agent.update()
        t_reward_len+=1
        if t_reward_len>=1000:
            t_reward_len%=1000
            t_reward[t_reward_len]=np.mean(rewards)
            t_end_reward[t_reward_len]=np.mean(endAveReward)
        else:
            t_reward.append(np.mean(rewards))
            t_end_reward.append(np.mean(endAveReward))
        if len(t_reward)>0 and t_reward[-1]>best_scores and is_training:
            torch.save(agent.target.state_dict(), m_store_path_a)
            best_scores= t_reward[-1]
            with open(updataInfo,'a',encoding='UTF-8') as updateInfoFile:
                updateInfoFile.write('eposides:'+str(K+1)+' ave_eposides_rewards:'+str(round(np.mean(t_reward),2))+\
                ' ave reward '+str(round(t_reward[-1],2)) +'\n end_reward:' +str(round(np.mean(t_end_reward[-1]),2))+' '+\
                ' ave_eposides_end_reward:'+str(round(np.mean(t_end_reward),2))+'\n'+\
                ' end_in: '+str(endIn)+' end_out: '+str(endOut)+' train_end_acc: '+str(round(endIn/(endOut+endIn),2))+\
                ' end_in_no_action: '+str(endInNoAct)+' end_out_no_act: '+str(endOutNoAct)+' acc:'+str(round(endInNoAct/(endInNoAct+endOutNoAct),2))+'\n'+\
                ' in: '+str(totalIn)+' _out:'+str(totalOut)+' train_ave_acc:'+str(round(totalIn/(totalIn+totalOut),2))+\
                ' in_no_act:'+str(inNoAct)+' out_no_act:'+str(outNoAct)+' acc:'+str(round(inNoAct/(inNoAct+outNoAct),2))+'\n'+\
                ' len_tra_over_one: '+str(traLenOverOne)+' no_action_num:'+str(noActionNum)+' no_action_num_80:'+str(noActionNumWithThreshold)+\
                ' acc:'+str(round(noActionNum/traLenOverOne,2))+' acc2:'+str(round(noActionNumWithThreshold/traLenOverOne,2))+'\n'+\
                ' len_tra_over_three: '+str(traLenOverThree)+' total_errors: '+str(totalErrors)+' acc: '+str(round(totalErrors/traLenOverThree,2))+'\n')
        with open(reward_path,'a',encoding='UTF-8') as rewardInfoFile:
            rewardInfoFile.write('eposides:'+str(K+1)+' ave_eposides_rewards:'+str(round(np.mean(t_reward),2))+\
                ' ave reward '+str(round(t_reward[-1],2)) +'\n end_reward:' +str(round(np.mean(t_end_reward[-1]),2))+' '+\
                ' ave_eposides_end_reward:'+str(round(np.mean(t_end_reward),2))+'\n'+\
                ' end_in: '+str(endIn)+' end_out: '+str(endOut)+' train_end_acc: '+str(round(endIn/(endOut+endIn),2))+\
                ' end_in_no_action: '+str(endInNoAct)+' end_out_no_act: '+str(endOutNoAct)+' acc:'+str(round(endInNoAct/(endInNoAct+endOutNoAct),2))+'\n'+\
                ' in: '+str(totalIn)+' _out:'+str(totalOut)+' train_ave_acc:'+str(round(totalIn/(totalIn+totalOut),2))+\
                ' in_no_act:'+str(inNoAct)+' out_no_act:'+str(outNoAct)+' acc:'+str(round(inNoAct/(inNoAct+outNoAct),2))+'\n'+\
                ' len_tra_over_one: '+str(traLenOverOne)+' no_action_num:'+str(noActionNum)+' no_action_num_80:'+str(noActionNumWithThreshold)+\
                ' acc:'+str(round(noActionNum/traLenOverOne,2))+' acc2:'+str(round(noActionNumWithThreshold/traLenOverOne,2))+'\n'+\
                ' len_tra_over_three: '+str(traLenOverThree)+' total_errors: '+str(totalErrors)+' acc: '+str(round(totalErrors/traLenOverThree,2))+'\n')
       


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-modelPath',type=str,default='20231022203322')
    parser.add_argument('-cuda',type=str,default='cuda:0')
    parser.add_argument('-mode',type=int,default=1)
    parser.add_argument('-net',type=int,default=1)
    parser.add_argument('-sup',type=str,default='50')
    parser.add_argument('-preload',type=bool,default=False)
    parser.add_argument('-lr',type=float,default=0.05)

    args=parser.parse_args()
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    agent=Agent(device=device)
    store=UTIL.getTimeStamp()

    if args.preload:
        actor_load=os.path.join('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain/',args.modelPath)
        actor_load=os.path.join(actor_load,'200pretrain.pt')
        agent.load(actor_load)

    for i in range(2,22):
        if i<10:
            env='0'+str(i)
        else:
            env=str(i)
        agent=Agent(device=device)

        if args.preload:
            actor_load=os.path.join('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain/',args.modelPath)
            actor_load=os.path.join(actor_load,'200pretrain.pt')
            agent.load(actor_load)
        store_path=os.path.join('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/',store[0:-6],'test1',env,store[-6:])
        json_path=os.path.join('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/json_path/',store[0:-6],'test1',env,store[-6:])
        value_path=os.path.join('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/value_path/',store[0:-6],'test1',env,store[-6:])
        critic_loss_path=os.path.join('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/critic_loss/',store[0:-6],'test1',env,store[-6:])
        if not os.path.exists(json_path):
            os.makedirs(json_path)
        if not os.path.exists(value_path):
            os.makedirs(value_path)
        if not os.path.exists(critic_loss_path):
            os.makedirs(critic_loss_path)


        if not os.path.exists(store_path):
            os.makedirs(store_path)
        with open(os.path.join(store_path,'envsinfo.txt'),'w') as f:
            f.write('envs'+str(env)+'\n'+'lr:'+str(args.lr)+' preload: '+str(args.preload))
            if args.sup!='50':
                f.write('\n'+args.sup)
        train_epoch(agent, args.lr, 20, 256,device,args.mode,store_path=store_path,envP=env,\
                    json_path=json_path,value_path=value_path,critic_loss_path=critic_loss_path)