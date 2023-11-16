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

        
def train_epoch(agent:Agent, lr, epochs, batch_size,device,mode,multienvs,testEnvs,store_path=None,json_path=None,value_path=None,critic_loss_path=None):
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
    env_str=''
    for e in multienvs:
        env_str+=e+' '
    f.write(env_str+'\n')
    f.close()
    f=open(updataInfo,'w')
    f.write(env_str+'\n')
    f.close()
    f=open(testInfo,'w')
    f.write(env_str+'\n')
    f.close()
    f=open(updateTestInfo,'w')
    f.write(env_str+'\n')
    f.close()
    t_reward=[]
    t_end_reward=[]
    t_reward_len=-1
    best_scores=-np.inf
    is_training=False
    # json_path='/home/wu_tian_ci/drl_project/mymodel/ddpg/pretrain_data/json_path'
    # envs 
    envs=[]
    envFlags=[]
    envPaths=[]
    trajectorys=[]
    testenvs=[]
    for env in multienvs:
        envPath=os.path.join('/home/wu_tian_ci/eyedata/seperate/',env,'1')
        envs.append(DQNEnv(envPath))
        envFlags.append(False)
        envPaths.append(envPath)
        trajectorys.append(DQNTrajectory())
    trajectorys.append(DQNTrajectory())
    for i in range(len(envs)):
        envs[i].load_dict()
        envs[i].get_file()
    for env in testEnvs:
        testEnvPath=os.path.join('/home/wu_tian_ci/eyedata/seperate/',env,'1')
        testenvs.append(DQNEnv(testEnvPath))

    beginFlags=[True for i in range(len(envs))]
    numRecorder=NumRecorder(len(envs))
    testRecorder=NumRecorder(len(testenvs))
    for K in tqdm(range(epochs)):
        jdr=DataRecorder()
        jdr.restore_path=os.path.join(json_path,str(K)+'.json')
        # vdr=ValueRecoder()
        # vdr.restore_path=os.path.join(value_path,str(K)+'.json')
        clp=ValueRecoder()
        clp.restore_path=os.path.join(critic_loss_path,str(K)+'.json')
        for env in envs:
            env.state_mode=mode
        ll, t = 0, 0
        # env.refresh()
        rewards=[]
        #ans -> state last_goal_state last_goal goal is_end
        
        for i in range(len(testenvs)):
            testenvs[i].load_dict()
            testenvs[i].get_file()
        stopFlags=[False for i in range(len(testEnvs))]
        surviveFlags=len(testenvs)

        # eye click goal isEnd


        rewardsTest=[]
        #ans -> state last_goal_state last_goal goal is_end
        endAveRewardTest=[]
        endInTest=0
        endOutTest=0
        totalInTest=0
        totalOutTest=0
        traLenOverOneTest=0
        noActionNumTest=0
        noActionNumWithThresholdTest=0
        lastWithNoActionTest=0
        inNoActTest=0
        outNoActTest=0
        totalErrorsTest=0
        traLenOverThreeTest=0
        endInNoActTest=0
        endOutNoActTest=0
        testTras=[DQNTrajectory() for i in range(surviveFlags)]
        with torch.no_grad():
            while surviveFlags>0:
                testEyeList=[]
                testClickList=[]
                testIndexList=[]
                testIndexesList=[]
                for i in range(len(testEnvs)):
                    if stopFlags[i]:
                        continue
                    ans=testenvs[i].act()
                    if isinstance(ans,bool):
                        stopFlags[i]=True
                        testRecorder.clearOne(i)
                        surviveFlags-=1
                        continue
                    else:
                        eye=torch.tensor([ans[0]],dtype=torch.int64).reshape((1,1,3)).to(device)
                        click=torch.tensor([ans[1]],dtype=torch.int64).reshape((1,1,3)).to(device)
                        index=[testRecorder.ind(i) for j in range(3)]
                        index=torch.tensor([index],dtype=torch.float32).reshape(1,3,1).to(device)
                        doneFlag=0
                        if ans[3]==1:
                            doneFlag=0
                        else:
                            doneFlag=1
                        testTras[i].push(ans[1],ans[0],ans[2],0,doneFlag*UTIL.GAMMA)
                        testRecorder.add(i)
                        testEyeList.append(eye)
                        testClickList.append(click)
                        testIndexList.append(index)
                        testIndexesList.append(i)
                if surviveFlags<=0:
                    break
                testEyeCat=torch.cat(testEyeList,dim=0)
                testClickCat=torch.cat(testClickList,dim=0)
                testIndexCat=torch.cat(testIndexList,dim=0)
                testActions=agent.act(testClickCat,testEyeCat,testIndexCat)
                if testActions.shape==():
                    testActions=[testActions]
                for testAction,testIndex in zip(testActions,testIndexesList):
                    testTras[testIndex].tras[-1][3]=dp(testAction)
                    if testTras[testIndex].tras[-1][4]==0:
                        testTras[testIndex].getNewTras()

                        traInfoTest=testTras[testIndex].getInfo()
                        if traInfoTest[0]>1:
                            traLenOverOneTest+=traInfoTest[0]
                            noActionNumTest+=traInfoTest[1]
                            noActionNumWithThresholdTest+=traInfoTest[2]
                        lastWithNoActionTest+=traInfoTest[3]
                        totalInTest+=traInfoTest[4]
                        totalOutTest+=traInfoTest[5]
                        endInTest+=traInfoTest[6]
                        endOutTest+=int(traInfoTest[6])^1
                        rewardsTest.append(traInfoTest[7])
                        endAveRewardTest.append(traInfoTest[8])
                        inNoActTest+=traInfoTest[9]
                        outNoActTest+=traInfoTest[10]
                        if traInfoTest[0]>=3:
                            traLenOverThreeTest+=traInfoTest[0]
                            # print('add')
                            totalErrorsTest+=traInfoTest[11]
                        if traInfoTest[12]!=-1:
                            endInNoActTest+=traInfoTest[12]
                            endOutNoActTest+=int(traInfoTest[12])^1
                        testTras[testIndex].clear()


        '''
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
        
        '''
        with open(testInfo,'a',encoding='UTF-8') as testInfoFile:
                testInfoFile.write('eposides:'+str(K+1)+' ave_eposides_rewards:'+str(round(np.mean(rewardsTest),2))+\
                ' end_reward:' +str(round(np.mean(endAveRewardTest),2))+'\n'+\
                ' end_in: '+str(endInTest)+' end_out: '+str(endOutTest)+' train_end_acc: '+str(round(endInTest/(endOutTest+endInTest),2))+\
                ' end_in_no_action: '+str(endInNoActTest)+' end_out_no_act: '+str(endOutNoActTest)+' acc:'+str(round(endInNoActTest/(endInNoActTest+endOutNoActTest),2))+'\n'+\
                ' in: '+str(totalInTest)+' _out:'+str(totalOutTest)+' train_ave_acc:'+str(round(totalInTest/(totalInTest+totalOutTest),2))+\
                ' in_no_act:'+str(inNoActTest)+' out_no_act:'+str(outNoActTest)+' acc:'+str(round(inNoActTest/(inNoActTest+outNoActTest),2))+'\n'+\
                ' len_tra_over_one: '+str(traLenOverOneTest)+' no_action_num:'+str(noActionNumTest)+' no_action_num_80:'+str(noActionNumWithThresholdTest)+\
                ' acc:'+str(round(noActionNumTest/traLenOverOneTest,2))+' acc2:'+str(round(noActionNumWithThresholdTest/traLenOverOneTest,2))+'\n'+\
                ' len_tra_over_three: '+str(traLenOverThreeTest)+' total_errors: '+str(totalErrorsTest)+' acc: '+str(round(totalErrorsTest/traLenOverThreeTest,2))+'\n')



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
    # else:
    #     actor_load='/home/wu_tian_ci/drl_project/mymodel/ddpg/pretrain_data/ddpg_train/1last_move5/ActorNet.pt'
    store_path=os.path.join('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/',store[0:-6],'offline',store[-6:])
    json_path=os.path.join('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/json_path/',store[0:-6],'offline',store[-6:])
    value_path=os.path.join('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/value_path/',store[0:-6],'offline',store[-6:])
    critic_loss_path=os.path.join('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/critic_loss/',store[0:-6],'offline',store[-6:])
    if not os.path.exists(json_path):
        os.makedirs(json_path)
    if not os.path.exists(value_path):
        os.makedirs(value_path)
    if not os.path.exists(critic_loss_path):
        os.makedirs(critic_loss_path)
    print(args.sup)

    envs=[]
    for i in range(2,22):
        if i<10:
            envs.append('0'+str(i))
        else:
            envs.append(str(i))
    random.seed(None)
    random.shuffle(envs)
    trainenvs=envs[:15]
    print(store[0:-6])
    print(store[-6:])
    testEnvs=envs[15:]
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    with open(os.path.join(store_path,'envsinfo.txt'),'w') as f:
        f.write('envs'+str(envs)+'\n trainEnvs:'+str(trainenvs)+'\n testEnvs:'+str(testEnvs)+'\n'+' lr:'+str(args.lr)+' preload: '+str(args.preload))
        if args.sup!='50':
            f.write('\n'+args.sup)
    train_epoch(agent, args.lr, 500, 256,device,args.mode,store_path=store_path,multienvs=trainenvs,testEnvs=testEnvs,\
                json_path=json_path,value_path=value_path,critic_loss_path=critic_loss_path)