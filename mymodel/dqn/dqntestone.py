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

        
def train_epoch(agent:Agent,device,testEnvPath,store_path=None):
    agent.online.to(device)
    agent.target.to(device)
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    testInfo=os.path.join(store_path,'testInfo.txt')
    f=open(testInfo,'w')
    f.write(str(testEnvPath)+'\n')
    f.close()
    testEnvP=os.path.join('/home/wu_tian_ci/eyedata/seperate/',testEnvPath,'1')
    testenvs=DQNEnv(testEnvP)

    testRecorder=0
    testenvs.load_dict()
    testenvs.get_file()
    #ans -> state last_goal_state last_goal goal is_end
    rewardsTest=[]
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
    testInOutList=[0,0]
    testTras=DQNTrajectory()
    testInOutListNoAction=[0,0]
    testInOuEndtList=[0,0]
    testInOutEndListNoAction=[0,0]
    while True:
        ans=testenvs.act()
        if isinstance(ans,bool):
            break
        else:
            eye=torch.tensor([ans[0]],dtype=torch.int64).reshape((1,1,3)).to(device)
            click=torch.tensor([ans[1]],dtype=torch.int64).reshape((1,1,3)).to(device)
            index=torch.tensor([testRecorder for i in range(3)],dtype=torch.float32).reshape((1,3,1)).to(device)
            doneFlag=0
            action=agent.act(click,eye,index)
            testRecorder+=1
            if ans[3]==0:
                doneFlag=1
                testTras.push(ans[1],ans[0],ans[2],action,doneFlag*UTIL.GAMMA)
            else:
                testRecorder=0
                doneFlag=0
                testTras.push(ans[1],ans[0],ans[2],action,doneFlag*UTIL.GAMMA)

                testTras.getNewTras()

                traInfoTest=testTras.getInfo()
                if traInfoTest[0]>1:
                    traLenOverOneTest+=traInfoTest[0]
                    noActionNumTest+=traInfoTest[1]
                    noActionNumWithThresholdTest+=traInfoTest[2]
                lastWithNoActionTest+=traInfoTest[3]
                totalInTest+=traInfoTest[4]
                totalOutTest+=traInfoTest[5]
                testInOutList[0]+=traInfoTest[4]
                testInOutList[1]+=traInfoTest[5]
                endInTest+=traInfoTest[6]
                endOutTest+=int(traInfoTest[6])^1
                testInOuEndtList[0]+=traInfoTest[6]
                testInOuEndtList[1]+=int(traInfoTest[6])^1
                rewardsTest.append(traInfoTest[7])
                endAveRewardTest.append(traInfoTest[8])
                inNoActTest+=traInfoTest[9]
                outNoActTest+=traInfoTest[10]
                testInOutListNoAction[0]+=traInfoTest[9]
                testInOutListNoAction[1]+=traInfoTest[10]
                if traInfoTest[0]>=3:
                    traLenOverThreeTest+=traInfoTest[0]
                    totalErrorsTest+=traInfoTest[11]
                if traInfoTest[12]!=-1:
                    endInNoActTest+=traInfoTest[12]
                    endOutNoActTest+=int(traInfoTest[12])^1
                    testInOutEndListNoAction[0]+=traInfoTest[12]
                    testInOutEndListNoAction[1]+=int(traInfoTest[12])^1
                testTras.clear()

    with open(testInfo,'a',encoding='UTF-8') as testInfoFile:
        testInfoFile.write(' ave_eposides_rewards:'+str(round(np.mean(rewardsTest),2))+\
        ' end_reward:' +str(round(np.mean(endAveRewardTest),2))+'\n'+\
        ' end_in: '+str(endInTest)+' end_out: '+str(endOutTest)+' train_end_acc: '+str(round(endInTest/(endOutTest+endInTest),2))+\
        ' end_in_no_action: '+str(endInNoActTest)+' end_out_no_act: '+str(endOutNoActTest)+' acc:'+str(round(endInNoActTest/(endInNoActTest+endOutNoActTest),2))+'\n'+\
        ' in: '+str(totalInTest)+' _out:'+str(totalOutTest)+' train_ave_acc:'+str(round(totalInTest/(totalInTest+totalOutTest),2))+\
        ' in_no_act:'+str(inNoActTest)+' out_no_act:'+str(outNoActTest)+' acc:'+str(round(inNoActTest/(inNoActTest+outNoActTest),2))+'\n'+\
        ' len_tra_over_one: '+str(traLenOverOneTest)+' no_action_num:'+str(noActionNumTest)+' no_action_num_80:'+str(noActionNumWithThresholdTest)+\
        ' acc:'+str(round(noActionNumTest/traLenOverOneTest,2))+' acc2:'+str(round(noActionNumWithThresholdTest/traLenOverOneTest,2))+'\n'+\
        ' len_tra_over_three: '+str(traLenOverThreeTest)+\
        ' total_errors: '+str(totalErrorsTest)+' acc: '+\
        str(round(totalErrorsTest/traLenOverThreeTest,2))+'\n')




if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-modelPath',type=str,default='204558')
    parser.add_argument('-cuda',type=str,default='cuda:0')
    parser.add_argument('-mode',type=int,default=1)
    parser.add_argument('-net',type=int,default=1)
    parser.add_argument('-sup',type=str,default='50')
    parser.add_argument('-preload',type=bool,default=True)
    parser.add_argument('-lr',type=float,default=0.05)

    args=parser.parse_args()
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    store=UTIL.getTimeStamp()
# /home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20231028/offline/204558/dqnnetoffline.pt

    for i in range(2,22):
        if i<10:
            env='0'+str(i)
        else:
            env=str(i)
        agent=Agent(device=device)
        if args.preload:
            actor_load=os.path.join('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20231028/offline/',args.modelPath)
            actor_load=os.path.join(actor_load,'dqnnetoffline.pt')
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
        print(f'env: {env}')
        train_epoch(agent,device,store_path=store_path,testEnvPath=env)