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
class PresentsRecorder(object):

    def __init__(self,num) -> None:
        self.num=num
        self.recorder=[[] for i in range(num)]
        self.flag=[False for i in range(num)]
    
    # init with last3goal
    def init(self,index,last3goal):
        self.recorder[index]=dp(last3goal)



    def add(self,index,num):
        for i in range(2):
            self.recorder[index][i]=self.recorder[index][i+1]
        if num!=12:
            self.recorder[index][2]=num


    def getLastPresents(self,index):
        return dp(self.recorder[index])

        
def train_epoch(agent:Agent, lr, epochs, batch_size,device,mode,multienvs,testEnvs,store_path=None,json_path=None,value_path=None,critic_loss_path=None):
    # random.seed(None)
    # np.random.seed(None)
    agent.online.to(device)
    agent.target.to(device)
    EPOSILON_DECAY=epochs
    EPOSILON_START=0.1
    EPOSILON_END=0.02
    # ebuffer=ExampleBuffer(2**17)
    trainer = torch.optim.Adam(lr=lr, params=agent.online.parameters())
    loss=nn.MSELoss()
    agentBuffer=ReplayBufferRNN(2**17,device)
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
        envs.append(DQNRNNEnv(envPath))
        envFlags.append(False)
        envPaths.append(envPath)
        trajectorys.append(DQNRNNTrajectory())
    trajectorys.append(DQNRNNTrajectory())
    for i in range(len(envs)):
        envs[i].load_dict()
        envs[i].get_file()
    for env in testEnvs:
        testEnvPath=os.path.join('/home/wu_tian_ci/eyedata/seperate/',env,'1')
        testenvs.append(DQNRNNEnv(testEnvPath))

    beginFlags=[True for i in range(len(envs))]

    prTrain=PresentsRecorder(len(envs))
    prTest=PresentsRecorder(len(testenvs))
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
        # 100
        for steps in range(750):
            eyeList=[]
            clickList=[]
            indexes=[]
            lastPList=[]
            lengthsList=[]
            # ans -> eye click goal isEnd
            for i in range(len(envs)):
                ans=envs[i].act()
                if isinstance(ans,bool):
                    envs[i].load_dict()
                    envs[i].get_file()
                    beginFlags[i]=True
                    continue
                else:
                    # print(ans[0],len(ans[0]))
                    eye=torch.tensor(ans[0],dtype=torch.long)
                    click=torch.tensor([ans[1]],dtype=torch.long).to(device)
                    lengths=torch.tensor(ans[4],dtype=torch.long)
                    if prTrain.flag[i]==False:
                        prTrain.init(i,ans[1])
                    lastP=torch.tensor([prTrain.getLastPresents(i)],dtype=torch.long).to(device)
                    lastPList.append(lastP)
                    eyeList.append(eye)
                    clickList.append(click)
                    lengthsList.append(lengths)
                    # ans -> eye click goal isEnd
                    doneFlag=0
                    if ans[3]==1:
                        doneFlag=0
                    else:
                        doneFlag=1
                    # print(ans[3])
                    # click,eye,goal,action,mask
                    trajectorys[i].push(click,eye,ans[2],0,doneFlag*UTIL.GAMMA,lastP,lengths)
                    indexes.append(i)
                    if ans[3]==1:
                        beginFlags[i]=True
                        # print(clickList)
            clickCat=torch.cat(clickList,dim=0).to(device)
            lastPCat=torch.cat(lastPList,dim=0).to(device)
            lengthStack=torch.stack(lengthsList,dim=0)
            eyeList=torch.stack(eyeList,dim=0).to(device)
            actions=agent.act(clickCat,eyeList,lastPCat,lengthStack)
            # print(f' aaaaa {actions}')
            for actS,index in zip(actions,indexes):
                prTrain.add(index,actS)
                eposilon=np.interp(K,[0,EPOSILON_DECAY],[EPOSILON_START,EPOSILON_END])
                prob=random.random()
                if prob>eposilon:
                    trajectorys[index].tras[-1][3]=dp(actS)
                else:
                    trajectorys[index].tras[-1][3]=random.randint(0,12)
                if trajectorys[index].tras[-1][4]==0:
                    trajectorys[index].getNewTras()
                    # 0:traLen
                    # 1:noActionNum
                    # 2:noActionNumWithThreshold
                    # 3:lastWithNoAction
                    # 4:_in
                    # 5:_out
                    # 6:inOut
                    # 7:reward
                    # 8:endRewards
                    # 9:inNoAct
                    # 10:outNoAct
                    # 11:errors
                    # 12:endInOutNoAct
                    traInfo=trajectorys[index].getInfo()
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
                    TDZeroList=trajectorys[index].getComTraZero()
                    if traInfo[12]!=-1:
                        endInNoAct+=traInfo[12]
                        endOutNoAct+=int(traInfo[12])^1
                    
                    agentBuffer.push(TDZeroList)

     
                    trajectorys[index].clear()
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
                clp.add(l.cpu().detach().numpy().tolist())
                if  steps%10==0 and steps!=0:
                    agent.update()
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
        testInOutList=[[0,0] for i in range(surviveFlags)]
        testInOutListNoAction=[[0,0] for i in range(surviveFlags)]
        testInOuEndtList=[[0,0] for i in range(surviveFlags)]
        testInOutEndListNoAction=[[0,0] for i in range(surviveFlags)]
        with torch.no_grad():
            while surviveFlags>0:
                testEyeList=[]
                testClickList=[]
                testLastPList=[]
                testIndexesList=[]
                testLengthsList=[]
                for i in range(len(testEnvs)):
                    if stopFlags[i]:
                        continue
                    ans=testenvs[i].act()
                    if isinstance(ans,bool):
                        stopFlags[i]=True
                        surviveFlags-=1
                        continue
                    else:
                                           # print(ans[0],len(ans[0]))
                        eye=torch.tensor(ans[0],dtype=torch.long)
                        click=torch.tensor([ans[1]],dtype=torch.long).to(device)
                        lengths=torch.tensor(ans[4],dtype=torch.long)
                        if prTest.flag[i]==False:
                            prTest.init(i,ans[1])
                        lastP=torch.tensor([prTrain.getLastPresents(i)],dtype=torch.long).to(device)
                        testLastPList.append(lastP)
                        testEyeList.append(eye)
                        testClickList.append(click)
                        testLengthsList.append(lengths)
                        # ans -> eye click goal isEnd
                        doneFlag=0
                        if ans[3]==1:
                            doneFlag=0
                        else:
                            doneFlag=1
                        # print(ans[3])
                        # click,eye,goal,action,mask
                        trajectorys[i].push(click,eye,ans[2],0,doneFlag*UTIL.GAMMA,lastP,lengths)
                        indexes.append(i)
                        if ans[3]==1:
                            beginFlags[i]=True
                            # print(clickList)
                if surviveFlags<=0:
                    break
                testClickCat=torch.cat(testClickList,dim=0).to(device)
                testLastPCat=torch.cat(lastPList,dim=0).to(device)
                testLengthStack=torch.stack(lengthsList,dim=0)
                testEyeList=torch.stack(eyeList,dim=0).to(device)
                testActions=agent.act(clickCat,eyeList,lastPCat,lengthStack)
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
                        testInOutList[testIndex][0]+=traInfoTest[4]
                        testInOutList[testIndex][1]+=traInfoTest[5]
                        endInTest+=traInfoTest[6]
                        endOutTest+=int(traInfoTest[6])^1
                        testInOuEndtList[testIndex][0]+=traInfoTest[6]
                        testInOuEndtList[testIndex][1]+=int(traInfoTest[6])^1
                        rewardsTest.append(traInfoTest[7])
                        endAveRewardTest.append(traInfoTest[8])
                        inNoActTest+=traInfoTest[9]
                        outNoActTest+=traInfoTest[10]
                        testInOutListNoAction[testIndex][0]+=traInfoTest[9]
                        testInOutListNoAction[testIndex][1]+=traInfoTest[10]
                        if traInfoTest[0]>=3:
                            traLenOverThreeTest+=traInfoTest[0]
                            totalErrorsTest+=traInfoTest[11]
                        if traInfoTest[12]!=-1:
                            endInNoActTest+=traInfoTest[12]
                            endOutNoActTest+=int(traInfoTest[12])^1
                            testInOutEndListNoAction[testIndex][0]+=traInfoTest[12]
                            testInOutEndListNoAction[testIndex][1]+=int(traInfoTest[12])^1
                        testTras[testIndex].clear()

        t_reward_len+=1
        if t_reward_len>=1000:
            t_reward_len%=1000
            t_reward[t_reward_len]=np.mean(rewards)
            t_end_reward[t_reward_len]=np.mean(endAveReward)
        else:
            t_reward.append(np.mean(rewards))
            t_end_reward.append(np.mean(endAveReward))


        inOutList=[]
        endInOutList=[]
        inOutListNoAction=[]
        endInOutListNoAction=[]
        for i in range(len(testEnvs)):
            inOutList.append(testInOutList[i][0]/(testInOutList[i][0]+testInOutList[i][1]))
            inOutListNoAction.append(testInOutListNoAction[i][0]/(testInOutListNoAction[i][0]+testInOutListNoAction[i][1]))
            endInOutList.append(testInOuEndtList[i][0]/(testInOuEndtList[i][0]+testInOuEndtList[i][1]))
            endInOutListNoAction.append(testInOutEndListNoAction[i][0]/(testInOutEndListNoAction[i][0]+testInOutEndListNoAction[i][1]))
        inOutMean=np.mean(inOutList)
        endInOutMean=np.mean(endInOutList)
        inOutNoActionMean=np.mean(inOutListNoAction)
        endInOutNoActionMean=np.mean(endInOutListNoAction)
        inOutList=np.array(inOutList)
        endInOutList=np.array(endInOutList)
        inOutListNoAction=np.array(inOutListNoAction)
        endInOutListNoAction=np.array(endInOutListNoAction)
        inOutStd=np.sqrt(np.mean((inOutList-inOutMean)**2))
        endInOutStd=np.sqrt(np.mean((endInOutList-endInOutMean)**2))
        inOutNoActionStd=np.sqrt(np.mean((inOutListNoAction-inOutNoActionMean)**2))
        endInOutNoActionStd=np.sqrt(np.mean((endInOutListNoAction-endInOutNoActionMean)**2))


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

        if len(rewardsTest)>0 and np.mean(rewardsTest)>best_scores and is_training:
            torch.save(agent.target.state_dict(), m_store_path_a)
            best_scores=np.mean(rewardsTest)
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
            with open(updateTestInfo,'a',encoding='UTF-8') as updateTestInfoFile:
                updateTestInfoFile.write('eposides:'+str(K+1)+' ave_eposides_rewards:'+str(round(np.mean(rewardsTest),2))+\
                ' end_reward:' +str(round(np.mean(endAveRewardTest),2))+'\n'+\
                ' end_in: '+str(endInTest)+' end_out: '+str(endOutTest)+' train_end_acc: '+str(round(endInTest/(endOutTest+endInTest),2))+\
                ' end_in_no_action: '+str(endInNoActTest)+' end_out_no_act: '+str(endOutNoActTest)+' acc:'+str(round(endInNoActTest/(endInNoActTest+endOutNoActTest),2))+'\n'+\
                ' in: '+str(totalInTest)+' _out:'+str(totalOutTest)+' train_ave_acc:'+str(round(totalInTest/(totalInTest+totalOutTest),2))+\
                ' in_no_act:'+str(inNoActTest)+' out_no_act:'+str(outNoActTest)+' acc:'+str(round(inNoActTest/(inNoActTest+outNoActTest),2))+'\n'+\
                ' len_tra_over_one: '+str(traLenOverOneTest)+' no_action_num:'+str(noActionNumTest)+' no_action_num_80:'+str(noActionNumWithThresholdTest)+\
                ' acc:'+str(round(noActionNumTest/traLenOverOneTest,2))+' acc2:'+str(round(noActionNumWithThresholdTest/traLenOverOneTest,2))+'\n'+\
                ' len_tra_over_three: '+str(traLenOverThreeTest)+\
                ' total_errors: '+str(totalErrorsTest)+' acc: '+\
                str(round(totalErrorsTest/traLenOverThreeTest,2))+'\n'+\
                ' inout_mean+std: '+str(round(inOutMean,2))+'+'+str(round(inOutStd,2))+\
                ' inout_no_action_mean+std: '+str(round(inOutNoActionMean,2))+'+'+str(round(inOutNoActionStd,2))+'\n'+\
                ' end_inout_mean+std: '+str(round(endInOutMean,2))+'+'+str(round(endInOutStd,2))+\
                ' end_inout_no_action_mean+std: '+str(round(endInOutNoActionMean,2))+'+'+str(round(endInOutNoActionStd,2))+'\n')
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
        with open(testInfo,'a',encoding='UTF-8') as testInfoFile:
            testInfoFile.write('eposides:'+str(K+1)+' ave_eposides_rewards:'+str(round(np.mean(rewardsTest),2))+\
            ' end_reward:' +str(round(np.mean(endAveRewardTest),2))+'\n'+\
            ' end_in: '+str(endInTest)+' end_out: '+str(endOutTest)+' train_end_acc: '+str(round(endInTest/(endOutTest+endInTest),2))+\
            ' end_in_no_action: '+str(endInNoActTest)+' end_out_no_act: '+str(endOutNoActTest)+' acc:'+str(round(endInNoActTest/(endInNoActTest+endOutNoActTest),2))+'\n'+\
            ' in: '+str(totalInTest)+' _out:'+str(totalOutTest)+' train_ave_acc:'+str(round(totalInTest/(totalInTest+totalOutTest),2))+\
            ' in_no_act:'+str(inNoActTest)+' out_no_act:'+str(outNoActTest)+' acc:'+str(round(inNoActTest/(inNoActTest+outNoActTest),2))+'\n'+\
            ' len_tra_over_one: '+str(traLenOverOneTest)+' no_action_num:'+str(noActionNumTest)+' no_action_num_80:'+str(noActionNumWithThresholdTest)+\
            ' acc:'+str(round(noActionNumTest/traLenOverOneTest,2))+' acc2:'+str(round(noActionNumWithThresholdTest/traLenOverOneTest,2))+'\n'+\
            ' len_tra_over_three: '+str(traLenOverThreeTest)+\
            ' total_errors: '+str(totalErrorsTest)+' acc: '+\
            str(round(totalErrorsTest/traLenOverThreeTest,2))+'\n'+\
            ' inout_mean+std: '+str(round(inOutMean,2))+'+'+str(round(inOutStd,2))+\
            ' inout_no_action_mean+std: '+str(round(inOutNoActionMean,2))+'+'+str(round(inOutNoActionStd,2))+'\n'+\
            ' end_inout_mean+std: '+str(round(endInOutMean,2))+'+'+str(round(endInOutStd,2))+\
            ' end_inout_no_action_mean+std: '+str(round(endInOutNoActionMean,2))+'+'+str(round(endInOutNoActionStd,2))+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-modelPath',type=str,default='20231022203322')
    parser.add_argument('-cuda',type=str,default='cuda:1')
    parser.add_argument('-mode',type=int,default=1)
    parser.add_argument('-net',type=int,default=1)
    parser.add_argument('-sup',type=str,default='50')
    parser.add_argument('-preload',type=bool,default=False)
    parser.add_argument('-lr',type=float,default=0.05)

    args=parser.parse_args()
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    agent=RNNAgent(device=device)
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
    # random.seed(3407) 
    random.shuffle(envs)
    # random.seed(None)
    # trainEnvs=envs[:15]
    # trainenvs=['19', '16', '17', '10', '11', '03', '07', '08', '15', '02', '04', '14', '06', '09', '21']
    # testEnvs=['20', '18', '13', '12', '05']
    print(store[0:-6])
    print(store[-6:])
    # testEnvs=envs[15:]
    trainEnvs=[]
    testEnvs=[]
    
    b=[6,9, 13, 16,14, 17]
    a=[ 2,4,7,8,11,12,21,3, 5 ,10 ,15 ,18 ,19,20]
    for num in a:
        if num>=10:
            trainEnvs.append(str(num))
        else:
            trainEnvs.append('0'+str(num))
    for num in b:
        if num>=10:
            testEnvs.append(str(num))
        else:
            testEnvs.append('0'+str(num))
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    with open(os.path.join(store_path,'envsinfo.txt'),'w') as f:
        f.write('envs'+str(envs)+'\n trainEnvs:'+str(trainEnvs)+'\n testEnvs:'+str(testEnvs)+'\n'+' lr:'+str(args.lr)+' preload: '+str(args.preload))
        if args.sup!='50':
            f.write('\n'+args.sup)
    train_epoch(agent, args.lr, 500, 256,device,args.mode,store_path=store_path,multienvs=trainEnvs,testEnvs=testEnvs,\
                json_path=json_path,value_path=value_path,critic_loss_path=critic_loss_path)