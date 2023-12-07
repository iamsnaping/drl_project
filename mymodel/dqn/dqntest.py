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

        
def train_epoch(agent:REMAgent,device,testEnvs,store_path=None):
    agent.online.to(device)
    agent.target.to(device)
    testInfo=os.path.join(store_path,'testInfo.txt')
    f=open(testInfo,'w')
    f.write(str(testEnvs)+'\n')
    f.close()
    def callFunc(recordList,recordTimes,data):
        traLen=len(data)
        corects=0
        t=1
        if len(data)>=len(recordList):
            return
        recordList[traLen][0]+=1
        for d in data:
            if d==-1:
                continue
            corects+=d
            recordList[traLen][t]+=corects
            recordTimes[traLen][t]+=t
            t+=1

    def callFunc2(recordList,recordTimes,data):
        traLen=len(data)
        t=1
        if traLen>=len(recordList):
            return
        recordList[traLen][0]+=1
        for d in data:
            if d==-1:
                continue
            recordList[traLen][t]+=d
            recordTimes[traLen][t]+=1
            t+=1


    def strFunc(recordList,recordTimes,index):
        returnStr='len:'+str(recordList[index][0])+' '
        for i in range(1,index+1):
            returnStr+=str(i)+':'+str(recordList[index][i])+','+str(round(recordList[index][i]/recordTimes[index][i],2))+'\t'
        return returnStr
        

    def strFunc2(recordList,recordTimes,index):
        returnStr='len:'+str(recordList[index][0])+' '
        for i in range(1,index+1):
            returnStr+=str(i)+':'+str(recordList[index][i])+','+str(round(recordList[index][i]/recordTimes[index][i],2))+'\t'
        return returnStr
    

    recorder=DQNDataRecorder(20,20,callFunc,strFunc)
    recorder2=DQNDataRecorder(20,20,callFunc2,strFunc2)
    recorder3=DQNDataRecorder(20,20,callFunc2,strFunc2)
    # every step no action_corrects
    recorder4=DQNDataRecorder(20,20,callFunc2,strFunc2)
    testenvs=[]
    testNum=[]
    for env in testEnvs:
        testEnvPath=os.path.join('/home/wu_tian_ci/eyedata/seperate/',env,'1')
        testenvs.append(DQNRNNEnv(testEnvPath))
        testNum.append(float(env))
        testenvs[-1].shuffle=False


    for i in range(len(testEnvs)):
        testenvs[i].load_dict()
        testenvs[i].get_file()
    stopFlags=[False for i in range(len(testEnvs))]
    surviveFlags=len(testEnvs)



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
    testTras=[DQNRNNTrajectory() for i in range(surviveFlags)]
    testInOutList=[[0,0] for i in range(surviveFlags)]
    testInOutListNoAction=[[0,0] for i in range(surviveFlags)]
    testInOuEndtList=[[0,0] for i in range(surviveFlags)]
    testInOutEndListNoAction=[[0,0] for i in range(surviveFlags)]
    testTras=[DQNRNNTrajectory() for i in range(surviveFlags)]
    prTest=PresentsRecorder(len(testenvs))
    while surviveFlags>0:
        testEyeList=[]
        testClickList=[]
        testLastPList=[]
        testIndexesList=[]
        testLengthsList=[]
        testPersonList=[]
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
                click=torch.tensor(ans[1],dtype=torch.long).to(device)
                lengths=torch.tensor(ans[4],dtype=torch.long)
                person=torch.tensor([testNum[i]],dtype=torch.float32).to(device)
                if prTest.flag[i]==False:
                    prTest.init(i,ans[1])
                lastP=torch.tensor(prTest.getLastPresents(i),dtype=torch.long).to(device)
                testLastPList.append(lastP)
                testEyeList.append(eye)
                testClickList.append(click)
                testLengthsList.append(lengths)
                testPersonList.append(person)
                # ans -> eye click goal isEnd
                doneFlag=0
                if ans[3]==1:
                    doneFlag=0
                else:
                    doneFlag=1
                # print(ans[3])
                # click,eye,goal,action,mask
                testTras[i].push(click,eye,ans[2],0,doneFlag*UTIL.GAMMA,lastP,lengths,person)
                testIndexesList.append(i)
                    # print(clickList)
        if surviveFlags<=0:
            break
        testClickStack=torch.stack(testClickList,dim=0).to(device)
        testLastPStack=torch.stack(testLastPList,dim=0).to(device)
        testLengthStack=torch.stack(testLengthsList,dim=0)
        testEyeListStack=torch.stack(testEyeList,dim=0).to(device)
        testPersonListStack=torch.stack(testPersonList,dim=0).to(device).unsqueeze(1)
        testActions=agent.act(testClickStack,testEyeListStack,testLastPStack,testLengthStack,testPersonListStack)
        # print(f'actions {testActions}')
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
                # print(testInOutList)
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

    inOutList=[]
    endInOutList=[]
    inOutListNoAction=[]
    endInOutListNoAction=[]
    for i in range(len(testEnvs)):
        try:
            inOutList.append(testInOutList[i][0]/(testInOutList[i][0]+testInOutList[i][1]))
        except:
            inOutList.append(0)
        try:
            inOutListNoAction.append(testInOutListNoAction[i][0]/(testInOutListNoAction[i][0]+testInOutListNoAction[i][1]))
        except:
            inOutListNoAction.append(0)
        try:
            endInOutList.append(testInOuEndtList[i][0]/(testInOuEndtList[i][0]+testInOuEndtList[i][1]))
        except:
            endInOutList.append(0)
        try:
            endInOutListNoAction.append(testInOutEndListNoAction[i][0]/(testInOutEndListNoAction[i][0]+testInOutEndListNoAction[i][1]))
        except:
            # print('error')
            endInOutListNoAction.append(0)

    with open(testInfo,'a',encoding='UTF-8') as testInfoFile:
        for i in range(len(testEnvs)):
            testInfoFile.write('env: '+str(testEnvs[i])+' '+'ave_acc: '+str(inOutList[i])+' ave_end_acc: '+str(endInOutList[i])+'\n'+\
                    'ave_acc_noaction: '+str(inOutListNoAction[i])+' ave_end_acc_noaction: '+str(endInOutListNoAction[i])+'\n\n')

    # inOutMean=np.mean(inOutList)
    # endInOutMean=np.mean(endInOutList)
    # inOutNoActionMean=np.mean(inOutListNoAction)
    # endInOutNoActionMean=np.mean(endInOutListNoAction)
    # inOutList=np.array(inOutList)
    # endInOutList=np.array(endInOutList)
    # inOutListNoAction=np.array(inOutListNoAction)
    # endInOutListNoAction=np.array(endInOutListNoAction)
    # inOutStd=np.sqrt(np.mean((inOutList-inOutMean)**2))
    # endInOutStd=np.sqrt(np.mean((endInOutList-endInOutMean)**2))
    # inOutNoActionStd=np.sqrt(np.mean((inOutListNoAction-inOutNoActionMean)**2))
    # endInOutNoActionStd=np.sqrt(np.mean((endInOutListNoAction-endInOutNoActionMean)**2))
    # with open(testInfo,'a',encoding='UTF-8') as testInfoFile:
    #     testInfoFile.write('ave_eposides_rewards:'+str(round(np.mean(rewardsTest),2))+\
    #     ' end_reward:' +str(round(np.mean(endAveRewardTest),2))+'\n'+\
    #     ' end_in: '+str(endInTest)+' end_out: '+str(endOutTest)+' train_end_acc: '+str(round(endInTest/(endOutTest+endInTest),2))+\
    #     ' end_in_no_action: '+str(endInNoActTest)+' end_out_no_act: '+str(endOutNoActTest)+' acc:'+str(round(endInNoActTest/(endInNoActTest+endOutNoActTest),2))+'\n'+\
    #     ' in: '+str(totalInTest)+' _out:'+str(totalOutTest)+' train_ave_acc:'+str(round(totalInTest/(totalInTest+totalOutTest),2))+\
    #     ' in_no_act:'+str(inNoActTest)+' out_no_act:'+str(outNoActTest)+' acc:'+str(round(inNoActTest/(inNoActTest+outNoActTest),2))+'\n'+\
    #     ' len_tra_over_one: '+str(traLenOverOneTest)+' no_action_num:'+str(noActionNumTest)+' no_action_num_80:'+str(noActionNumWithThresholdTest)+\
    #     ' acc:'+str(round(noActionNumTest/traLenOverOneTest,2))+' acc2:'+str(round(noActionNumWithThresholdTest/traLenOverOneTest,2))+'\n'+\
    #     ' len_tra_over_three: '+str(traLenOverThreeTest)+\
    #     ' total_errors: '+str(totalErrorsTest)+' acc: '+\
    #     str(round(totalErrorsTest/traLenOverThreeTest,2))+'\n'+\
    #     ' inout_mean+std: '+str(round(inOutMean,2))+'+'+str(round(inOutStd,2))+\
    #     ' inout_no_action_mean+std: '+str(round(inOutNoActionMean,2))+'+'+str(round(inOutNoActionStd,2))+'\n'+\
    #     ' end_inout_mean+std: '+str(round(endInOutMean,2))+'+'+str(round(endInOutStd,2))+\
    #     ' end_inout_no_action_mean+std: '+str(round(endInOutNoActionMean,2))+'+'+str(round(endInOutNoActionStd,2))+'\n')






if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-cuda',type=str,default='cuda:1')
    parser.add_argument('-net',type=int,default=1)
    parser.add_argument('-sup',type=str,default='50')
    parser.add_argument('-layers',type=int,default=5)
    parser.add_argument('-embed',type=int,default=128)
    parser.add_argument('-rems',type=int,default=5)


    args=parser.parse_args()
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    store=UTIL.getTimeStamp()
# /home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20231030/trainall/201132/dqnnetoffline.pt
    envs=[]
    agent=REMAgent(device=device,rnn_layer=args.layers,embed_n=args.embed)
    for i in range(2,22):
        if i<10:
            env='0'+str(i)
        else:
            env=str(i)
        envs.append(env)
    # /home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20231030/trainall/192145/dqnnetoffline.pt
    agent.load('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20231130/trainall/164830/dqnnetoffline.pt')
    store_path=os.path.join('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/',store[0:-6],'testmore',store[-6:])
    if not os.path.exists(store_path):
        os.makedirs(store_path)
        with open(os.path.join(store_path,'envsinfo.txt'),'w') as f:
            f.write('envs'+str(envs)+'\n'\
                +'layers: '+str(args.layers))
        if args.sup!='50':
            f.write('\n'+args.sup)
    train_epoch(agent,device,store_path=store_path,testEnvs=envs)