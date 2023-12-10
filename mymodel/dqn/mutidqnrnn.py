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
        
def train_epoch(agent:REMAgent, lr,
                 epochs, batch_size,device,mode,multienvs,testEnvs,
                 topN=5,remBlocskNum=5,randLow=2,randHigh=17,randSize=5,skipFlag=False,
                 store_path=None,json_path=None,value_path=None,critic_loss_path=None):
    # random.seed(None)
    # np.random.seed(None)
    agent.online.to(device)
    agent.target.to(device)
    EPOSILON_DECAY=epochs
    EPOSILON_START=0.1
    EPOSILON_END=0.02
    # ebuffer=ExampleBuffer(2**17)
    trainer = torch.optim.Adam(lr=lr, params=agent.online.parameters())
    loss=nn.HuberLoss()
    agentBuffer=ReplayBufferRNN2(2**17,device)
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
    trainNum=[]
    testNum=[]
    # testScene=[]
    # trainScene=[]
    trainSceneNum=[]
    testSceneNum=[]
    for scene in range(1,5):
        for env in multienvs:
            envPath=os.path.join('/home/wu_tian_ci/eyedata/seperate/',env,str(scene))
            envs.append(DQNRNNEnv(envPath))
            envFlags.append(False)
            envPaths.append(envPath)
            trajectorys.append(DQNRNNTrajectory2())
            trainNum.append(int(env))
            trainSceneNum.append(scene)
    # envPath=os.path.join('/home/wu_tian_ci/eyedata/seperate/','22','1')
    # trainNum.append(22.0)
    # e=DQNRNNEnv(envPath)
    # e.topN=5
    # e.shuffle=False
    # envs.append(e)
    # trajectorys.append(DQNRNNTrajectory())
    for scene in range(1,5):
        for env in testEnvs:
            testEnvPath=os.path.join('/home/wu_tian_ci/eyedata/seperate/',env,str(scene))
            testenvs.append(DQNRNNEnv(testEnvPath))
            testNum.append(float(env))
            testenvs[-1].shuffle=False
            testenvs[-1].topN=topN
            testenvs[-1].eval=True
            envPath=os.path.join('/home/wu_tian_ci/eyedata/seperate/',env,'1')
            envs.append(DQNRNNEnv(envPath))
            envFlags.append(False)
            envPaths.append(envPath)
            trajectorys.append(DQNRNNTrajectory2())
            trainNum.append(int(env))
            envs[-1].shuffle=False
            envs[-1].topN=3
            testSceneNum.append(scene)
            trainSceneNum.append(scene)


    for i in range(len(envs)):
        envs[i].load_dict()
        envs[i].get_file()

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
        processReward=[]
        # 100
        agent.train()


        #scene
        trainEndInS=[0 for i in range(5)]
        trainEndOutS=[0 for i in range(5)]
        trainInS=[0 for i in range(5)]
        trainOutS=[0 for i in range(5)]
        trainErrorsS=[0 for i in range(5)]
        trainLenS=[0 for i in range(5)]

        testEndInS=[0 for i in range(5)]
        testEndOutS=[0 for i in range(5)]
        testInS=[0 for i in range(5)]
        testOutS=[0 for i in range(5)]
        testErrorsS=[0 for i in range(5)]
        testLenS=[0 for i in range(5)]

        for steps in range(500):
            eyeList=[]
            clickList=[]
            indexes=[]
            lastPList=[]
            lengthsList=[]
            personList=[]
            sceneList=[]
            # ans -> eye click goal isEnd length
            skipNum=np.random.randint(low=randLow,high=randHigh,size=randSize)
            for i in range(len(envs)):
                if skipFlag:
                    if i in skipNum:
                        continue
                ans=envs[i].act()
                if isinstance(ans,bool):
                    envs[i].load_dict()
                    envs[i].get_file()
                    beginFlags[i]=True
                    continue
                else:
                    # print(ans[0],len(ans[0]))
                    eye=torch.tensor(ans[0],dtype=torch.long)
                    click=torch.tensor(ans[1],dtype=torch.long).to(device)
                    lengths=torch.tensor(ans[4],dtype=torch.long)
                    person=torch.tensor([trainNum[i]],dtype=torch.long).to(device)
                    scene=torch.tensor([trainSceneNum[i]],dtype=torch.long).to(device)
                    if prTrain.flag[i]==False:
                        prTrain.init(i,ans[1])
                    lastP=torch.tensor(prTrain.getLastPresents(i),dtype=torch.long).to(device)
                    lastPList.append(lastP)
                    eyeList.append(eye)
                    clickList.append(click)
                    lengthsList.append(lengths)
                    personList.append(person)
                    sceneList.append(scene)
                    # ans -> eye click goal isEnd
                    doneFlag=0
                    if ans[3]==1:
                        doneFlag=0
                    else:
                        doneFlag=1
                    # print(ans[3])
                    # click,eye,goal,action,mask
                    trajectorys[i].push(click,eye,ans[2],0,doneFlag*UTIL.GAMMA,lastP,lengths,person,scene)
                    indexes.append(i)
                    if ans[3]==1:
                        beginFlags[i]=True
            clickCat=torch.stack(clickList,dim=0).to(device)
            lastPCat=torch.stack(lastPList,dim=0).to(device)
            lengthStack=torch.stack(lengthsList,dim=0)
            eyeStack=torch.stack(eyeList,dim=0).to(device)
            personStack=torch.stack(personList,dim=0).to(device).unsqueeze(1)
            sceneStack=torch.stack(sceneList,dim=0).to(device).unsqueeze(1) 
            actions=agent.act(clickCat,eyeStack,lastPCat,lengthStack,personStack,sceneStack)
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
                    trainInS[trainSceneNum[index]]+=traInfo[4]
                    trainOutS[trainSceneNum[index]]+=traInfo[5]



                    endIn+=traInfo[6]
                    endOut+=int(traInfo[6])^1

                    trainEndInS[trainSceneNum[index]]+=traInfo[6]
                    trainEndOutS[trainSceneNum[index]]+=int(traInfo[6])^1

                    rewards.append(traInfo[7])
                    endAveReward.append(traInfo[8])
                    processReward.append(traInfo[7]-traInfo[8])
                    inNoAct+=traInfo[9]
                    outNoAct+=traInfo[10]
                    if traInfo[0]>=3:
                        traLenOverThree+=traInfo[0]
                        totalErrors+=traInfo[11]
                        trainLenS[trainSceneNum[index]]+=traInfo[0]
                        trainErrorsS[trainSceneNum[index]]+=traInfo[11]


                    TDZeroList=trajectorys[index].getComTraZero()
                    if traInfo[12]!=-1:
                        endInNoAct+=traInfo[12]
                        endOutNoAct+=int(traInfo[12])^1
                    
                    agentBuffer.push(TDZeroList)
                    trajectorys[index].clear()
            dbatch_size=int((1+(agentBuffer.getRatio()))*batch_size)
            if (agentBuffer.holding>=dbatch_size):
                # print('train')
                if not is_training:
                    with open(reward_path,'a') as f:
                        f.write('begin to train\n')
                    is_training=True
                clickList,eyeList,lastPList,lengths,person,scene,actionList,rewardList,maskList,nclickList,neyeList,nlastPList,nlengths,nperson,nscene= agentBuffer.sample(dbatch_size)
                deltaP = torch.rand(remBlocskNum).to(device)
                deltaP =deltaP/ deltaP.sum()
                with torch.no_grad():
                    onlineValues=agent.online(clickList,eyeList,lastPList,lengths,person,scene)
                    onlineValues=sum(onlineValues)/len(onlineValues)
                    yAction=torch.argmax(onlineValues,dim=-1,keepdim=True)
                    targetValues=agent.target(nclickList,neyeList,nlastPList,nlengths,nperson,nscene)
                    for i in range(remBlocskNum):
                        targetValues[i]=targetValues[i]*deltaP[i]
                    targetValues=sum(targetValues)
                    y=targetValues.gather(dim=-1,index=yAction)*maskList+rewardList
                values=agent.online(clickList,eyeList,lastPList,lengths,person,scene)
                for i in range(remBlocskNum):
                    values[i]=values[i]*deltaP[i]
                values=sum(values)
                values=values.gather(dim=-1,index=actionList)
                l = loss(values, y)
                # if steps%100==0:
                #     print(f'loss {l}')
                #     for name,param in agent.online.named_parameters():
                #         if param.requires_grad and param.grad is not None:
                #             print(f'name {name} grad {param.grad}')

                trainer.zero_grad()
                l.backward()
                trainer.step()
                # clp.add(l.cpu().detach().numpy().tolist())
                if  steps%5==0 and steps!=0:
                    agent.update()
       
        for i in range(len(testenvs)):
            testenvs[i].load_dict()
            testenvs[i].get_file()
        stopFlags=[False for i in range(len(testenvs))]
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
        testTras=[DQNRNNTrajectory2() for i in range(surviveFlags)]
        testInOutList=[[0,0] for i in range(surviveFlags)]
        testInOutListNoAction=[[0,0] for i in range(surviveFlags)]
        testInOuEndtList=[[0,0] for i in range(surviveFlags)]
        testInOutEndListNoAction=[[0,0] for i in range(surviveFlags)]
        agent.eval()
        with torch.no_grad():
            while surviveFlags>0:
                testEyeList=[]
                testClickList=[]
                testLastPList=[]
                testIndexesList=[]
                testLengthsList=[]
                testPersonList=[]
                testSceneList=[]
                for i in range(len(testenvs)):
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
                        person=torch.tensor([testNum[i]],dtype=torch.long).to(device)
                        scene=torch.tensor([testSceneNum[i]],dtype=torch.long).to(device)

                        if prTest.flag[i]==False:
                            prTest.init(i,ans[1])
                        lastP=torch.tensor(prTest.getLastPresents(i),dtype=torch.long).to(device)
                        testLastPList.append(lastP)
                        testEyeList.append(eye)
                        testClickList.append(click)
                        testLengthsList.append(lengths)
                        testPersonList.append(person)
                        testSceneList.append(scene)
                        # ans -> eye click goal isEnd
                        doneFlag=0
                        if ans[3]==1:
                            doneFlag=0
                        else:
                            doneFlag=1
                        # print(ans[3])
                        # click,eye,goal,action,mask
                        testTras[i].push(click,eye,ans[2],0,doneFlag*UTIL.GAMMA,lastP,lengths,person,scene)
                        testIndexesList.append(i)
                            # print(clickList)
                if surviveFlags<=0:
                    break
                # print(testClickList,testLastPList,testLengthsList,surviveFlags)
                testClickStack=torch.stack(testClickList,dim=0).to(device)
                testLastPStack=torch.stack(testLastPList,dim=0).to(device)
                testLengthStack=torch.stack(testLengthsList,dim=0)
                testEyeListStack=torch.stack(testEyeList,dim=0).to(device)
                testPersonListStack=torch.stack(testPersonList,dim=0).to(device).unsqueeze(1)
                testSceneStack=torch.stack(testSceneList,dim=0).to(device).unsqueeze(1)       
                testActions=agent.act(testClickStack,testEyeListStack,testLastPStack,testLengthStack,testPersonListStack,testSceneStack)
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
                        testInS[testSceneNum[testIndex]]+=traInfoTest[4]
                        testOutS[testSceneNum[testIndex]]+=traInfoTest[5]

                        testInOutList[testIndex][0]+=traInfoTest[4]
                        testInOutList[testIndex][1]+=traInfoTest[5]
                        # print(testInOutList)
                        endInTest+=traInfoTest[6]
                        endOutTest+=int(traInfoTest[6])^1
                        testInOuEndtList[testIndex][0]+=traInfoTest[6]
                        testInOuEndtList[testIndex][1]+=int(traInfoTest[6])^1

                        testEndInS[testSceneNum[testIndex]]+=traInfoTest[6]
                        testEndOutS[testSceneNum[testIndex]]+=int(traInfoTest[6])^1



                        rewardsTest.append(traInfoTest[7])
                        endAveRewardTest.append(traInfoTest[8])
                        inNoActTest+=traInfoTest[9]
                        outNoActTest+=traInfoTest[10]
                        testInOutListNoAction[testIndex][0]+=traInfoTest[9]
                        testInOutListNoAction[testIndex][1]+=traInfoTest[10]
                        if traInfoTest[0]>=3:
                            traLenOverThreeTest+=traInfoTest[0]
                            totalErrorsTest+=traInfoTest[11]
                            testLenS[testSceneNum[testIndex]]+=traInfoTest[0]
                            testErrorsS[testSceneNum[testIndex]]+=traInfoTest[11]
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
        for i in range(len(testenvs)):
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
        trainErrorR='\nerror: '
        trainAccR='\nIPA1: '
        trainEndAccR='\nIPA2: '

        testErrorR='\nerror: '
        testAccR='\nIPA1: '
        testEndAccR='\nIPA2: '
        
        for i in range(1,5):
            testEndAccR+=str(i)+' '+str(round(testEndInS[i]/(testEndInS[i]+testEndOutS[i]) if (testEndInS[i]+testEndOutS[i])>0  else 0 ,2))+' '
            testAccR+=str(i)+' '+str(round(testInS[i]/(testInS[i]+testOutS[i]) if (testInS[i]+testOutS[i])>0  else 0 ,2))+' '
            testErrorR+=str(i)+' '+str(round(testErrorsS[i]/testLenS[i] if testLenS[i]>0  else 0 ,2))+' '
            
            trainEndAccR+=str(i)+' '+str(round(trainEndInS[i]/(trainEndInS[i]+trainEndOutS[i]) if (trainEndInS[i]+trainEndOutS[i])>0  else 0 ,2))+' '
            trainAccR+=str(i)+' '+str(round(trainInS[i]/(trainInS[i]+trainOutS[i]) if (trainInS[i]+trainOutS[i])>0  else 0 ,2))+' '
            trainErrorR+=str(i)+' '+str(round(trainErrorsS[i]/trainLenS[i] if trainLenS[i]>0  else 0 ,2))+' '
   

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
                ' len_tra_over_three: '+str(traLenOverThree)+' total_errors: '+str(totalErrors)+' acc: '+str(round(totalErrors/traLenOverThree,2))+\
                trainErrorR+trainAccR+trainEndAccR+'\n')
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
                ' end_inout_no_action_mean+std: '+str(round(endInOutNoActionMean,2))+'+'+str(round(endInOutNoActionStd,2))+\
            testErrorR+testAccR+testEndAccR+'\n')
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
                ' len_tra_over_three: '+str(traLenOverThree)+' total_errors: '+str(totalErrors)+' acc: '+str(round(totalErrors/traLenOverThree,2))+\
                trainErrorR+trainAccR+trainEndAccR+'\n')
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
            ' end_inout_no_action_mean+std: '+str(round(endInOutNoActionMean,2))+'+'+str(round(endInOutNoActionStd,2))+\
            testErrorR+testAccR+testEndAccR+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-modelPath',type=str,default='20231022203322')
    parser.add_argument('-cuda',type=str,default='cuda:1')
    parser.add_argument('-mode',type=int,default=1)
    parser.add_argument('-net',type=int,default=1)
    parser.add_argument('-sup',type=str,default='50')
    parser.add_argument('-preload',type=bool,default=False)
    parser.add_argument('-lr',type=float,default=0.0005)
    parser.add_argument('-layers',type=int,default=5)
    parser.add_argument('-embed',type=int,default=128)
    parser.add_argument('-rems',type=int,default=5)
    parser.add_argument('-topN',type=int,default=5)
    parser.add_argument('-skip',type=bool,default=False)
    parser.add_argument('-low',type=int,default=2)
    parser.add_argument('-high',type=int,default=15)
    parser.add_argument('-skipN',type=int,default=10)
    parser.add_argument('-epochs',type=int,default=500)
    parser.add_argument('-batchsize',type=int,default=256)

    args=parser.parse_args()
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    agent=REMAgent2(device=device,rnn_layer=args.layers,embed_n=args.embed)
    store=UTIL.getTimeStamp()

    if args.preload:
        # actor_load=os.path.join(,args.modelPath)
        # actor_load=os.path.join(actor_load,'dqnnetoffline.pt')
        actor_load='/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20231126/trainall/165620/dqnnetoffline.pt'
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
    
    b=[i for i in range(17,22)]
    a=[ i for i in range(2,17)]
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
        f.write('envs'+str(envs)+'\n trainEnvs:'+str(trainEnvs)+'\n testEnvs:'+str(testEnvs)+'\n'+' lr:'+str(args.lr)+' preload: '+str(args.preload)\
                +' topN:'+str(args.topN)+' skip:'+str(args.skip)+' skip_info:' +str(args.low)+' '+str(args.high)+' '+str(args.skipN)+'\n'\
                +'layers: '+str(args.layers))
        if args.sup!='50':
            f.write('\n'+args.sup)
    train_epoch(agent, args.lr, args.epochs, args.batchsize,device,args.mode,store_path=store_path,multienvs=trainEnvs,testEnvs=testEnvs,\
                json_path=json_path,value_path=value_path,critic_loss_path=critic_loss_path,remBlocskNum=args.rems,topN=args.topN,\
                randLow=args.low,randHigh=args.high,randSize=args.skipN,skipFlag=args.skip)