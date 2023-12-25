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
import warnings
warnings.filterwarnings("error") 



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
        for item in self.seq_num.items():
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

def testFun(agent,testAllEnvs,testAllNum,testAllScene,thr):

    preTest=PresentsRecorder(len(testAllEnvs))
    stopFlags=[False for i in range(len(testAllEnvs))]
    surviveFlags=len(testAllEnvs)
    if thr:
        numFlag=surviveFlags//3
    else:
        numFlag=surviveFlags//4
    testTras=[DQNRNNTrajectory2() for i in range(surviveFlags)]
    testEndInS=[[0 for i in range(5)] for i in range(numFlag)]
    testEndOutS=[[0 for i in range(5)] for i in range(numFlag)]
    testInS=[[0 for i in range(5)] for i in range(numFlag)]
    testOutS=[[0 for i in range(5)] for i in range(numFlag)]
    testErrorsS=[[0 for i in range(5)] for i in range(numFlag)]
    testLenS=[[0 for i in range(5)] for i in range(numFlag)]
    agent.eval()

    for i in range(len(testAllEnvs)):
        testAllEnvs[i].load_dict()
        testAllEnvs[i].get_file()

    while surviveFlags>0:
        testEyeList=[]
        testClickList=[]
        testLastPList=[]
        testIndexesList=[]
        testLengthsList=[]
        testPersonList=[]
        testSceneList=[]
        for i in range(len(testAllEnvs)):
            if stopFlags[i]:
                continue
            ans=testAllEnvs[i].act()
            if isinstance(ans,bool):
                stopFlags[i]=True
                surviveFlags-=1
                continue
            else:

                eye=torch.tensor(ans[0],dtype=torch.long)
                click=torch.tensor(ans[1],dtype=torch.long).to(device)
                lengths=torch.tensor(ans[4],dtype=torch.long)
                person=torch.tensor([testAllNum[i]],dtype=torch.long).to(device)
                scene=torch.tensor([testAllScene[i]],dtype=torch.long).to(device)

                if preTest.flag[i]==False:
                    preTest.init(i,ans[1])
                lastP=torch.tensor(preTest.getLastPresents(i),dtype=torch.long).to(device)
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

                # click,eye,goal,action,mask
                testTras[i].push(click,eye,ans[2],0,doneFlag*UTIL.GAMMA,lastP,lengths,person,scene)
                testIndexesList.append(i)

        if surviveFlags<=0:
            break

        testClickStack=torch.stack(testClickList,dim=0).to(device)
        testLastPStack=torch.stack(testLastPList,dim=0).to(device)
        testLengthStack=torch.stack(testLengthsList,dim=0)
        testEyeListStack=torch.stack(testEyeList,dim=0).to(device)
        testPersonListStack=torch.stack(testPersonList,dim=0).to(device).unsqueeze(1)
        testSceneStack=torch.stack(testSceneList,dim=0).to(device).unsqueeze(1)       
        testActions=agent.act(testClickStack,testEyeListStack,testLastPStack,testLengthStack,testPersonListStack,testSceneStack)
        if testActions.shape==():
            testActions=[testActions]
        for testAction,testIndex in zip(testActions,testIndexesList):
            testTras[testIndex].tras[-1][3]=dp(testAction)
            if testTras[testIndex].tras[-1][4]==0:
                testTras[testIndex].getNewTras()
                traInfoTest=testTras[testIndex].getInfo()
                testInS[testAllNum[testIndex]-np.min(testAllNum)][testAllScene[testIndex]]+=traInfoTest[4]
                testOutS[testAllNum[testIndex]-np.min(testAllNum)][testAllScene[testIndex]]+=traInfoTest[5]
                testEndInS[testAllNum[testIndex]-np.min(testAllNum)][testAllScene[testIndex]]+=traInfoTest[6]
                testEndOutS[testAllNum[testIndex]-np.min(testAllNum)][testAllScene[testIndex]]+=int(traInfoTest[6])^1
                if traInfoTest[0]>=3:
                    testLenS[testAllNum[testIndex]-np.min(testAllNum)][testAllScene[testIndex]]+=traInfoTest[0]
                    testErrorsS[testAllNum[testIndex]-np.min(testAllNum)][testAllScene[testIndex]]+=traInfoTest[11]
                testTras[testIndex].clear()
    return testEndInS,testEndOutS,testInS,testOutS,testErrorsS,testLenS
        


def train_epoch(agent:REMAgent, lr,
                 epochs, batch_size,device,mode,multienvs,testEnvs,
                 topN=5,remBlocskNum=5,randLow=2,randHigh=17,randSize=5,skipFlag=False,
                 store_path=None,addPre=True,iterationFlag=True,load=False,loadPath='',restrict=False,thr=False):
    agent.online.to(device)
    agent.target.to(device)
    EPOSILON_DECAY=epochs
    EPOSILON_START=0.1
    EPOSILON_END=0.02
    # ebuffer=ExampleBuffer(2**17)
    trainer = torch.optim.Adam(lr=lr, params=agent.online.parameters())
    loss=nn.HuberLoss()

    if not os.path.exists(store_path):
        os.makedirs(store_path)
    m_store_path_a=os.path.join(store_path,'dqnnetoffline.pt')
    reward_path=os.path.join(store_path,'reward.txt')
    updataInfo=os.path.join(store_path,'updateInfo.txt')
    testInfo=os.path.join(store_path,'testInfo.txt')
    updateTestInfo=os.path.join(store_path,'updateTestInfo.txt')
    preTestInfos=[]
    env_str=''
    for e in multienvs:
        env_str+=e+' '
    f=open(reward_path,'w')
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
    for t in testEnvs:
        preTestInfo=os.path.join(store_path,t+'preAfterTest.txt')
        preTestInfos.append(preTestInfo)
        f=open(preTestInfo,'w')
        f.write(env_str+' '+t+'\n')
        f.close()
    agentBuffer=ReplayBufferRNN2(2**17,device)

    # json_path='/home/wu_tian_ci/drl_project/mymodel/ddpg/pretrain_data/json_path'
    # envs 
    envs=[]
    envFlags=[]
    envPaths=[]
    trajectorys=[]
    testenvs=[]
    trainNum=[]
    testNum=[]
    trainSceneNum=[]
    testSceneNum=[]
    if addPre==True:
        for scene in range(1,5):
            if thr and scene==3:
                continue
            for env in multienvs:
                envPath=os.path.join('/home/wu_tian_ci/eyedata/seperate/',env,str(scene))
                envs.append(DQNRNNEnv(envPath),restrict=restrict)
                envFlags.append(False)
                envPaths.append(envPath)
                trajectorys.append(DQNRNNTrajectory2())
                trainNum.append(int(env))
                trainSceneNum.append(scene)



    testAllEnvs=[]
    testAllNum=[]
    testAllScene=[]

    for scene in range(1,5):
        if thr and scene==3:
            continue
        for env in testEnvs:
            testEnvPath=os.path.join('/home/wu_tian_ci/eyedata/seperate/',env,str(scene))
            testAllEnvs.append(DQNRNNEnv(testEnvPath,num=int(env),scene=int(scene),restrict=restrict))
            testAllNum.append(int(env))
            testAllEnvs[-1].shuffle=False
            testAllEnvs[-1].topN=topN
            testAllEnvs[-1].eval=True
            testAllScene.append(scene)


    for scene in range(1,5):
        if thr and scene==3:
            continue
        for env in testEnvs:
            testEnvPath=os.path.join('/home/wu_tian_ci/eyedata/seperate/',env,str(scene))
            testenvs.append(DQNRNNEnv(testEnvPath,restrict=restrict))
            testNum.append(int(env))
            testenvs[-1].shuffle=False
            testenvs[-1].topN=topN
            testenvs[-1].eval=True
            testSceneNum.append(scene)

    for i in range(len(envs)):
        envs[i].load_dict()
        envs[i].get_file()

    prTrain=PresentsRecorder(len(envs))
    prTest=PresentsRecorder(len(testenvs))


    for tt in testEnvs:
        envBestModelPath=os.path.join(store_path,tt)
        if not os.path.exists(envBestModelPath):
            os.makedirs(envBestModelPath)
        envBestModel=os.path.join(envBestModelPath,'best.pt')
        envBestModelRecord=os.path.join(envBestModelPath,'record.txt')
        f=open(envBestModelRecord,'w')
        f.write('begin to train env '+tt+'\n')
        f.close()
        agentBuffer.refresh()
       # pre-test

        t_reward=[]
        t_reward_len=-1
        best_scores=-np.inf
        is_training=False

        if iterationFlag==False:
            envs=[]
            envFlags=[]
            trajectorys=[]
            trainNum=[]
            trainSceneNum=[]
            agent.load(loadPath)
        for scene in range(1,5):
            if thr and scene==3:
                continue
            testEnvPath=os.path.join('/home/wu_tian_ci/eyedata/seperate/',tt,str(scene))
            envs.append(DQNRNNEnv(testEnvPath,restrict=restrict))
            envFlags.append(False)
            trajectorys.append(DQNRNNTrajectory2())
            trainNum.append(int(tt))
            envs[-1].shuffle=False
            envs[-1].topN=topN
            trainSceneNum.append(scene)
            prTrain.addRecorder()
        # pre test

        testEndInS,testEndOutS,testInS,testOutS,testErrorsS,testLenS=testFun(agent,testAllEnvs,testAllNum,testAllScene,thr)
        for j in range(len(testEnvs)):
            testErrorR='\nerror: '
            testAccR='\nIPA1: '
            testEndAccR='\nIPA2: ' 
            for i in range(1,5):
                testEndAccR+=str(i)+' '+str(round(testEndInS[j][i]/(testEndInS[j][i]+testEndOutS[j][i]) if (testEndInS[j][i]+testEndOutS[j][i])>0  else 0 ,2))+' '
                testAccR+=str(i)+' '+str(round(testInS[j][i]/(testInS[j][i]+testOutS[j][i]) if (testInS[j][i]+testOutS[j][i])>0  else 0 ,2))+' '
                testErrorR+=str(i)+' '+str(round(testErrorsS[j][i]/testLenS[j][i] if testLenS[j][i]>0  else 0 ,2))+' '
            with open(preTestInfos[j],'a',encoding='UTF-8') as f:
                f.write('\npretest trainenv: '+tt+'\n+env: '+str(j+17)+'\n'+testErrorR+testAccR+testEndAccR+'\n')

        '''
        
            testAllEnvs[i].load_dict()
            testAllEnvs[i].get_file()
        '''
        for envI in range(len(envs)):
            envs[envI].load_dict()
            envs[envI].get_file()
        for K in tqdm(range(epochs)):
            for env in envs:
                env.state_mode=mode
            ll, t = 0, 0
            # env.refresh()
            rewards=[]
            #ans -> state last_goal_state last_goal goal is_end
   
            # 100
            agent.train()
            #scene
            numFlagTrain=len(multienvs)
            numFlagTest=len(testEnvs)
            trainEndInS=[[0 for i in range(5)] for i in range(numFlagTrain)]
            trainEndOutS=[[0 for i in range(5)] for i in range(numFlagTrain)]
            trainInS=[[0 for i in range(5)] for i in range(numFlagTrain)]
            trainOutS=[[0 for i in range(5)] for i in range(numFlagTrain)]
            trainErrorsS=[[0 for i in range(5)] for i in range(numFlagTrain)]
            trainLenS=[[0 for i in range(5)] for i in range(numFlagTrain)]

            testEndInS=[[0 for i in range(5)] for i in range(numFlagTest)]
            testEndOutS=[[0 for i in range(5)] for i in range(numFlagTest)]
            testInS=[[0 for i in range(5)] for i in range(numFlagTest)]
            testOutS=[[0 for i in range(5)] for i in range(numFlagTest)]
            testErrorsS=[[0 for i in range(5)] for i in range(numFlagTest)]
            testLenS=[[0 for i in range(5)] for i in range(numFlagTest)]

            for steps in range(200):
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
                        continue
                    else:
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

                        # click,eye,goal,action,mask
                        trajectorys[i].push(click,eye,ans[2],0,doneFlag*UTIL.GAMMA,lastP,lengths,person,scene)
                        indexes.append(i)
                if len(clickList)==0:
                    continue
                clickCat=torch.stack(clickList,dim=0).to(device)
                lastPCat=torch.stack(lastPList,dim=0).to(device)
                lengthStack=torch.stack(lengthsList,dim=0)
                eyeStack=torch.stack(eyeList,dim=0).to(device)
                personStack=torch.stack(personList,dim=0).to(device).unsqueeze(1)
                sceneStack=torch.stack(sceneList,dim=0).to(device).unsqueeze(1) 
                actions=agent.act(clickCat,eyeStack,lastPCat,lengthStack,personStack,sceneStack)

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

                        traInfo=trajectorys[index].getInfo()
                        # print(f'this is tra info {traInfo}')
                        trainInS[trainNum[index]-np.min(trainNum)][trainSceneNum[index]]+=traInfo[4]
                        trainOutS[trainNum[index]-np.min(trainNum)][trainSceneNum[index]]+=traInfo[5]


                        trainEndInS[trainNum[index]-np.min(trainNum)][trainSceneNum[index]]+=traInfo[6]
                        trainEndOutS[trainNum[index]-np.min(trainNum)][trainSceneNum[index]]+=int(traInfo[6])^1
                        rewards.append(traInfo[7])
                        if traInfo[0]>=3:

                            trainLenS[trainNum[index]-np.min(trainNum)][trainSceneNum[index]]+=traInfo[0]
                            trainErrorsS[trainNum[index]-np.min(trainNum)][trainSceneNum[index]]+=traInfo[11]

                        # print(f'train data {trainLenS} {trainErrorsS} {trainInS}')
                        TDZeroList=trajectorys[index].getComTraZero()
                        agentBuffer.push(TDZeroList)
                        trajectorys[index].clear()
                dbatch_size=int((1+(agentBuffer.getRatio()))*batch_size)
                if (agentBuffer.holding>=dbatch_size):
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


                    trainer.zero_grad()
                    l.backward()
                    trainer.step()

                    if  steps%2==0 and steps!=0:
                        agent.update()
                    agent.update()
            for i in range(len(testenvs)):
                testenvs[i].load_dict()
                testenvs[i].get_file()
            stopFlags=[False for i in range(len(testenvs))]
            surviveFlags=len(testAllEnvs)

    # eye click goal isEnd

            rewardsTest=[]
    #ans -> state last_goal_state last_goal goal is_end
    

            testTras=[DQNRNNTrajectory2() for i in range(surviveFlags)]

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

                            # click,eye,goal,action,mask
                            testTras[i].push(click,eye,ans[2],0,doneFlag*UTIL.GAMMA,lastP,lengths,person,scene)
                            testIndexesList.append(i)

                    if surviveFlags<=0:
                        break

                    testClickStack=torch.stack(testClickList,dim=0).to(device)
                    testLastPStack=torch.stack(testLastPList,dim=0).to(device)
                    testLengthStack=torch.stack(testLengthsList,dim=0)
                    testEyeListStack=torch.stack(testEyeList,dim=0).to(device)
                    testPersonListStack=torch.stack(testPersonList,dim=0).to(device).unsqueeze(1)
                    testSceneStack=torch.stack(testSceneList,dim=0).to(device).unsqueeze(1)       
                    testActions=agent.act(testClickStack,testEyeListStack,testLastPStack,testLengthStack,testPersonListStack,testSceneStack)
    
                    if testActions.shape==():
                        testActions=[testActions]
                    for testAction,testIndex in zip(testActions,testIndexesList):
                        testTras[testIndex].tras[-1][3]=dp(testAction)
                        if testTras[testIndex].tras[-1][4]==0:
                            testTras[testIndex].getNewTras()
                            traInfoTest=testTras[testIndex].getInfo()


                            testInS[testNum[testIndex]-np.min(testNum)][testSceneNum[testIndex]]+=traInfoTest[4]
                            testOutS[testNum[testIndex]-np.min(testNum)][testSceneNum[testIndex]]+=traInfoTest[5]
                            testEndInS[testNum[testIndex]-np.min(testNum)][testSceneNum[testIndex]]+=traInfoTest[6]
                            testEndOutS[testNum[testIndex]-np.min(testNum)][testSceneNum[testIndex]]+=int(traInfoTest[6])^1
                            rewardsTest.append(traInfoTest[7])
                            if traInfoTest[0]>=3:
                                testLenS[testNum[testIndex]-np.min(testNum)][testSceneNum[testIndex]]+=traInfoTest[0]
                                testErrorsS[testNum[testIndex]-np.min(testNum)][testSceneNum[testIndex]]+=traInfoTest[11]

                            testTras[testIndex].clear()

            t_reward_len+=1
            if t_reward_len>=20:
                t_reward_len%=20
                t_reward[t_reward_len]=np.mean(rewards)
            else:
                t_reward.append(np.mean(rewards))

            testEndInS,testEndOutS,testInS,testOutS,testErrorsS,testLenS=testFun(agent,testAllEnvs,testAllNum,testAllScene,thr)
            with open(testInfo,'a',encoding='UTF-8') as f:
                f.write('\n'+'testnum:' +str(testNum)+'epochs: '+str(K)+'\n')
            for j in range(len(testEnvs)):
                testErrorR='\nerror: '
                testAccR='\nIPA1: '
                testEndAccR='\nIPA2: ' 
                for i in range(1,5):
                    testEndAccR+=str(i)+' '+str(round(testEndInS[j][i]/(testEndInS[j][i]+testEndOutS[j][i]) if (testEndInS[j][i]+testEndOutS[j][i])>0  else 0 ,2))+' '
                    testAccR+=str(i)+' '+str(round(testInS[j][i]/(testInS[j][i]+testOutS[j][i]) if (testInS[j][i]+testOutS[j][i])>0  else 0 ,2))+' '
                    testErrorR+=str(i)+' '+str(round(testErrorsS[j][i]/testLenS[j][i] if testLenS[j][i]>0  else 0 ,2))+' '
                with open(testInfo,'a',encoding='UTF-8') as f:
                    f.write('\nenv: '+str(j+np.min(testNum))+'\n'+testErrorR+testAccR+testEndAccR+'\n')
            with open(reward_path,'a',encoding='UTF-8') as f:
                f.write('\nenv: '+str(trainNum)+'epochs: '+str(K)+'\n')
            for j in range(len(multienvs)):
                trainErrorR='\nerror: '
                trainAccR='\nIPA1: '
                trainEndAccR='\nIPA2: ' 
                for i in range(1,5):
                    trainEndAccR+=str(i)+' '+str(round(trainEndInS[j][i]/(trainEndInS[j][i]+trainEndOutS[j][i]) if (trainEndInS[j][i]+trainEndOutS[j][i])>0  else 0 ,2))+' '
                    trainAccR+=str(i)+' '+str(round(trainInS[j][i]/(trainInS[j][i]+trainOutS[j][i]) if (trainInS[j][i]+trainOutS[j][i])>0  else 0 ,2))+' '
                    trainErrorR+=str(i)+' '+str(round(trainErrorsS[j][i]/trainLenS[j][i] if trainLenS[j][i]>0  else 0 ,2))+' '
                with open(reward_path,'a',encoding='UTF-8') as f:
                    f.write('\nenv: '+str(j+np.min(trainNum))+'\n'+trainErrorR+trainAccR+trainEndAccR+'\n')


            if len(rewardsTest)>0 and np.mean(rewardsTest)>best_scores and is_training:
                torch.save(agent.target.state_dict(), m_store_path_a)
                torch.save(agent.target.state_dict(),envBestModel)

                best_scores=np.mean(rewardsTest)
                with open(testInfo,'a',encoding='UTF-8') as f:
                    f.write('\n'+'testnum:' +str(testNum)+'epochs: '+str(K)+'\n')
                with open(envBestModelRecord,'a',encoding='UTF-8') as f:
                    f.write('\n'+'testnum:' +str(testNum)+'epochs: '+str(K)+'\n')
                for j in range(len(testEnvs)):
                    testErrorR='\nerror: '
                    testAccR='\nIPA1: '
                    testEndAccR='\nIPA2: ' 
                    for i in range(1,5):
                        testEndAccR+=str(i)+' '+str(round(testEndInS[j][i]/(testEndInS[j][i]+testEndOutS[j][i]) if (testEndInS[j][i]+testEndOutS[j][i])>0  else 0 ,2))+' '
                        testAccR+=str(i)+' '+str(round(testInS[j][i]/(testInS[j][i]+testOutS[j][i]) if (testInS[j][i]+testOutS[j][i])>0  else 0 ,2))+' '
                        testErrorR+=str(i)+' '+str(round(testErrorsS[j][i]/testLenS[j][i] if testLenS[j][i]>0  else 0 ,2))+' '
                    with open(testInfo,'a',encoding='UTF-8') as f:
                        f.write('\nenv: '+str(j+np.min(testNum))+'\n'+testErrorR+testAccR+testEndAccR+'\n')
                    with open(envBestModelRecord,'a',encoding='UTF-8') as f:
                        f.write('\nenv: '+str(j+np.min(testNum))+'\n'+testErrorR+testAccR+testEndAccR+'\n')
                with open(reward_path,'a',encoding='UTF-8') as f:
                    f.write('\nenv: '+str(j+np.min(trainNum))+'\n'+'epochs: '+str(K)+'\n')
                for j in range(len(multienvs)):
                    trainErrorR='\nerror: '
                    trainAccR='\nIPA1: '
                    trainEndAccR='\nIPA2: ' 
                    for i in range(1,5):
                        trainEndAccR+=str(i)+' '+str(round(trainEndInS[j][i]/(trainEndInS[j][i]+trainEndOutS[j][i]) if (trainEndInS[j][i]+trainEndOutS[j][i])>0  else 0 ,2))+' '
                        trainAccR+=str(i)+' '+str(round(trainInS[j][i]/(trainInS[j][i]+trainOutS[j][i]) if (trainInS[j][i]+trainOutS[j][i])>0  else 0 ,2))+' '
                        trainErrorR+=str(i)+' '+str(round(trainErrorsS[j][i]/trainLenS[j][i] if trainLenS[j][i]>0  else 0 ,2))+' '
                    with open(reward_path,'a',encoding='UTF-8') as f:
                        f.write('\nenv: '+str(j+np.min(trainNum))+'\n'+trainErrorR+trainAccR+trainEndAccR+'\n')
        
        agent.load(envBestModel)
        testEndInS,testEndOutS,testInS,testOutS,testErrorsS,testLenS=testFun(agent,testAllEnvs,testAllNum,testAllScene,thr)
        for j in range(len(testEnvs)):
            testErrorR='\nerror: '
            testAccR='\nIPA1: '
            testEndAccR='\nIPA2: ' 
            for i in range(1,5):
                testEndAccR+=str(i)+' '+str(round(testEndInS[j][i]/(testEndInS[j][i]+testEndOutS[j][i]) if (testEndInS[j][i]+testEndOutS[j][i])>0  else 0 ,2))+' '
                testAccR+=str(i)+' '+str(round(testInS[j][i]/(testInS[j][i]+testOutS[j][i]) if (testInS[j][i]+testOutS[j][i])>0  else 0 ,2))+' '
                testErrorR+=str(i)+' '+str(round(testErrorsS[j][i]/testLenS[j][i] if testLenS[j][i]>0  else 0 ,2))+' '
            with open(preTestInfos[j],'a',encoding='UTF-8') as f:
                f.write('\naftertest  trainenv: '+tt+'\n+env: '+str(j+17)+'\n'+testErrorR+testAccR+testEndAccR+'\n')



if __name__=='__main__':

    def str2bool(v):
        if isinstance(v,bool):
            return v
        if v.lower() in ('true','True','yes'):
            return True
        elif v.lower() in ('no','false','False'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser=argparse.ArgumentParser()
    parser.add_argument('-modelPath',type=str,default='20231022203322')
    parser.add_argument('-cuda',type=str,default='cuda:1')
    parser.add_argument('-mode',type=int,default=1)
    parser.add_argument('-net',type=int,default=1)
    parser.add_argument('-sup',type=str,default='50')
    parser.add_argument('-preload',type=str2bool,default=True)
    parser.add_argument('-lr',type=float,default=0.0005)
    parser.add_argument('-layers',type=int,default=5)
    parser.add_argument('-embed',type=int,default=128)
    parser.add_argument('-rems',type=int,default=5)
    parser.add_argument('-topN',type=int,default=5)
    parser.add_argument('-skip',type=str2bool,default=False)
    parser.add_argument('-low',type=int,default=2)
    parser.add_argument('-high',type=int,default=15)
    parser.add_argument('-skipN',type=int,default=10)
    parser.add_argument('-epochs',type=int,default=500)
    parser.add_argument('-batchsize',type=int,default=256)
    parser.add_argument('-addPre',type=str2bool,default=False)
    # iteration flag true->iteration donot use before False->iteration use before
    parser.add_argument('-itFlag',type=str2bool,default=False)
    parser.add_argument('-load',type=str2bool,default=False)
    parser.add_argument('-idFlag',type=str2bool,default=False)
    parser.add_argument('-restrict',type=str2bool,default=False)
    # TRUE -> skip the scene three
    parser.add_argument('-thr',type=str2bool,default=False)
    parser.add_argument('-path',type=str,default='restrict')
    args=parser.parse_args()
    print(args.idFlag)
    print(args.load)
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    agent=REMAgent2(device=device,rnn_layer=args.layers,embed_n=args.embed,idFlag=args.idFlag)
    store=UTIL.getTimeStamp()
    actor_load=''
    if args.preload:
        # no id
        if args.idFlag==False:
            actor_load='/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20231214/trainallscene/181521/dqnnetoffline.pt'
        # id
        else:
            # actor_load='/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20231221/trainallscene/restrict/dqnnetoffline.pt'
            actor_load=os.path.join('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20231221/trainallscene/',args.path,'dqnnetoffline.pt')
        agent.load(actor_load)
    # else:
    #     actor_load='/home/wu_tian_ci/drl_project/mymodel/ddpg/pretrain_data/ddpg_train/1last_move5/ActorNet.pt'
    mainPath=os.path.join('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/',store[0:-6],'offline')
    if not os.path.exists(mainPath):
        os.makedirs(mainPath)
    fileNum=len(os.listdir(mainPath))
    store_path=os.path.join(mainPath,str(fileNum))
    # json_path=os.path.join('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/json_path/',store[0:-6],'offline',store[-6:])
    # value_path=os.path.join('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/value_path/',store[0:-6],'offline',store[-6:])
    # critic_loss_path=os.path.join('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/critic_loss/',store[0:-6],'offline',store[-6:])
    # if not os.path.exists(json_path):
    #     os.makedirs(json_path)
    # if not os.path.exists(value_path):
    #     os.makedirs(value_path)
    # if not os.path.exists(critic_loss_path):
    #     os.makedirs(critic_loss_path)
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
        f.write('time:'+store+'\n'+'envs'+str(envs)+'\n trainEnvs:'+str(trainEnvs)+'\n testEnvs:'+str(testEnvs)+'\n'+' lr:'+str(args.lr)+' preload: '+str(args.preload)\
                +' topN:'+str(args.topN)+' skip:'+str(args.skip)+' skip_info:' +str(args.low)+' '+str(args.high)+' '+str(args.skipN)+'\n'\
                +'layers: '+str(args.layers)+'\nactorload: '+actor_load+'\n idFlag '+str(args.idFlag)+' itFlag: '+str(args.itFlag)+' thr: '+str(args.thr)+'\n'\
                + 'fileNum: '+str(fileNum)+' restrict: '+str(args.restrict))
        if args.sup!='50':
            f.write('\n'+args.sup)
    train_epoch(agent, args.lr, args.epochs, args.batchsize,device,args.mode,store_path=store_path,multienvs=trainEnvs,testEnvs=testEnvs,\
                remBlocskNum=args.rems,topN=args.topN,\
                randLow=args.low,randHigh=args.high,randSize=args.skipN,skipFlag=args.skip,addPre=args.addPre,iterationFlag=args.itFlag,\
                load=args.load,loadPath=actor_load,restrict=args.restrict,thr=args.thr)