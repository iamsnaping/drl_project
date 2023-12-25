# 导入类库
from flask import Flask
from flask import make_response
from flask import request
import json
import requests
from json import JSONEncoder
import jsonpickle
import threading
import torch
import os
import sys
import tqdm
import numpy as np
import argparse
import torch
from distutils.util import strtobool
sys.path.append('/home/wu_tian_ci/drl_project/mymodel/dqn')
from dqnenv import *
from dqnutils import *
import dqnutils as UTIL
import dqnbasenet
import dqnagent
import json
from dqnagent import *
import csv
import pandas as pd
from onlineconfig import *
import warnings
warnings.filterwarnings("error") 
#  flask 传送原始数据给 trainModel-> click_x,click_y,eye_x,eye_y,state,再由flask端进行区域划分
#  也由flask端进行数据存储


# to stop training
modelConditioner=threading.Condition(threading.Lock())
# for train
bufferConditioner=threading.Condition(threading.Lock())
STOP_TRAIN_FLAG=False

# response sample 
# class Responser(object):
#     def __init__(self):
#         # type 0-> wanted prediect 1-> train data
#         self.type=0
#         self.data={}


REWARDS=20

# create data

class DataCreater(object):
    def __init__(self) -> None:
        self.eye=[]
        self.click=[]
        self.pList=[]

    def refreshTra(self):
        self.eye=[]
        # self.click=[]
        # self.pList=[]
    
    def refresh(self):
        self.eye=[]
        self.click=[]
        self.pList=[]

    # region_level data
    # false -> not predict

    def getEye(self):
        eyeList=[]
        if len(self.eye)<10:
            eyeList=dp(self.eye)
        else:
            step=(len(self.eye)//10)
            for i in range(9):
                eyeList.append(self.eye[i*step])
            eyeList.append(self.eye[-1])
        length=len(eyeList)
        if len(eyeList)<10:
            eyeList.extend([0 for i in range(10-length)])
        return eyeList,length

    def addEye(self,eye):
        self.eye.append(eye)
        return self.getEye()
    
    def addP(self,p):
        self.pList.append(p)
        return dp(self.pList[-3:])

    def addClick(self,click,mouseS):
        clickFlag= True if len(self.click)>=3 else False
        if clickFlag:
            clickAns=dp(self.click[-3:])
        else:
            clickAns=len(self.click)
        if mouseS==0:
            if len(self.click)==0 or self.click[-1]!=click:
                self.click.append(click)
        
        if len(self.click)>=3:
            return 
        return False,len(self.click)
    
    def initP(self):
        self.pList=dp(self.click)

        

class PresentsRecorder(object):

    def __init__(self,num) -> None:
        self.num=num
        self.recorder=[[] for i in range(num)]
        self.flag=[False for i in range(num)]
    
    # init with last3goal
    def init(self,index,last3goal):
        self.recorder[index]=dp(last3goal)

    def addRecorder(self):
        self.recorder.append([])
        self.flag.append(False)
        self.num+=1

    def add(self,index,num):
        for i in range(2):
            self.recorder[index][i]=self.recorder[index][i+1]
        if num!=12:
            self.recorder[index][2]=num


    def getLastPresents(self,index):
        return dp(self.recorder[index])


# class reocord the data UNDO
# class NumRecorder(object):

#     def __init__(self,nums) -> None:
#         self.nums=nums
#         self.recoder=[0 for i in range(nums)]
    
#     def add(self,index):
#         self.recoder[index]+=1

#     def clearOne(self,index):
#         self.recoder[index]=0

#     def ind(self,index):
#         return self.recoder[index]
#     def addRecorder(self):
#         self.nums+=1
#         self.recoder.append(0)

#     def clear(self):
#         self.recoder=[0 for i in range(self.nums)]




class TrainModel(object):
    def __init__(self) -> None:
        

        # 1 iteration mode 2->no iteration mode 0->offline
        self.mode=0
        self.switchModeFlag=False

        self.device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        # train 
        self.agent=REMAgent2(device=self.device,rnn_layer=5,embed_n=128)

        #predict
        self.predictModel=REMAgent2(device=self.device,rnn_layer=5,embed_n=128)

        # 
        self.isTraining=False
        self.checkPointNum=0
        self.predictCnts=1
        self.onlineAcc=[]
        self.onlinePRR=[]
        self.onlineAveAcc=[]
        # predict right rate PRR
        self.onlineLen=[]
        self.traAcc=[]
        self.traLen=[]
        self.traEndAcc=[]
        self.traPRR=[]
        # load path
        # store path
        # recorde the index of the sequence
        self.numberRecorder=0.
        self.timeStamp=getTimeStamp()
        self.storePath=os.path.join(OnlineConfig.ONLIEN_DATA_PATH.value,self.timeStamp[0:-6],'model',self.timeStamp[-6:],str(self.mode))
        if not os.path.exists(self.storePath):
            os.makedirs(self.storePath)
        self.modelSavePath=os.path.join(self.storePath,'onlineModel.pt')
        self.rewardPath=os.path.join(self.storePath,'reward.txt')   
        self.onlineReward=os.path.join(self.storePath,'onlineReward.txt')    
        self.updateModel=os.path.join(self.storePath,'updateModel.txt')    
        self.update=False

        # self.dataCreater=DataCreater()
        self.stop=False
        self.tras=DQNRNNTrajectory2()

        # for dataset
        # self.dataSetBuffer=ReplayBufferRNN2(2**19,self.device)
        # for test people data
        self.trainBuffer=ReplayBufferRNN2(2**17,self.device)
        self.iterationBuffer=ReplayBufferRNN2(2**17,self.device)
        self.batchSize=8
        self.lr=OnlineConfig.LR.value
        self.agent.load(OnlineConfig.LOAD_PATH.value)
        self.predictModel.load(OnlineConfig.LOAD_PATH.value)
        self.baseStorePath=OnlineConfig.EYE_DATA_PATH.value
        self.dc=DataCreater()
        self.newFileNums=0
        self.remBlocskNum=5
        self.best_scores=-np.inf
    
    def loadModel(self):
        if self.mode==0:
            self.agent.load(OnlineConfig.LOAD_PATH.value)
            self.predictModel.load(OnlineConfig.LOAD_PATH.value)
        elif self.mode==1:
            self.agent.load(OnlineConfig.ITERATION_PATH.value)
            self.predictModel.load(OnlineConfig.ITERATION_PATH.value)
        elif self.mode==2:
            self.agent.load(OnlineConfig.NO_ITERATION_PATH.value)
            self.predictModel.load(OnlineConfig.NO_ITERATION_PATH.value)


    def modeSwitch(self):
        self.loadModel()
        self.isTraining=False

        self.predictCnts=1
        self.onlineAcc=[]
        self.onlinePRR=[]
        self.onlineAveAcc=[]
        # predict right rate PRR
        self.onlineLen=[]
        self.traAcc=[]
        self.traLen=[]
        self.traEndAcc=[]
        self.traPRR=[]
        # load path
        # store path
        # recorde the index of the sequence
        self.numberRecorder=0.
        self.checkPointNum=0
        self.storePath=os.path.join(OnlineConfig.ONLIEN_DATA_PATH.value,self.timeStamp[0:-6],'model',self.timeStamp[-6:],str(self.mode))
        if not os.path.exists(self.storePath):
            os.makedirs(self.storePath)
        self.modelSavePath=os.path.join(self.storePath,'onlineModel.pt')
        self.rewardPath=os.path.join(self.storePath,'reward.txt')   
        self.onlineReward=os.path.join(self.storePath,'onlineReward.txt')    
        self.updateModel=os.path.join(self.storePath,'updateModel.txt')    
        self.update=False

        self.stop=False
        self.tras=DQNRNNTrajectory2()

        # for dataset
        # self.dataSetBuffer=ReplayBufferRNN2(2**19,self.device)
        # for test people data
        self.batchSize=8
        self.dc=DataCreater()
        self.newFileNums=0
        self.remBlocskNum=5
        self.switchModeFlag=False     

    # multi threading
    def train(self):
        self.modeSwitch()
        global dataRecorder
        trainer = torch.optim.Adam(lr=self.lr, params=self.agent.online.parameters())
        EPOSILON=0.02
        loss=nn.HuberLoss()
        envs=[]
        t_reward=[]
        trajectorys=[]
        pr=PresentsRecorder(20)
        t_reward_len=-1
        trainNums=[]
        trainScene=[]
        skipSize=5

        if os.path.exists(OnlineConfig.INDIVIDUAL.value):
            fileNum=len(os.listdir(OnlineConfig.INDIVIDUAL.value))
        else:
            fileNum=0
        onlineNum=int(np.ceil(fileNum/5))
        for scene in range(1,5):
            for i in range(onlineNum):
                envPath=os.path.join(OnlineConfig.INDIVIDUAL_NO_SCENE.value,str(scene))
                if not os.path.exists(envPath):
                    continue
                if len(os.listdir(envPath))==0:
                    continue
                envs.append(DQNRNNEnv(envPath))
                trajectorys.append(DQNRNNTrajectory2())
                pr.addRecorder()
                trainNums.append(OnlineConfig.PERSON.value)
                trainScene.append(scene)
        for i in range(len(envs)):
            envs[i].setMODE(self.mode)
            envs[i].load_dict()
            envs[i].get_file()
        for env in envs:
            env.state_mode=1
        K=0
        device=self.device
        while True:
            if self.stop:
                with modelConditioner:
                    print('stop to train1')
                    modelConditioner.wait()
                print('awake to train1')

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

            if self.switchModeFlag:
                self.modeSwitch()
                self.isTraining=False
                print('mode stop')
                return
                trainer = torch.optim.Adam(lr=self.lr, params=self.agent.online.parameters())

            for steps in range(200):
                if self.stop:
                    with modelConditioner:
                        print('stop train2')
                        modelConditioner.wait()
                    print('awake2')
                if self.switchModeFlag:
                    break
                if self.newFileNums%5==0 and self.newFileNums!=0:
                    updateTimes=int(np.ceil(self.newFileNums/5))
                    skipSize+=1
                    for sceneNum in range(1,5):
                        for i in range(updateTimes):
                            envPath=os.path.join(OnlineConfig.INDIVIDUAL_NO_SCENE.value,str(scene))
                            if not os.path.exists(envPath):
                                continue
                            if len(os.listdir(envPath))==0:
                                continue
                            envs.append(DQNRNNEnv(envPath))
                            envs[i].setMODE(self.mode)
                            envs[-1].load_dict()
                            envs[-1].get_file()
                            pr.addRecorder()
                            trajectorys.append(DQNRNNTrajectory2())
                            trainNums.append(OnlineConfig.PERSON.value)
                            trainScene.append(sceneNum)
                    
                    self.newFileNums=0
                # print('train2')
                eyeList=[]
                clickList=[]
                indexes=[]
                lastPList=[]
                lengthsList=[]
                personList=[]
                sceneList=[]
                # ans -> eye click goal isEnd
                skipSize=np.minimum(15,skipSize)
                # skipNum=np.random.randint(low=0,high=20,size=skipSize)
                for i in range(len(envs)):
                    # if OnlineConfig.SKIP.value and i in skipNum:
                    #     continue
                    ans=envs[i].act()
                    if isinstance(ans,bool):
                        envs[i].setMODE(self.mode)
                        envs[i].load_dict()
                        envs[i].get_file()
                        continue
                    else:
                        eye=torch.tensor(ans[0],dtype=torch.long)
                        click=torch.tensor(ans[1],dtype=torch.long).to(device)
                        lengths=torch.tensor(ans[4],dtype=torch.long)
                        if pr.flag[i]==False:
                            pr.init(i,ans[1])
                        lastP=torch.tensor(pr.getLastPresents(i),dtype=torch.long).to(device)
                        person=torch.tensor([trainNums[i]],dtype=torch.long).to(device)
                        scene=torch.tensor([trainScene[i]],dtype=torch.long).to(device)
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
                        trajectorys[i].push(click,eye,ans[2],0,doneFlag*UTIL.GAMMA,lastP,lengths,person,scene)
                        indexes.append(i)
                clickCat=torch.stack(clickList,dim=0).to(device)
                lastPCat=torch.stack(lastPList,dim=0).to(device)
                lengthStack=torch.stack(lengthsList,dim=0)
                personStack=torch.stack(personList,dim=0).to(device).unsqueeze(1)   
                sceneStack=torch.stack(sceneList,dim=0).to(device).unsqueeze(1)   

                eyeList=torch.stack(eyeList,dim=0).to(device)
                actions=self.agent.act(clickCat,eyeList,lastPCat,lengthStack,personStack,sceneStack)
                if actions.shape==():
                    actions=[actions]
                
                for actS,index in zip(actions,indexes):
                    pr.add(index,actS)
                    prob=random.random()
                    if prob>EPOSILON:
                        trajectorys[index].tras[-1][3]=dp(actS)
                    else:
                        trajectorys[index].tras[-1][3]=random.randint(0,12)
                    if trajectorys[index].tras[-1][4]==0:
                        trajectorys[index].getNewTras()
                        traInfo=trajectorys[index].getInfo()
                        if traInfo[0]>1 and trainNums[index]>=OnlineConfig.PERSON_ENVS.value:
                            traLenOverOne+=traInfo[0]
                            noActionNum+=traInfo[1]
                            noActionNumWithThreshold+=traInfo[2]
                        if trainNums[index]>=OnlineConfig.PERSON_ENVS.value:
                            lastWithNoAction+=traInfo[3]
                            totalIn+=traInfo[4]
                            totalOut+=traInfo[5]
                            endIn+=traInfo[6]
                            endOut+=int(traInfo[6])^1
                        if trainNums[index]>=OnlineConfig.PERSON_ENVS.value:
                            rewards.append(traInfo[7])
                            endAveReward.append(traInfo[8])
                            inNoAct+=traInfo[9]
                            outNoAct+=traInfo[10]
                        if traInfo[0]>=3 and trainNums[index]>=OnlineConfig.PERSON_ENVS.value:
                            traLenOverThree+=traInfo[0]
                            totalErrors+=traInfo[11]
                        TDZeroList=trajectorys[index].getComTraZero()
                        if traInfo[12]!=-1 and trainNums[index]>=OnlineConfig.PERSON_ENVS.value:
                            endInNoAct+=traInfo[12]
                            endOutNoAct+=int(traInfo[12])^1
                        if self.mode==1:
                            self.iterationBuffer.push(TDZeroList)
                        elif self.mode==0:
                            self.iterationBuffer.push(TDZeroList)
                        else:
                            self.trainBuffer.push(TDZeroList)
        
                        trajectorys[index].clear()

                holding=0
                capacity=0
                with bufferConditioner:
                    if self.mode ==1:
                        datasetBatchSize=int((1+(self.iterationBuffer.getRatio()))*self.batchSize)
                        holding=self.iterationBuffer.holding
                        capacity=self.iterationBuffer.capacity
                    elif self.mode==0:
                        datasetBatchSize=int((1+(self.iterationBuffer.getRatio()))*self.batchSize)
                        holding=self.iterationBuffer.holding
                        capacity=self.iterationBuffer.capacity
                    else:
                        datasetBatchSize=int((1+(self.trainBuffer.getRatio()))*self.batchSize)
                        holding=self.trainBuffer.holding
                        capacity=self.trainBuffer.capacity
                
                    if holding<datasetBatchSize:
                        continue

                if datasetBatchSize*2 > capacity and self.batchSize<256:
                    self.batchSize*=2
                if self.mode==2:
                    clickList,eyeList,lastPList,lengths,person,scene,actionList,rewardList,maskList,nclickList,neyeList,nlastPList,nlengths,nperson,nscene = self.trainBuffer.sample(datasetBatchSize)
                elif self.mode==1 :
                    clickList,eyeList,lastPList,lengths,person,scene,actionList,rewardList,maskList,nclickList,neyeList,nlastPList,nlengths,nperson,nscene = self.iterationBuffer.sample(datasetBatchSize)
                elif self.mode==0:
                    clickList,eyeList,lastPList,lengths,person,scene,actionList,rewardList,maskList,nclickList,neyeList,nlastPList,nlengths,nperson,nscene = self.iterationBuffer.sample(datasetBatchSize)
        
                deltaP = torch.rand(self.remBlocskNum).to(device)
                deltaP =deltaP/ deltaP.sum()
                with torch.no_grad():
                    onlineValues=self.agent.online(clickList,eyeList,lastPList,lengths,person,scene)
                    onlineValues=sum(onlineValues)/len(onlineValues)
                    yAction=torch.argmax(onlineValues,dim=-1,keepdim=True)
                    targetValues=self.agent.target(nclickList,neyeList,nlastPList,nlengths,nperson,nscene)
                    for i in range(self.remBlocskNum):
                        targetValues[i]=targetValues[i]*deltaP[i]
                    targetValues=sum(targetValues)
                    y=targetValues.gather(dim=-1,index=yAction)*maskList+rewardList
                values=self.agent.online(clickList,eyeList,lastPList,lengths,person,scene)
                for i in range(self.remBlocskNum):
                    values[i]=values[i]*deltaP[i]
                values=sum(values)
                values=values.gather(dim=-1,index=actionList)
                l = loss(values, y)
                try:
                    trainer.zero_grad()
                    l.backward()
                    trainer.step()
                except:
                    self.modeSwitch()
                    self.isTraining=False
                    print('mode stop')
                    return
                if steps%10==0 and steps!=0:
                    self.agent.update()

            t_reward_len+=1
            if t_reward_len>=10:
                t_reward_len%=10
                t_reward[t_reward_len]=np.mean(rewards)
            else:
                t_reward.append(np.mean(rewards))

            if len(t_reward)>5 and np.mean(t_reward)>self.best_scores:
                print('save')
                with modelConditioner:
                    torch.save(self.agent.target.state_dict(), self.modelSavePath)
                    checkPointPath=os.path.join(self.storePath,str(self.checkPointNum))
                    if not os.path.exists(checkPointPath):
                        os.makedirs(checkPointPath)
                        # print(f'create path {checkPointPath}')
                    self.checkPointNum+=1
                    checkPointPath=os.path.join(checkPointPath,'checkpointModel.pt')
                    torch.save(self.agent.target.state_dict(), checkPointPath)
                    self.update=True
                self.best_scores=np.mean(t_reward)
                with open(self.updateModel,'a',encoding='UTF-8') as updateModel:
                    updateModel.write('eposides:'+str(K+1)+' ave_eposides_rewards:'+str(round(np.mean(t_reward),2))+\
                    ' ave reward '+str(round(t_reward[-1],2)) +'\n'+\
                    ' end_in: '+str(endIn)+' end_out: '+str(endOut)+' IPA2: '+str(round(endIn/(endOut+endIn) if (endOut+endIn) >0 else 0,2))+\
                    ' end_in_no_action: '+str(endInNoAct)+' end_out_no_act: '+str(endOutNoAct)+' IPA2_NOACTION:'+str(round(endInNoAct/(endInNoAct+endOutNoAct) if (endInNoAct+endOutNoAct) >0 else 0,2))+'\n'+\
                    ' in: '+str(totalIn)+' _out:'+str(totalOut)+' IPA1:'+str(round(totalIn/(totalIn+totalOut) if (totalIn+totalOut) >0 else 0,2))+'\n')

            K+=1
            with open(self.rewardPath,'a',encoding='UTF-8') as rewardFile:
                rewardFile.write('eposides:'+str(K+1)+' ave_eposides_rewards:'+str(round(np.mean(t_reward),2))+\
                ' ave reward '+str(round(t_reward[-1],2)) +'\n'+\
                ' end_in: '+str(endIn)+' end_out: '+str(endOut)+' IPA2: '+str(round(endIn/(endOut+endIn) if  (endOut+endIn)>0 else 0,2))+\
                ' end_in_no_action: '+str(endInNoAct)+' end_out_no_act: '+str(endOutNoAct)+' IPA2_NOACTION:'+str(round(endInNoAct/(endInNoAct+endOutNoAct) if (endInNoAct+endOutNoAct) >0 else 0,2))+'\n'+\
                ' in: '+str(totalIn)+' _out:'+str(totalOut)+' IPA1:'+str(round(totalIn/(totalIn+totalOut) if (totalIn+totalOut) >0 else 0,2))+'\n')
    

    def getAveTraAcc(self):
        try:
            aveAcc='file_IPA1: '+str(round(np.sum(self.traAcc)/np.sum(self.traLen),2))+' file_IPA2: '+str(round(np.mean(self.traEndAcc),2))+' file_PRR: '+str(round(np.mean(self.traPRR),2))
        except:
            aveAcc='error'
        return aveAcc

    def traAccClear(self):
        self.traAcc=[]
        self.traLen=[]
        self.traEndAcc=[]
        self.traPRR=[]

    def writeReward(self):
        info=self.tras.getInfo2()
        # print(info)
        # len(self.tras),totalCorrect,endCorrect,totalReward,endReward
        if len(self.onlineAveAcc)<1000:
            self.onlineAcc.append(info[2])
            self.onlineAveAcc.append(info[1])
            self.onlineLen.append(info[0])
            self.onlinePRR.append(info[5])
        else:
            self.onlineAcc[self.predictCnts%1000]=info[2]
            self.onlineAveAcc[self.predictCnts%1000]=info[1]
            self.onlineLen[self.predictCnts%1000]=info[0]
            self.onlinePRR[self.predictCnts%1000]=info[5]
        self.traAcc.append(info[1])
        self.traEndAcc.append(info[2])
        self.traLen.append(info[0])
        self.traPRR.append(info[5])
        r=info[4]/info[0] if info[0] >0 else 0
        r/=100
        if r<0:
            self.best_scores+=r
        with open(self.onlineReward,'a') as f:
            f.write('mode: '+str(self.mode)+' '+'tras: '+str(self.predictCnts)+' tra_len: '+str(info[0])+' '+\
                    'total_correct: '+str(info[1]) +' end_correct: '+str(info[2])+'\n'+\
                    'total_reward: '+str(info[3])+' end_reward: '+str(info[4])+' ave_reward: '+str(round(info[4]/info[0] if info[0] >0 else 0,2))+'\n'+\
                    'IPA2: '+str(round(np.mean(self.onlineAcc),2))+' TOTALIPA1: '+str(round(float(np.sum(self.onlineAveAcc)/np.sum(self.onlineLen) if np.sum(self.onlineLen) >0 else 0),2))+\
                    'onlinePRR: '+str(round(np.mean(self.traPRR),2))+'\n')
        self.predictCnts+=1
    
    '''
    name:snapping
    TODO:write txt into file->self.onlineReward
    time:2023.11.23
    params:writeTxt write content
    return: None
    '''
    def writeTxt(self,writeTxt='\n'):
        with open(self.onlineReward,'a') as f:
            f.write(writeTxt)

    def predict(self,*data):
        # eyeList,clickList,pList,mouseS,person,sceneId,length
        eyeList,clickList,lastP,state,person,scene,length,goal=data
        mask=UTIL.GAMMA
        if self.update:
            with modelConditioner:
                print('used the update')
                self.predictModel.load(self.modelSavePath)
                self.update=False

        clickTensor=torch.tensor([clickList],dtype=torch.long).unsqueeze(0).to(self.device)

        eyeTensor=torch.tensor([eyeList],dtype=torch.long).unsqueeze(0).to(self.device)
        lenTensor=torch.tensor([length],dtype=torch.long)
        pTensor=torch.tensor([lastP],dtype=torch.long).unsqueeze(0).to(self.device)
        personTensor=torch.tensor([[[person]]],dtype=torch.long).to(self.device)
        sceneTensor=torch.tensor([[[scene]]],dtype=torch.long).to(self.device)
        ans=int(self.predictModel.act(clickTensor,eyeTensor,pTensor,lenTensor,personTensor,sceneTensor))
        # click,eye,goal,action,mask
        mask=UTIL.GAMMA
        if state==0:
            mask=0
        self.tras.push(clickTensor.squeeze(),eyeTensor.squeeze(),0,ans,mask,pTensor.squeeze(),\
                       lenTensor.squeeze(),personTensor.squeeze(0).squeeze(0),sceneTensor.squeeze(0).squeeze(0))
        if state==0 and len(self.tras.tras)>0:
            print(f'add')
            self.tras.setGoal(goal)
            self.numberRecorder=0
            self.tras.getNewTras()
            TDZeroList=self.tras.getComTraZero()
            # print(TDZeroList)]
            try:
                with bufferConditioner:
                    if self.mode==2:
                        self.trainBuffer.push(TDZeroList)
                    elif self.mode==1:
                        self.iterationBuffer.push(TDZeroList)
                    elif self.mode==0:
                        self.iterationBuffer.push(TDZeroList)
            except:
                print(TDZeroList)
            self.writeReward()
            self.tras.clear()

        return ans,'success'

model=TrainModel()
# dataRecorder=OnlineDataRecorder(storePath='/home/wu_tian_ci/eyedatanew')

app = Flask(__name__)
# 0：指令 1：数据
# 添加视图


def getResponse(flag=True,message='success',data=True):
    responseDict={}
    responseDict['data']=data
    responseDict['flag']=flag
    responseDict['message']=message
    return responseDict


@app.route('/sendData',methods=['post','get'])
def sendData():
    global model
    print('send')
    model.dc.refresh()
    writeTxt=model.getAveTraAcc()
    model.writeTxt(writeTxt)
    print(f'this is txt {writeTxt}')
    model.traAccClear()
    model.writeTxt('\nend\n\n')
    resp=request.get_json()
    fileName=resp.get('fileName')
    dataList=resp.get('dataList')
    peopleNum=resp.get('peopleNum')
    sceneNum=resp.get('sceneNum')
    savePath=os.path.join(str(OnlineConfig.EYE_DATA_PATH.value),str(peopleNum),str(sceneNum))
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    savePath=os.path.join(savePath,fileName)
    df=pd.DataFrame(dataList)
    df.to_csv(savePath,header=None,index=False)
    responseDict=getResponse()
    responseJson=json.dumps(responseDict)
    response=make_response(responseJson)    
    model.newFileNums+=1
    return response
    



@app.route('/excuteOrder',methods=['post','get'])
def excuteOrder():
    # order -> 0-> stop train 1->begin to train 2->start to record 3->end record
    resp=request.get_json()
    orderType=int(resp.get('order'))
    responseDict=getResponse()
    if orderType==0:
        with modelConditioner:
            model.stop=True
        responseDict['data']=True
        responseDict['flag']=0
        responseDict['message']='success'
    elif orderType==1:
        # if model.trainBuffer.holding<model.batchSize:
        #     pass
        # else:
        if model.isTraining==False:
            print('start to train')
            model.isTraining=True
            t=threading.Thread(target=model.train)
            t.start()
            model.stop=False
        else:
            model.stop=False
            with modelConditioner:
                print('awake')
                modelConditioner.notify_all()
            responseDict['data']=True
            responseDict['flag']=0
            responseDict['message']='success'
    elif orderType==2:
        responseDict['data']=True
        responseDict['flag']=0
        responseDict['message']='success'
    elif orderType==3:
        responseDict['data']=True
        responseDict['flag']=0
        responseDict['message']='success'
    # switch mode 
    elif orderType==5:
        mode=int(resp.get('mode'))
        responseDict['message']='success'
        with modelConditioner:
            if model.mode!=mode:
                if model.mode!=0:
                    model.switchModeFlag=True
                model.mode=mode
            else:
                responseDict['message']='mode not change'
            
        responseDict['data']=True
        responseDict['flag']=0


    responseJson=json.dumps(responseDict)
    response=make_response(responseJson)
    return response




@app.route('/getPredict',methods=['post','get'])
def getPredict():
    global model,STOP_TRAIN_FLAG
    resp=request.get_json()
    responseDict=getResponse()
    eyeList,clickList,pList,mouseS,person,sceneId,length,goal=resp.get('eyeList'),resp.get('clickList'),resp.get('pList'),\
    resp.get('mouseS'),resp.get('person'),resp.get('scene'),resp.get('length'),resp.get('goal')
    
    print(f'scene {sceneId}')
    try:
        ans=model.predict(eyeList,clickList,pList,mouseS,person,sceneId,length,goal)
    except:
        ans=12
    print(ans)
    responseDict['data']=ans[0]
    responseDict['flag']=True
    responseDict['message']=ans[1]
    responseJson=json.dumps(responseDict)
    response=make_response(responseJson)
    return response


# 启动程序
if __name__ == '__main__':
    app.run( host='0.0.0.0', port=7788)
    

