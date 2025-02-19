from typing import Any
import torch
import numpy as np
import random
import PIL
# import pynput
from PIL import ImageGrab
import numpy as np
# import pynput.keyboard
import os
import collections
import pandas as pd

from copy import deepcopy as dp
from dqnenv import *
from dqnagent import *

import enum


class AgentFlag(enum.Enum):
    REMNet2=1
    REMNet2_NO_PID=2
    REMNet2_NO_SID=3
    REMNet2_NO_SPID=4
    REMNet2_NO_LP=5



GAMMA=0.96

class DQNReward(object):

    def __init__(self) -> None:
        self.prediction=[]
        self.errors=0.
        self.lastErrors=0.
    def initPrediction(self,initList):
        self.prediction=dp(initList)

    def addPrediction(self,predict):
        if len(self.prediction)<3:
            self.prediction.append(predict)
        else:
            self.prediction[0]=self.prediction[1]
            self.prediction[1]=self.prediction[2]
            self.prediction[2]=predict

    # right -> normal 3 -> error -3  end: right->15 error->-15
    # noaction ->right -> normal ->3 end ->15 error-> normal-> -1 end-> -15
    # 20 pace 
    # no ratio
    def getReward(self,predict,goal,mask,present,judgeFlag):
        if predict!=12:
            present=predict
        if present==goal:
            if int(mask+0.5)==0:
                return 20
            return 3
        else:
            if int(mask+0.5)==0:
                return -20
            # if not judgeFlag:
            #     return -5
            if predict==12:
                return -1
            return -3

# true -> False
    def judgeError(self):
        flag=True
        if self.prediction[-1] != self.prediction[-2] and self.prediction[-2] !=self.prediction[-3]:
            flag=False
        if flag==False:
            self.errors+=1
        return flag
    
class ReplayBufferRNN(object):
    #click->3 eye->3 aciton->1 nclick->3 eye->3 mask->1 reward->1 seq->3 nseq->3
    #click->3 eye->3 goal->1 action->1 seq->3 mask->1 reward->1 nclick->3 neye->3 nseq->3
    def __init__(self,maxLen,device) -> None:
        self.capacity=maxLen
        self.lengths=torch.empty((maxLen),dtype=torch.long)
        self.nlengths=torch.empty((maxLen),dtype=torch.long)

        self.person=torch.empty((maxLen,1),dtype=torch.float32).to(device)
        self.nperson=torch.empty((maxLen,1),dtype=torch.float32).to(device)


        self.clickList=torch.empty((maxLen,3),dtype=torch.long).to(device)
        self.eyeList=torch.empty((maxLen,10),dtype=torch.long).to(device)
        self.lastPList=torch.empty((maxLen,3),dtype=torch.long).to(device)
        self.actionList=np.empty((maxLen),dtype=np.int64)
        self.maskList=np.empty((maxLen),dtype=np.int64)
        self.rewardList=np.empty((maxLen),dtype=np.int64)
        self.nclickList=torch.empty((maxLen,3),dtype=torch.long).to(device)
        self.neyeList=torch.empty((maxLen,10),dtype=torch.long).to(device)
        self.nlastPList=torch.empty((maxLen,3),dtype=torch.long).to(device)

        self.cursor=0
        self.holding=0
        self.isFull=False
        self.device=device
        self.ratio=0.


    def getRatio(self):
        self.ratio=self.holding/self.capacity
        return self.ratio
    
    # memoList -> [[click],[eye],[length(eye)]] ->tensor
    # clickList,eyeList,actionList,maskList,rewardList,nclickList,neyeList
    def push(self,memoTuple:tuple):
        clickList,eyeList,lastPList,lengths,person,goalList,actionList,rewardList,maskList,nclickList,neyeList,nlastPList,nlenghts,nperson=memoTuple
        cursor=len(lengths)+self.cursor
        if cursor>self.capacity:
            c1=cursor-self.capacity
            c2=len(lengths)-c1
            self.lengths[self.cursor:]=lengths[0:c2]
            self.nlengths[self.cursor:]=nlenghts[0:c2]
            self.clickList[self.cursor:]=clickList[0:c2]
            self.eyeList[self.cursor:]=eyeList[0:c2]
            self.actionList[self.cursor:]=actionList[0:c2]
            self.maskList[self.cursor:]=maskList[0:c2]
            self.nclickList[self.cursor:]=nclickList[0:c2]
            self.neyeList[self.cursor:]=neyeList[0:c2]
            self.rewardList[self.cursor:]=rewardList[0:c2]
            self.lastPList[self.cursor:]=lastPList[0:c2]
            self.nlastPList[self.cursor:]=nlastPList[0:c2]
            self.person[self.cursor:]=person[0:c2]
            self.nperson[self.cursor:]=nperson[0:c2]

            self.lengths[0:c1]=lengths[c2:]
            self.nlengths[0:c1]=nlenghts[c2:]
            self.clickList[0:c1]=clickList[c2:]
            self.eyeList[0:c1]=eyeList[c2:]
            self.actionList[0:c1]=actionList[c2:]
            self.maskList[0:c1]=maskList[c2:]
            self.rewardList[0:c1]=rewardList[c2:]
            self.nclickList[0:c1]=nclickList[c2:]
            self.neyeList[0:c1]=neyeList[c2:]
            self.lastPList[0:c1]=lastPList[c2:]
            self.nlastPList[0:c1]=nlastPList[c2:]

            self.person[0:c1]=person[c2:]
            self.nperson[0:c1]=nperson[c2:]

            if not self.isFull:
                self.holding=self.capacity
            self.cursor=c1
        else:
            self.lengths[self.cursor:cursor]=lengths
            self.nlengths[self.cursor:cursor]=nlenghts
            self.clickList[self.cursor:cursor]=clickList
            self.eyeList[self.cursor:cursor]=eyeList
            self.actionList[self.cursor:cursor]=actionList
            self.maskList[self.cursor:cursor]=maskList
            self.nclickList[self.cursor:cursor]=nclickList
            self.neyeList[self.cursor:cursor]=neyeList
            self.rewardList[self.cursor:cursor]=rewardList
            self.lastPList[self.cursor:cursor]=lastPList
            self.nlastPList[self.cursor:cursor]=nlastPList

            self.person[self.cursor:cursor]=person
            self.nperson[self.cursor:cursor]=nperson

            self.cursor=cursor%self.capacity
            if not self.isFull:
                if cursor==self.capacity:
                    self.holding=self.capacity
                    self.isFull=True
                else:
                    self.holding=self.cursor
        

    # click to device 
    #  clickList 0 ,eyeList 1,lengths 2,actionList 3,maskList 4,rewardList 5,nclickList 6,neyeList 7 lengths 8
    def sample(self,batchSize):
        indexes=np.random.randint(low=0,high=self.holding,size=batchSize).tolist()
        # batch=np.random.sample(self.dataList,batchSize)
        # batch.sort(key=lambda x: x[2],reverse=True)
        eyeList=self.eyeList[indexes]
        clickList=self.clickList[indexes]
        lengths=self.lengths[indexes]
        actionList=self.actionList[indexes]
        maskList=self.maskList[indexes]
        rewardList=self.rewardList[indexes]
        nclickList=self.nclickList[indexes]
        neyeList=self.neyeList[indexes]
        nlengths=self.nlengths[indexes]
        lastPList=self.lastPList[indexes]
        nlastPList=self.nlastPList[indexes]
        person=self.person[indexes]
        nperson=self.nperson[indexes]


        actionList=torch.tensor(actionList,dtype=torch.long).reshape(-1,1,1).to(self.device)
        rewardList=torch.tensor(rewardList,dtype=torch.float32).reshape(-1,1,1).to(self.device)
        maskList=torch.tensor(maskList,dtype=torch.float32).reshape(-1,1,1).to(self.device)
        clickList=torch.squeeze(clickList,dim=1)
        nclickList=torch.squeeze(nclickList,dim=1)
        lastPList=torch.squeeze(lastPList,dim=1)
        nlastPList=torch.squeeze(nlastPList,dim=1)
        eyeList=torch.squeeze(eyeList,dim=1)
        neyeList=torch.squeeze(neyeList,dim=1)
        person=torch.unsqueeze(person,dim=1)
        nperson=torch.unsqueeze(nperson,dim=1)

        return clickList,eyeList,lastPList,lengths,person,actionList,rewardList,maskList,nclickList,neyeList,nlastPList,nlengths,nperson

# scene
class ReplayBufferRNN2(object):
    #click->3 eye->3 aciton->1 nclick->3 eye->3 mask->1 reward->1 seq->3 nseq->3
    #click->3 eye->3 goal->1 action->1 seq->3 mask->1 reward->1 nclick->3 neye->3 nseq->3
    def __init__(self,maxLen,device) -> None:
        self.capacity=maxLen
        self.lengths=torch.empty((maxLen),dtype=torch.long)
        self.nlengths=torch.empty((maxLen),dtype=torch.long)

        self.person=torch.empty((maxLen,1),dtype=torch.long).to(device)
        self.nperson=torch.empty((maxLen,1),dtype=torch.long).to(device)

        self.scene=torch.empty((maxLen,1),dtype=torch.long).to(device)
        self.nscene=torch.empty((maxLen,1),dtype=torch.long).to(device)

        self.clickList=torch.empty((maxLen,3),dtype=torch.long).to(device)
        self.eyeList=torch.empty((maxLen,10),dtype=torch.long).to(device)
        self.lastPList=torch.empty((maxLen,3),dtype=torch.long).to(device)
        self.actionList=np.empty((maxLen),dtype=np.int64)
        self.maskList=np.empty((maxLen),dtype=np.int64)
        self.rewardList=np.empty((maxLen),dtype=np.int64)
        self.nclickList=torch.empty((maxLen,3),dtype=torch.long).to(device)
        self.neyeList=torch.empty((maxLen,10),dtype=torch.long).to(device)
        self.nlastPList=torch.empty((maxLen,3),dtype=torch.long).to(device)

        self.cursor=0
        self.holding=0
        self.isFull=False
        self.device=device
        self.ratio=0.


    def refresh(self):
        device=self.device
        maxLen=self.capacity
        self.lengths=torch.empty((maxLen),dtype=torch.long)
        self.nlengths=torch.empty((maxLen),dtype=torch.long)

        self.person=torch.empty((maxLen,1),dtype=torch.long).to(device)
        self.nperson=torch.empty((maxLen,1),dtype=torch.long).to(device)

        self.scene=torch.empty((maxLen,1),dtype=torch.long).to(device)
        self.nscene=torch.empty((maxLen,1),dtype=torch.long).to(device)

        self.clickList=torch.empty((maxLen,3),dtype=torch.long).to(device)
        self.eyeList=torch.empty((maxLen,10),dtype=torch.long).to(device)
        self.lastPList=torch.empty((maxLen,3),dtype=torch.long).to(device)
        self.actionList=np.empty((maxLen),dtype=np.int64)
        self.maskList=np.empty((maxLen),dtype=np.int64)
        self.rewardList=np.empty((maxLen),dtype=np.int64)
        self.nclickList=torch.empty((maxLen,3),dtype=torch.long).to(device)
        self.neyeList=torch.empty((maxLen,10),dtype=torch.long).to(device)
        self.nlastPList=torch.empty((maxLen,3),dtype=torch.long).to(device)

        self.cursor=0
        self.holding=0
        self.isFull=False
        self.ratio=0.


    def getRatio(self):
        self.ratio=self.holding/self.capacity
        return self.ratio
    
    # memoList -> [[click],[eye],[length(eye)]] ->tensor
    # clickList,eyeList,actionList,maskList,rewardList,nclickList,neyeList
    def push(self,memoTuple:tuple):
        clickList,eyeList,lastPList,lengths,person,scene,goalList,actionList,rewardList,maskList,nclickList,neyeList,nlastPList,nlenghts,nperson,nscene=memoTuple
        cursor=len(lengths)+self.cursor
        if cursor>self.capacity:
            c1=cursor-self.capacity
            c2=len(lengths)-c1
            self.lengths[self.cursor:]=lengths[0:c2]
            self.nlengths[self.cursor:]=nlenghts[0:c2]
            self.clickList[self.cursor:]=clickList[0:c2]
            self.eyeList[self.cursor:]=eyeList[0:c2]
            self.actionList[self.cursor:]=actionList[0:c2]
            self.maskList[self.cursor:]=maskList[0:c2]
            self.nclickList[self.cursor:]=nclickList[0:c2]
            self.neyeList[self.cursor:]=neyeList[0:c2]
            self.rewardList[self.cursor:]=rewardList[0:c2]
            self.lastPList[self.cursor:]=lastPList[0:c2]
            self.nlastPList[self.cursor:]=nlastPList[0:c2]
            self.person[self.cursor:]=person[0:c2]
            self.nperson[self.cursor:]=nperson[0:c2]

            self.scene[self.cursor:]=scene[0:c2]
            self.nscene[self.cursor:]=nscene[0:c2]

            self.lengths[0:c1]=lengths[c2:]
            self.nlengths[0:c1]=nlenghts[c2:]
            self.clickList[0:c1]=clickList[c2:]
            self.eyeList[0:c1]=eyeList[c2:]
            self.actionList[0:c1]=actionList[c2:]
            self.maskList[0:c1]=maskList[c2:]
            self.rewardList[0:c1]=rewardList[c2:]
            self.nclickList[0:c1]=nclickList[c2:]
            self.neyeList[0:c1]=neyeList[c2:]
            self.lastPList[0:c1]=lastPList[c2:]
            self.nlastPList[0:c1]=nlastPList[c2:]
            self.person[0:c1]=person[c2:]
            self.nperson[0:c1]=nperson[c2:]
            
            self.scene[0:c1]=scene[c2:]
            self.nscene[0:c1]=nscene[c2:]

            if not self.isFull:
                self.holding=self.capacity
            self.cursor=c1
        else:
            self.lengths[self.cursor:cursor]=lengths
            self.nlengths[self.cursor:cursor]=nlenghts
            self.clickList[self.cursor:cursor]=clickList
            self.eyeList[self.cursor:cursor]=eyeList
            self.actionList[self.cursor:cursor]=actionList
            self.maskList[self.cursor:cursor]=maskList
            self.nclickList[self.cursor:cursor]=nclickList
            self.neyeList[self.cursor:cursor]=neyeList
            self.rewardList[self.cursor:cursor]=rewardList
            self.lastPList[self.cursor:cursor]=lastPList
            self.nlastPList[self.cursor:cursor]=nlastPList

            self.person[self.cursor:cursor]=person
            self.nperson[self.cursor:cursor]=nperson
            self.scene[self.cursor:cursor]=scene
            self.nscene[self.cursor:cursor]=nscene
            self.cursor=cursor%self.capacity
            if not self.isFull:
                if cursor==self.capacity:
                    self.holding=self.capacity
                    self.isFull=True
                else:
                    self.holding=self.cursor
        

    # click to device 
    #  clickList 0 ,eyeList 1,lengths 2,actionList 3,maskList 4,rewardList 5,nclickList 6,neyeList 7 lengths 8
    def sample(self,batchSize):

        indexes=np.random.randint(low=0,high=self.holding,size=batchSize).tolist()

        eyeList=self.eyeList[indexes]
        clickList=self.clickList[indexes]
        lengths=self.lengths[indexes]
        actionList=self.actionList[indexes]
        maskList=self.maskList[indexes]
        rewardList=self.rewardList[indexes]
        nclickList=self.nclickList[indexes]
        neyeList=self.neyeList[indexes]
        nlengths=self.nlengths[indexes]
        lastPList=self.lastPList[indexes]
        nlastPList=self.nlastPList[indexes]
        person=self.person[indexes]
        nperson=self.nperson[indexes]

        scene=self.scene[indexes]
        nscene=self.nscene[indexes]


        actionList=torch.tensor(actionList,dtype=torch.long).reshape(-1,1,1).to(self.device)
        rewardList=torch.tensor(rewardList,dtype=torch.float32).reshape(-1,1,1).to(self.device)
        maskList=torch.tensor(maskList,dtype=torch.float32).reshape(-1,1,1).to(self.device)
        clickList=torch.squeeze(clickList,dim=1)
        nclickList=torch.squeeze(nclickList,dim=1)
        lastPList=torch.squeeze(lastPList,dim=1)
        nlastPList=torch.squeeze(nlastPList,dim=1)
        eyeList=torch.squeeze(eyeList,dim=1)
        neyeList=torch.squeeze(neyeList,dim=1)
        person=torch.unsqueeze(person,dim=1)
        nperson=torch.unsqueeze(nperson,dim=1)

        scene=torch.unsqueeze(scene,dim=1)
        nscene=torch.unsqueeze(nscene,dim=1)

        return clickList,eyeList,lastPList,lengths,person,scene,actionList,rewardList,maskList,nclickList,neyeList,nlastPList,nlengths,nperson,nscene
    
    # def loadBuffer(self,loadPath):
    #     f=open(loadPath,'r')
    #     data=json.load(f)
    #     self.isFull=bool(data.get('isFull'))
    #     self.capacity=int(data.get('capacity'))
    #     self.cursor=int(data.get('cursor'))
    #     if self.isFull:
    #         for i in range(self.capacity):
    #             self.eyeList[i]=torch.tensor(data.get(str(i)).get('eyeList'),dtype=torch.long)
    #             subDict['eyeList']=eyeList[i]
    #             subDict['clickList']=clickList[i]
    #             subDict['lengths']=lengths[i]
    #             subDict['actionList']=actionList[i]
    #             subDict['maskList']=maskList[i]
    #             subDict['rewardList']=rewardList[i]
    #             subDict['nclickList']=nclickList[i]
    #             subDict['neyeList']=neyeList[i]
    #             subDict['nlengths']=nlengths[i]
    #             subDict['lastPList']=lastPList[i]
    #             subDict['nlastPList']=nlastPList[i]
    #             subDict['person']=person[i]
    #             subDict['nperson']=nperson[i]
    #             subDict['scene']=scene[i]
    #             subDict['nscene']=nscene[i]
    #     else:

    # def saveBuffer(self,savePath):
    #     eyeList=self.eyeList.cpu().detach().numpy().tolist()
    #     clickList=self.clickList.cpu().detach().numpy().tolist()
    #     lengths=self.lengths.cpu().detach().numpy().tolist()
    #     actionList=self.actionList.cpu().detach().numpy().tolist()
    #     maskList=self.maskList.cpu().detach().numpy().tolist()
    #     rewardList=self.rewardList.cpu().detach().numpy().tolist()
    #     nclickList=self.nclickList.cpu().detach().numpy().tolist()
    #     neyeList=self.neyeList.cpu().detach().numpy().tolist()
    #     nlengths=self.nlengths.cpu().detach().numpy().tolist()
    #     lastPList=self.lastPList.cpu().detach().numpy().tolist()
    #     nlastPList=self.nlastPList.cpu().detach().numpy().tolist()
    #     person=self.person.cpu().detach().numpy().tolist()
    #     nperson=self.nperson.cpu().detach().numpy().tolist()

    #     scene=self.scene.cpu().detach().numpy().tolist()
    #     nscene=self.nscene.cpu().detach().numpy().tolist()
    #     saveDict={}
    #     saveDict['isFull']=self.isFull
    #     saveDict['capacity']=self.capacity
    #     saveDict['cursor']=self.cursor
    #     if self.isFull:
    #         subDict={}
    #         for i in range(self.capacity):
    #             subDict['eyeList']=eyeList[i]
    #             subDict['clickList']=clickList[i]
    #             subDict['lengths']=lengths[i]
    #             subDict['actionList']=actionList[i]
    #             subDict['maskList']=maskList[i]
    #             subDict['rewardList']=rewardList[i]
    #             subDict['nclickList']=nclickList[i]
    #             subDict['neyeList']=neyeList[i]
    #             subDict['nlengths']=nlengths[i]
    #             subDict['lastPList']=lastPList[i]
    #             subDict['nlastPList']=nlastPList[i]
    #             subDict['person']=person[i]
    #             subDict['nperson']=nperson[i]
    #             subDict['scene']=scene[i]
    #             subDict['nscene']=nscene[i]
    #         saveDict[str(i)]=subDict
    #     else:
    #         subDict={}
    #         for i in range(self.cursor):
    #             subDict['eyeList']=eyeList
    #             subDict['clickList']=clickList
    #             subDict['lengths']=lengths
    #             subDict['actionList']=actionList
    #             subDict['maskList']=maskList
    #             subDict['rewardList']=rewardList
    #             subDict['nclickList']=nclickList
    #             subDict['neyeList']=neyeList
    #             subDict['nlengths']=nlengths
    #             subDict['lastPList']=lastPList
    #             subDict['nlastPList']=nlastPList
    #             subDict['person']=person
    #             subDict['nperson']=nperson
    #             subDict['scene']=scene
    #             subDict['nscene']=nscene
    #         saveDict[str(i)]=subDict
    #     f=open(savePath,'w')
    #     json.dump(saveDict,f,indent=2)
    #     f.close()


from functools import cmp_to_key
def cmp(x,y):
    if len(x[0])!=len(y[0]):
        if len(x[0])>len(y[0]):
            return 1
        else:
            return -1
    else:
        if x>y:
            return 1
        else:
            return -1
# scene
class DQNRNNTrajectory2(object):
    def __init__(self) -> None:
        self.tras=[]
        self.newTras=[]
        self.rewardFun=DQNReward()
        self.aveLen=[]
        # self.distribution=[0 for i in range(250)]
        self.counting=0
        self.noActionNum=0
        self.noActionNumWithThreshold=0.
        self.corrects=[]
        self.noAction=[]
        self.action=[]
        self.noActionCorrect=[]
        self.probs=[]
        self.timeStamps=[]

    # TD(N) N->(0)
    # 得到 td 序列
    def getComTraZero(self):
        self.counting+=1
        traLen=len(self.newTras)
        # zTra=[]
        lengths=[]
        nlenghts=[]
        clickList=[]
        eyeList=[]
        goalList=[]
        actionList=[]
        maskList=[]
        rewardList=[]
        nclickList=[]
        lastPList=[]
        nlastPList=[]
        neyeList=[]
        personList=[]
        npersonList=[]
        sceneList=[]
        nsceneList=[]
        for i in range(traLen-1):
            click,eye,goal,action,mask,reward,lastP,length,person,scene=self.newTras[i]

            nclick,neye,ngoal,naction,nmask,nreward,nlastP,nlength,npersion,nscene=self.newTras[i+1]
            clickList.append(click) # tensor
            eyeList.append(eye) # tensor
            goalList.append(goal)# long
            actionList.append(action) # long
            maskList.append(mask)# float32
            rewardList.append(reward)# float32
            nclickList.append(nclick)# tensor
            neyeList.append(neye)# tensor
            nlenghts.append(nlength)# long
            lengths.append(length)# long
            lastPList.append(lastP)
            nlastPList.append(nlastP)
            sceneList.append(scene)
            nsceneList.append(nscene)
            personList.append(person)
            npersonList.append(npersion)
        click,eye,goal,action,mask,reward,lastP,length,persion,scene=self.newTras[-1]
        if int(0.5+mask)==0:
            # length=torch.tensor(len(eye),dtype=torch.long)
            # nlength=torch.tensor(len(eye),dtype=torch.long)
            nclick=torch.tensor([0 for i in range(len(click))],dtype=torch.long).to(click.device)
            neye=torch.tensor([0 for i in range(len(eye))],dtype=torch.long).to(eye.device)
            nlastP=torch.tensor([0 for i in range(len(lastP))],dtype=torch.long).to(lastP.device)
            # zTra.append([click,eye,length,goal,action,0,reward,nclick,neye,nlength])
            clickList.append(click) # tensor
            eyeList.append(eye) # tensor
            goalList.append(goal)# long
            actionList.append(action) # long
            maskList.append(mask)# float32
            rewardList.append(reward)# float32
            nclickList.append(nclick)# tensor
            neyeList.append(neye)# tensor
            nlenghts.append(length)# long
            lengths.append(length)# long
            lastPList.append(lastP)
            nlastPList.append(nlastP)

            personList.append(persion)
            npersonList.append(persion)

            sceneList.append(scene)
            nsceneList.append(scene)

        lengths=torch.stack(lengths,dim=0).squeeze()
        nlenghts=torch.stack(nlenghts,dim=0).squeeze()

        npersonList=torch.stack(npersonList,dim=0)
        personList=torch.stack(personList,dim=0)

        sceneList=torch.stack(sceneList,dim=0)
        nsceneList=torch.stack(nsceneList,dim=0)

        clickList=torch.stack(clickList,dim=0)
        nclickList=torch.stack(nclickList,dim=0)
        lastPList=torch.stack(lastPList,dim=0)
        nlastPList=torch.stack(nlastPList,dim=0)

        eyeList=torch.stack(eyeList,dim=0)
        neyeList=torch.stack(neyeList,dim=0)
        lengths=lengths.reshape(-1)
        nlenghts=nlenghts.reshape(-1)
        return clickList,eyeList,lastPList,lengths,personList,sceneList,goalList,actionList,\
        rewardList,maskList,nclickList,neyeList,nlastPList,nlenghts,npersonList,nsceneList

    # n-> n+1 steps
    # def getComTraN(self,N=2):
    #     traLen=len(self.newTras)
    #     self.counting+=1
    #     # print(f"new tras {self.newTras} zero {self.newTras[0]}")
    #     lastClick=[0 for i in range(len(self.newTras[0][0]))]
    #     lastEye=[0 for  i in range(len(self.newTras[0][1]))]
    #     lastSeq=[0 for i in range(len(self.newTras[0][-1]))]
    #     NTras=[]
    #     #seq should be 3
    #     for i in range(traLen,-1,-1):
    #         if i<=N:
    #             break
    #         if i==traLen:
    #             clickL,eyeL,seqL=lastClick,lastEye,lastSeq
    #         else:
    #             clickL,eyeL,goalL,actionL,maskL,rewardL,seqL=self.newTras[i]
    #         maskAcu=1.
    #         rewardAcu=0.
    #         for j in range(1,N+2):
    #             click,eye,goal,action,mask,reward,seq=self.newTras[i-j]
    #             rewardAcu=rewardAcu*mask+reward
    #             maskAcu*=mask
    #         NTras.append([click,eye,goal,action,seq,maskAcu,rewardAcu,clickL,eyeL,seqL])
    #     return NTras
            

    def getNewTras(self):
        traLen=len(self.tras)
        self.aveLen.append(traLen)
        # self.distribution[traLen]+=1
        present=self.tras[0][0][-1]
        presents=[]
        # self.rewardFun.initPrediction(self.tras[0][0][0])
        errorFlags=[]
        self.rewardFun.errors=0
        self.rewardFun.lastErrors=0
        self.rewardFun.prediction=[]
        for i in range(traLen):
            click,eye,goal,action,mask,lastP,length,person,scene=self.tras[i]
            # print(f'action {action} p {presents}')
            presents.append(present)
            if action!=12:
                # print(f'action {action}')
                present=action
            else:
                present=lastP.squeeze().cpu().detach().numpy()[-1]
                # print(f'present {present} {lastP}')
                # breakpoint()
            self.rewardFun.addPrediction(present)
            if len(self.rewardFun.prediction)>=3:
                errorFlags.append(self.rewardFun.judgeError())
            else:
                errorFlags.append(True)
        self.rewardFun.lastErrors=self.rewardFun.errors
        self.rewardFun.errors=0
        self.rewardFun.prediction=[]
        # print(f'presnts0 {presents}')
        for i in range(traLen):
            # predict,goal,mask,present,judgeFlag
            click,eye,goal,action,mask,lastP,length,person,scene=self.tras[i]

            reward=self.rewardFun.getReward(action,goal,mask,presents[i],errorFlags[i])
            if traLen>1:
                if action==12:
                    if (i+1)/traLen<0.8:
                        self.noActionNumWithThreshold+=1
                    self.noActionNum+=1

            self.newTras.append([click,eye,goal,action,mask,reward,lastP,length,person,scene])
    
    def push(self,*data):
        click,eye,goal,action,mask,lastP,length,person,scene=dp(data)
        # traNum=len(self.tras)
        length=length.reshape(-1)
        self.tras.append([click,eye,goal,action,mask,lastP,length,person,scene])
    
    def pushProbs(self,prob):
        self.probs.append(prob)
    
    def pushTimeStamp(self,timeStamp):
        self.timeStamps.append(timeStamp)
    

    def setGoal(self,goal):
        for tra in self.tras:
            tra[2]=goal
    def setMask(self,index,mask):
        self.tras[index][4]=mask
    
    # after get new tras
    '''
    no action ratio before 80% with length beyond 80% length beyond 1
    errors ratio in the len beyond 3
    in,out elimilated no actions
    rewards ,endrewards
    '''
    def getInfo(self):
        _in,_out,inOut=0.,0.,0.
        inNoAct,outNoAct=0.,0.
        rewards,endRewards=0.,0.
        # print(self.newTras[0][0])
        present=self.newTras[0][0][-1].cpu().detach().numpy()
        lastWithNoAction=0
        endInOutNoAction=-1
        for tra in self.newTras:
            click,eye,goal,action,mask,reward,lastP,length,person,scene=tra
            rewards+=reward
            if action==12:
                self.noAction.append(1)
            else:
                self.noAction.append(0)
            if action!=12:
                if action==goal:
                    self.noActionCorrect.append(1)
                    inNoAct+=1
                else:
                    outNoAct+=1
                    self.noActionCorrect.append(0)
            else:
                self.noActionCorrect.append(-1)
            # if action==12 and 
            if int(0.5+mask)==0:
                if action==12:
                    lastWithNoAction+=1# error name
                endRewards+=reward
            if action!=12:
                present=action
            if present==goal:
                _in+=1
                self.corrects.append(1)
                if int(0.5+mask)==0:
                    inOut=1
                    if action!=12:
                        endInOutNoAction=1
            else:
                self.corrects.append(0)
                _out+=1
                if int(0.5+mask)==0 and action!=12:
                    endInOutNoAction=0
        '''
        no action ratio before 80% with length beyond 80% length beyond 1
        errors ratio in the len beyond 3
        in,out elimilated no actions
        rewards ,endrewards
        '''
        traLen=len(self.newTras)
        return traLen,self.noActionNum,self.noActionNumWithThreshold,lastWithNoAction,\
            _in,_out,inOut,reward,endRewards,inNoAct,outNoAct,self.rewardFun.lastErrors,endInOutNoAction

    def getInfo2(self):
        present=self.tras[0][0][-1].cpu().detach().numpy()
        IPA2=0
        totalCorrect=0
        endCorrect=0
        totalReward=0
        endReward=0
        for tra in self.newTras:
            click,eye,goal,action,mask,reward,lastP,length,person,scene=tra
            if action!=12:
                present=action
            if present==goal:
                if IPA2==0:
                    IPA2=1
                totalCorrect+=1
                if int(mask+0.5)==0:
                    endCorrect+=1
            totalReward+=reward
            if int(mask+0.5)==0:
                endReward=reward
        
        return len(self.tras),totalCorrect,endCorrect,totalReward,endReward,IPA2
            


    def getCorrects(self):
        return self.corrects

    def everyStepsStr(self):
        ansStr=''
        for tra,prob,timeStamp in zip(self.tras,self.probs,self.timeStamps):
            ansStr+=str(tra)+'\n'+str(prob)+'\n'+str(timeStamp) +'\n'
        ansStr+='\n'
        return ansStr

    def clear(self):
        self.tras=[]
        self.newTras=[]
        self.noActionNum=0.
        self.noActionNumWithThreshold=0.
        self.corrects=[]
        self.noAction=[]
        self.action=[]
        self.noActionCorrect=[]
        self.probs=[]
        self.timeStamps=[]
    
    
    # def __del__(self):
    #     print(f'ave len: {np.mean(self.aveLen)} max len: {np.max(self.aveLen)} min len: {np.min(self.aveLen)} total: {np.sum(self.distribution)} counting: {self.counting}')
    #     print('distribution:')
    #     for i in range(25):
    #         for j in range(10):
    #             print(f'nums {i*10+j} {self.distribution[i*10+j]}',end=' ')
    #         print('')


class DataCollector(object):
    #所有指针都指向下一个元素
    def __init__(self,base_path,num=0,scene=0,MODE=1,restrict=False):
        self.env=DQNRNNEnv(base_path=base_path,restrict=restrict)
        self.num=num
        self.scene=scene
        self.file_finish = False
        self.files_len=0
        self.df=None
        self.current_file=-1
        self.cfile_len=0
        self.last_goal_list=[]
        self.goal=[0.,0.]

        self.MODE=MODE

        self.lastEye=-1
 
        self.goal_nums=0
        self.states_idx=0
        self.states_len=0
        self.files_path=[]
        self.base_path=base_path
        self.begin_idx=0
        self.end_idx=0
        # data
        self.saveDistance=[]
        self.bestSaving=[]
        self.curve=[]
        self.straight=[]
        self.movingTime=[]
        self.groundTruth=[]
        self.bestSaving=[]
        self.totalIn=[]
        self.totalOut=[]
        self.endIn=[]
        self.endOut=[]

        self.restrict=restrict
        self.restrictArea=[0 ,1, 2 ,3 ,6, 9]

        self.clickRegion=Region('click')
        self.eyeRegion=Region('eye')
        self.clickRegion.setRegions(CLICKAREAS)
        self.eyeRegion.setRegions(EYEAREAS)
        self.beginPoint=[]
        self.endPoint=[]
        self.beginTimestamp=''
        self.endTimestamp=''

 
        # 1-> origin  2-> not origin
    
    def setMODE(self,MODE):
        self.MODE=MODE

    def load_dict(self):
        dirs=os.listdir(self.base_path)
        for dir in dirs:
            if len(dir)<15:
                continue
            if dir[-6]=='_':
                if dir[-5]==str(self.MODE):
                    self.files_path.append(os.path.join(self.base_path,dir))
            else:
                self.files_path.append(os.path.join(self.base_path,dir))
       
        self.current_dir_len=len(self.files_path)
        self.saveDistance=[0 for i in range(self.current_dir_len)]
        self.curve=[0 for i in range(self.current_dir_len)]
        self.straight=[0 for i in range(self.current_dir_len)]
        self.movingTime=[0 for i in range(self.current_dir_len)]
        self.groundTruth=[0 for i in range(self.current_dir_len)]
        self.bestSaving=[0 for i in range(self.current_dir_len)]

        self.totalIn=[0 for i in range(self.current_dir_len)]
        self.totalOut=[0 for i in range(self.current_dir_len)]
        self.endIn=[0 for i in range(self.current_dir_len)]
        self.endOut=[0 for i in range(self.current_dir_len)]

    def get_file(self, appoint_path=None):
        if appoint_path is not None:
            self.df=pd.read_csv(self.files_path[self.current_file],header=None)
            return True
        self.current_file += 1
        if self.current_file >= self.current_dir_len:
            self.finish = True
            return False
        self.df=pd.read_csv(self.files_path[self.current_file],header=None)
        self.round_refresh()
        self.cfile_len = len(self.df)
        return True

    def round_refresh(self):
        self.file_finish = False
        self.files_len=0
        self.cfile_len=0
        self.last_goal_list=[]
        self.goal=[0.,0.]
        self.goal_nums=0

        self.states_idx=0
        self.states_len=0
        self.begin_idx=0
        self.end_idx=0

        self.lengths=[]
        self.lastEye=-1
    
    def refresh(self):
        self.file_finish = False
        self.files_len=0
        self.df=None
        self.current_file=-1
        self.cfile_len=0
        self.last_goal_list=[]
        self.goal=0.

        self.goal_nums=0
        self.next_state=[]
        self.files_path=[]
        self.states_idx=0
        self.states_len=0
        self.begin_idx=0
        self.end_idx=0

        self.lastEye=-1
        self.lengths=[]

    def get_goal(self):
        flag=-1
        goalFlag=False
        # self.cfile_len=len(self.df)
        if self.goal_nums>0:
            t=self.goal_nums-1
            while t>0:
                row=self.df.iloc[t]
                if np.isnan(row[5]) or np.isnan(row[1]) or np.isnan(row[2]) or np.isnan(row[3]) or np.isnan(row[4]):
                    t-=1
                flag=int(row[5])
                if flag==0:
                    
                    self.beginPoint=[row[3]*1.25,row[4]*1.25]
                    self.beginTimestamp=row[0]
                    break
                t-=1

        while self.goal_nums<self.cfile_len:

            row=self.df.iloc[self.goal_nums]
 
            if len(row)<6:
                continue
            if np.isnan(row[5]) or np.isnan(row[1]) or np.isnan(row[2]) or np.isnan(row[3]) or np.isnan(row[4]):
                self.goal_nums+=1
                continue  
            flag=row[5]
            if flag==0:
                goalFlag=True
                self.goal=self.clickRegion.judge(row[3]*1.25,row[4]*1.25)
                self.endPoint=[row[3]*1.25,row[4]*1.25]
                self.endTimestamp=row[0]
                break
            self.goal_nums+=1
        while flag ==0 and self.goal_nums<self.cfile_len:
            row=self.df.iloc[self.goal_nums]
            if np.isnan(row[5]) or np.isnan(row[1]) or np.isnan(row[2]) or np.isnan(row[3]) or np.isnan(row[4]):
                break
            flag=int(row[5])
            if flag !=0:
                break
            self.goal=self.clickRegion.judge(row[3]*1.25,row[4]*1.25)            
            self.goal_nums+=1
        if self.goal_nums>=self.cfile_len-1:
            if not goalFlag:
                self.file_finish=True
    
    def get_1round(self):
        t=0
        endFlag=False
        self.get3Goal()
        self.get_goal()
        nowGoal=self.last_goal_list[-1]
        if self.end_idx!=0:
            self.begin_idx=self.end_idx
        self.end_idx=self.goal_nums
        predictX,predictY=-1,-1
        predictFlag=False
        for i in range(self.begin_idx,self.end_idx):
            row=self.df.iloc[i]
            if len(row)<5:
                continue
            if np.isnan(
                row[5]) or np.isnan(row[1]) or np.isnan(row[2]) or np.isnan(row[3]) or np.isnan(row[4]):
                continue
                
            if int(row[8])==1:
                predictFlag=True
                predictX,predictY=int(row[6]),int(row[7])
                region=self.clickRegion.judge(predictX,predictY)
                predictAns=self.clickRegion.judgeHeart(region)
                predictX,predictY=predictAns[0],predictAns[1]
            if int(row[6])!=0 and int(row[7])!=0:
                nowRegion=self.clickRegion.judge(row[6],row[7])
                nowGoal=nowRegion
                if nowGoal==self.goal:
                    self.totalIn[self.current_file]+=1
                else:
                    self.totalOut[self.current_file]+=1
                
            if self.MODE==0:
                if int(row[6])!=0 and int(row[7])!=0:
                    predictX,predictY=int(row[6]),int(row[7])
                    if int(row[8])==1:
                        region=self.clickRegion.judge(predictX,predictY)
                        predictAns=self.clickRegion.judgeHeart(region)
                        predictX,predictY=predictAns[0],predictAns[1]
            if (i==self.end_idx-1) and (self.end_idx>=self.cfile_len-5):
                endFlag=True
                break
        if nowGoal==self.goal:
            self.endIn[self.current_file]+=1
        else:
            self.endOut[self.current_file]+=1
        # if predictFlag:
        if predictX!=-1 and predictY!=-1:
            endRegion=self.clickRegion.judge(self.endPoint[0],self.endPoint[1])
            originDistance=np.sqrt((self.beginPoint[0]-self.endPoint[0])**2+(self.beginPoint[1]-self.endPoint[1])**2)
            movingDistance=np.sqrt((self.endPoint[0]-predictX)**2+(self.endPoint[1]-predictY)**2)
            self.saveDistance[self.current_file]+=originDistance-movingDistance
            self.groundTruth[self.current_file]+=originDistance
            bestPredict=self.clickRegion.judgeHeart(endRegion)
            bestDis=np.sqrt((self.endPoint[0]-bestPredict[0])**2+(self.endPoint[1]-bestPredict[1])**2)
            self.bestSaving[self.current_file]+=originDistance-bestDis
            # if (originDistance-movingDistance)<0:
            print(self.beginPoint,self.endPoint,predictX,predictY,originDistance,movingDistance,originDistance-movingDistance,self.beginTimestamp,self.endTimestamp)
        else:
            # pass
            beginRegion=self.clickRegion.judge(self.beginPoint[0],self.beginPoint[1])
            endRegion=self.clickRegion.judge(self.endPoint[0],self.endPoint[1])
            if beginRegion!=endRegion:
                bestPredict=self.clickRegion.judgeHeart(endRegion)
                bestDis=np.sqrt((self.endPoint[0]-bestPredict[0])**2+(self.endPoint[1]-bestPredict[1])**2)
                originDistance=np.sqrt((self.beginPoint[0]-self.endPoint[0])**2+(self.beginPoint[1]-self.endPoint[1])**2)
                self.saveDistance[self.current_file]+=0
                self.groundTruth[self.current_file]+=originDistance
                self.bestSaving[self.current_file]+=originDistance-bestDis
                # print(self.beginPoint,self.endPoint,predictX,predictY,originDistance,beginRegion,endRegion)
        self.last_goal_list.append(self.goal)
        if self.end_idx>=self.cfile_len or endFlag:
            self.file_finish=True
        self.states_idx=0
   
    def tallySaving(self):

        self.load_dict()
        flag=self.get_file()
        print(self.files_path[self.current_file])
        if flag==False:
            print('error')
            return False,'error'
        while flag:
            self.get_1round()
            if self.file_finish:
                flag=self.get_file()
                if flag:
                    print(self.files_path[self.current_file])
        ansList=[]
        for f,dis,s,totalIn,totalout,endIn,endOut,bdis in zip(self.files_path,self.saveDistance,self.groundTruth,self.totalIn,self.totalOut,self.endIn,self.endOut,self.bestSaving):
            ansList.append([f,dis,s,totalIn,totalout,totalIn/(totalIn+totalout),endIn,endOut,endIn/(endIn+endOut),bdis])
        ansList1=sorted(ansList,key=cmp_to_key(cmp))
        return ansList1
    
    def tallyCurrentFileStraight(self):
        self.cfile_len=len(self.df)
        k=self.goal_nums
        beginX,beginY=-1,-1
        while k>0:
            row=self.df.iloc[k]
            if len(row)<5:
                continue
            if np.isnan(row[5]) or np.isnan(row[1]) or np.isnan(row[2]) or np.isnan(row[3]) or np.isnan(row[4]):
                continue
            if int(row[5])==0:
                beginX,beginY=int(1.25*row[3]),int(1.25*row[4])
                break
            k-=1

        for i in range(k,self.cfile_len):
            row=self.df.iloc[i]
            if len(row)<5:
                continue
            if np.isnan(row[5]) or np.isnan(row[1]) or np.isnan(row[2]) or np.isnan(row[3]) or np.isnan(row[4]):
                continue
            if int(row[5])==0:
                X,Y=int(1.25*row[3]),int(1.25*row[4])
                # print(self.files_path,self.current_file,self.current_dir_len)
                self.straight[self.current_file]+=np.sqrt((X-beginX)**2+(Y-beginY)**2)
                beginX,beginY=X,Y

    def get3Goal(self):
        if len(self.last_goal_list)<3:
            self.goal_nums=0
            self.last_goal_list=[]
        else:
            return self.goal_nums
        while len(self.last_goal_list)<3 and (not self.file_finish):
                self.get_goal()
                if (len(self.last_goal_list)==0 or self.last_goal_list[-1]!=self.goal) and (not self.file_finish):
                    # print(self.goal,self.cfile_len,self.goal_nums,self.files_path[self.current_file])
                    if (self.goal in self.restrictArea):
                        self.begin_idx=self.goal_nums
                        if self.restrict:
                            continue
                        else:
                            self.last_goal_list.append(self.goal)        
                    else:
                        self.last_goal_list.append(self.goal)
                
        # row=self.df.iloc[self.goal_nums]
        # # print(row[3]*1.25,row[4]*1.25)
        self.tallyCurrentFileStraight()
        # print(self.straight)
        return self.goal_nums

    def tallyOffLineCurveAndStraightLineAndMovingTime(self):
        # self.load_dict()
        # self.get_file()
        straight=[0 for i in range(self.current_dir_len)]
        movingTime=[0 for i in range(self.current_dir_len)]
        curve=[0 for i in range(self.current_dir_len)]
        for i in range(self.current_dir_len):
            self.current_file=i
            lastTime=0
            self.df=pd.read_csv(self.files_path[i],header=None)
            df=pd.read_csv(self.files_path[i],header=None)
            fileLen=len(df)
            # it is click coordinate
            lastClickX,lastClickY=-1,-1
            lastX,lastY=-1,-1
            # tally data when this flag decrease to zero
            self.goal_nums=0
            beginIndex=self.get3Goal()
            self.goal_nums=0
            t=beginIndex
            while True:
                row=df.iloc[t]
                if np.isnan(row[5]) or np.isnan(row[1]) or np.isnan(row[2]) or np.isnan(row[3]) or np.isnan(row[4]):
                    t+=1
                    continue
                break
            lastClickX,lastClickY=int(row[3]*1.25),int(row[4]*1.25)
            lastX,lastY=int(row[3]*1.25),int(row[4]*1.25)
            lastTime=int(row[0])
            for j in range(beginIndex,fileLen):
                row=df.iloc[j]
                if np.isnan(row[5]) or np.isnan(row[1]) or np.isnan(row[2]) or np.isnan(row[3]) or np.isnan(row[4]):
                    continue
                clickFlag=True if int(row[5])==0 else False
                # if clickFlag and tallyFlag[i]>0:
                #     tallyFlag[i]-=1
                #     lastClickX,lastClickY=int(row[3]*1.25),int(row[4]*1.25)
                #     lastX,lastY=int(row[3]*1.25),int(row[4]*1.25)
                #     lastTime=int(row[0])
                #     continue
                mouseX,mouseY=int(row[3]*1.25),int(row[4]*1.25)
                thisTime=int(row[0])
                curve[i]+=np.sqrt((mouseX-lastX)**2+(mouseY-lastY)**2)
                if mouseX!=lastX and mouseY!=lastY:
                    movingTime[i]+=thisTime-lastTime
                lastX,lastY=mouseX,mouseY
                lastTime=thisTime
                if clickFlag:
                    straight[i]+=np.sqrt((mouseX-lastClickX)**2+(mouseY-lastClickY)**2)
                    lastClickX,lastClickY=mouseX,mouseY
        
        ansList=[]
        for f,s,c,m in zip(self.files_path,straight,curve,movingTime):
            ansList.append([f,s,c,m])
        return ansList

    def tallyIPA1IPA2(self):
        pass

    def tallySavingDisFromAgent(self,agent,testAllEnvs,testAllNum,testAllScene,device,savePath,restrict):
        # print(restrict)
        self.restrict=restrict
        # print(self.restrict)
        # print(savePath)
        t=0
        clickRegion=Region('click')
        clickRegion.setRegions(CLICKAREAS)
        preTest=PresentsRecorder(len(testAllEnvs))
        agent.eval()
        savingDis=0
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        fileNum=len(os.listdir(savePath))
        savePath=os.path.join(savePath,str(fileNum))
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        savePath=os.path.join(savePath,'save.txt')
        f=open(savePath,'w')
        f.write('write\n')
        f.close()
        print(len(testAllEnvs))
        # breakpoint()
        total=0
        for i in range(len(testAllEnvs)):
            testAllEnvs[i].load_dict()
            testAllEnvs[i].get_file()
            testAllEnvs[i].disMode=True
        for i in range(len(testAllEnvs)): 
            savingDis=0
            nowPredict=[-1,-1]
            self.base_path=testAllEnvs[i].base_path
            totalDis=0
            while True:
                ans=testAllEnvs[i].act()
                if isinstance(ans,bool):
                    break
                else:
                    eye=torch.tensor([ans[0]],dtype=torch.long).to(device)
                    click=torch.tensor([ans[1]],dtype=torch.long).to(device)
                    lengths=torch.tensor([ans[4]],dtype=torch.long)
                    person=torch.tensor([[[testAllNum[i]]]],dtype=torch.long).to(device)
                    scene=torch.tensor([[[testAllScene[i]]]],dtype=torch.long).to(device)

                    if preTest.flag[i]==False:
                        preTest.init(i,ans[1])
                    lastP=torch.tensor([preTest.getLastPresents(i)],dtype=torch.long).to(device)
                    ansRegion=int(agent.act(click,eye,lastP,lengths,person,scene))
                    preTest.add(i,ansRegion)
                    if nowPredict[0]==-1:
                        subRegion=clickRegion.judgeHeart(ans[1][-1])
                        nowPredict[0],nowPredict[1]=subRegion[0],subRegion[1]
                    if ansRegion != 12:
                        heart=clickRegion.judgeHeart(ansRegion)
                        nowPredict[0],nowPredict[1]=heart[0],heart[1]
                    # ans -> eye click goal isEnd
                    # print(ans)
                    # breakpoint()
                    if ans[3]==1:
                        total+=1
                        if nowPredict[0]==-1 and nowPredict[1]==-1:
                            continue
                        beginPoint=dp(ans[-2])
                        endPoint=dp(ans[-1])
                        if beginPoint[0]==-1 and beginPoint[1]==-1:
                            beginPoint=clickRegion.judgeHeart(ans[1][-1])
                        dis=np.sqrt((beginPoint[0]-endPoint[0])**2+(beginPoint[1]-endPoint[1])**2)
                        trueDis=np.sqrt((nowPredict[0]-endPoint[0])**2+(nowPredict[1]-endPoint[1])**2)
                        savingDis+=(dis-trueDis)
                        totalDis+=dis
                        # print(beginPoint,endPoint,testAllEnvs[i].goal,ans)
                        # print(beginPoint,endPoint,nowPredict,dis,trueDis,dis-trueDis)
                        nowPredict=[-1,-1]
            # print(total)
            # breakpoint()
            self.files_path=dp(testAllEnvs[i].files_path)
            self.current_dir_len=len(self.files_path)
            # self.current_file=
            # lineInfo=self.tallyOffLineCurveAndStraightLineAndMovingTime()
            straight=0
            # for l in lineInfo:
            #     straight+=l[1]

            with open(savePath,'a',encoding='UTF-8') as f:
                f.write('num: '+str(testAllNum[i])+' scene: '+str(testAllScene[i])+' savingdis: '+str(savingDis/testAllEnvs[i].current_dir_len)+' straightDis: '+str(straight)+' totalDis: '+str(totalDis)+'\n')
            totalDis=0

  
EYEAREAS=[[0,0,1300,1080],[1300,0,1920,390],[1300,390,1920,1080],
          [0+1920,0,560+1920,260],[0+1920,260,560+1920,670],[0+1920,670,560+1920,1080],#left
          [560+1920,0,1360+1920,680],[560+1920,680,975+1920,1080],[975+1920,680,1380+1920,1080],#mid
          [1380+1920,0,1920+1920,260],[1380+1920,260,1920+1920,680],[1380+1920,680,1920+1920,1080]]# right


# 0 1 2 3 6 9
CLICKAREAS=[[0,0,1300,1080],[1300,0,1920,390],[1300,390,1920,1080],
          [0+1920,0,560+1920,260],[0+1920,260,560+1920,670],[0+1920,670,560+1920,1080],#left
          [560+1920,0,1360+1920,680],[560+1920,680,975+1920,1080],[975+1920,680,1380+1920,1080],#mid
          [1380+1920,0,1920+1920,260],[1380+1920,260,1920+1920,680],[1380+1920,680,1920+1920,1080]]# right


MOVEAREAS=[[686.0 , 532.0],[1569.0 , 243.0],[1536.0 , 569.0],
           [2241.0 , 179.0],[2212.0 , 490.0],[2220.0 , 817.0],
           [2880.0 , 610.0],[2684.0 , 871.0],[3051.0 , 875.0],
           [3542.0 , 179.0],[3574.0 , 413.0],[3567.0 , 820.0]]

class Region(object):

    def __init__(self,name):
        # 0-> round 1->rectangle
        self.regions=[]
        self.regionNum=[]
        self.regionHeart=[]
        self.counting=[]
        self.name=name


    def setRegions(self,regions):

        number=0
        for region in regions:
            self.regions.append(region)
            self.regionNum.append(number)
            number+=1
            self.regionHeart.append([0.5*(region[0]+region[2]),0.5*(region[1]+region[3])])
        self.counting=[0 for i in range(number+4)]
        self.regionHeart=MOVEAREAS
    
    def judge(self,x,y):
        # if self.name=='click':
        #     y=np.clip(y,0,864)
        # x=x*1.25
        # y=y*1.25
        for region,num in zip(self.regions,self.regionNum):
            if region[0]<=x<=region[2] and region[1]<=y<=region[3]:
                self.counting[num]+=1
                return num
        #  1
        # 2 4
        #  3
        flagX=0
        flagY=0
        if x<0:
            flagX=2
            x=int(np.abs(x))
        elif x>3840:
            flagX=4
            x-=3840
        if y<0:
            flagY=1
            y=int(np.abs(y))
        elif y>1080:
            flagY=3
            y-=1080
        if flagX!=0 and flagY!=0:
            if x>=y:
                flagY=0
            else:
                flagX=0
        # print(flagX,flagY)
        self.counting[len(self.regionNum)+flagX+flagY-1]+=1
        return len(self.regionNum)+flagX+flagY-1
        
    def judgeHeart(self, regionNum:int):
        # print(self.regionNum,self.regionHeart)
        # print(f'this is regionNum {regionNum}')
        for heart, num in zip(self.regionHeart, self.regionNum):
            # print(f'this is num {num}')
            if regionNum==num:
                return heart



    # def __del__(self):
    #     print(f'name: {self.name}')
    #     print('distribution:')
    #     for region,num,counting in zip(self.regions,self.regionNum,self.counting):
    #         print(f'region: {region} num: {num} counting: {counting}')
    #     print(f'region: {"top"} num: {len(self.regionNum)} counting: {self.counting[-4]}')
    #     print(f'region: {"left"} num: {len(self.regionNum)+1} counting: {self.counting[-3]}')
    #     print(f'region: {"bottom"} num: {len(self.regionNum)+2} counting: {self.counting[-2]}')
    #     print(f'region: {"right"} num: {len(self.regionNum)+3} counting: {self.counting[-1]}')

def getTimeStamp():
    import time
    import calendar
    now=time.gmtime()
    now=calendar.timegm(now)

    from  datetime import datetime
    timeFormat=datetime.fromtimestamp(now)
    ans=timeFormat.strftime('%Y%m%d%H%M%S')
    return ans

class DQNDataRecorder(object):
    def __init__(self,fillSize,size,callFunc,strFunc) -> None:
        self.list=[[0 for i in range(fillSize+1)] for i in range(size+1)]
        self.recordTimes=[[0 for i in range(fillSize+1)] for i in range(size+1)]
        self.size=size
        self.callFunc=callFunc
        self.strFunc=strFunc
    
    def callFun(self,data):
        self.callFunc(self.list,self.recordTimes,data)

    def __str__(self) -> str:
        dataStr='\n'
        for i in range(1,self.size+1):
            dataStr+=self.strFunc(self.list,self.recordTimes,i)+'\n'
        return dataStr

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


def tallyOfflineDis():
    envs=[]
    for i in range(17,22): 
        envs.append(str(i))
    testAllEnvs=[]
    testAllNum=[]
    testAllScene=[]
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    agent=REMAgent2(device=device)

    # agent.load('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20231222/offline/2/18/best.pt')
    basePath='/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20231222/offline/'
    restrict=[True,False,True,False]
    timeStamp=getTimeStamp()
    filePath=os.path.join('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/',timeStamp[0:-6],'dis')
    if not os.path.exists(filePath):
        os.makedirs(filePath)
    fileNum=len(os.listdir(filePath))
    filePath=os.path.join(filePath,str(fileNum))
    if not os.path.exists(filePath):
        os.makedirs(filePath)
    loadPaths=['/home/wu_tian_ci/drl_project/mymodel/dqn/dqnonline/onlinedata/20231228/model/191200/onlineModel.pt',
               '/home/wu_tian_ci/drl_project/mymodel/dqn/dqnonline/onlinedata/20231228/model/214020/onlineModel.pt',
               '/home/wu_tian_ci/drl_project/mymodel/dqn/dqnonline/onlinedata/20231229/model/003342/onlineModel.pt']
    testEnv=['23','24','25','26','27','28']
    for k in range(1,4):
        envs=testEnv[0:k*2]
        loadPath=loadPaths[k-1]
        agent.load(loadPath)
        for env in envs:
            testAllEnvs=[]
            testAllNum=[]
            testAllScene=[]
            savePath=os.path.join(filePath,str(k),env)
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            for scene in range(3,4):
                testEnvPath=os.path.join('/home/wu_tian_ci/dataset/mix',env,str(scene))
                testAllEnvs.append(DQNRNNEnv(base_path=testEnvPath,num=int(env),scene=int(scene),restrict=True))
                testAllNum.append(int(env))
                testAllEnvs[-1].shuffle=False
                testAllEnvs[-1].topN=0
                testAllEnvs[-1].eval=True
                testAllEnvs[-1].MODE=1
                testAllEnvs[-1].mode=True
                testAllScene.append(scene)
                testAllEnvs[-1].restrict=True
                testAllEnvs[-1].load_dict()
                testAllEnvs[-1].get_file()
                t=0
                while True:
                    flag=testAllEnvs[-1].act()
                    t+=1
                    if flag==False:
                        break
                    # print(flag)
                # print(t)
                # breakpoint()
            dt=DataCollector('',MODE=1)
            print(envs)
            dt.tallySavingDisFromAgent(agent,testAllEnvs,testAllNum,testAllScene,device,savePath,True)


if __name__ == '__main__':
    # tallyOfflineDis()
    # env=[]
    # for i in range(17,22): 
    #     env.append(str(i))

    timeStamp=getTimeStamp()[0:-6]
    # env=['33','34','37','38','41','23','27','28','30','32']
    env=[]
    for i in range(23,29):
        env.append(str(i))
    # env=['24']
    basePath='/home/wu_tian_ci/drl_project/mymodel/dqn/offlinedatatally'
    storePath=os.path.join(basePath,timeStamp)
    if not os.path.exists(storePath):
        os.makedirs(storePath)
    fileNum=len(os.listdir(storePath))
    storePath=os.path.join(storePath,str(fileNum))
    if not os.path.exists(storePath):
        os.makedirs(storePath)
    for i in range(len(env)):
        savePath=os.path.join(storePath,env[i])
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        writePath=os.path.join(savePath,'tally.txt')
        f=open(writePath,'w')
        f.write('offline\n')
        f.close()
        for m in [0,1]:
            with open(writePath,'a') as f:
                f.write(str(m)+'\n')
            for j in range(3,4):
                basePath=''
                basePath1='/home/wu_tian_ci/dataset/mix'
                basePath2='/home/wu_tian_ci/dataset/mix'
                filePath=os.path.join(basePath1,env[i],str(j))
                if not os.path.exists(filePath):
                    filePath=os.path.join(basePath2,env[i],str(j))
                # print(filePath)
                dt=DataCollector(filePath,MODE=m)
                dt.restrict=True
                saving=dt.tallySaving()
                savingDis,movingDis,movingT=[],[],[]
                for item in saving:
                    # print(item)
                    print(f'file{item[0]}')
                    print(item[5],item[8],item[1],item[2],item[3],item[4],item[6],item[7])
                    print(item[0],item[1],item[2],item[3],item[4],item[5],item[6],item[7],item[8])
                    with open(writePath,'a') as f:
                        f.write(item[0]+'\n'+str(item[5])+' '+str(item[8])+' '+str(item[1])+' '+str(item[2])+' '+str(item[3])+' '+str(item[4])+' '+\
                                str(item[6])+' '+str(item[7])+' '+str(item[-1])+'\n')
                    savingDis.append(item[1])
                # print(savingDis)
                # print(i,j,' ',int(np.mean(savingDis)))
        print('')
    

    
    