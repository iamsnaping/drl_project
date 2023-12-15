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
            # print(self.cursor,cursor,len(lengths),c1,c2,self.capacity)
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

class DQNRNNTrajectory(object):
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
    # TD(N) N->(0)


    # 
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
        for i in range(traLen-1):
            click,eye,goal,action,mask,reward,lastP,length,person=self.newTras[i]
            # print(click,eye,goal,action,mask,reward)
            nclick,neye,ngoal,naction,nmask,nreward,nlastP,nlength,npersion=self.newTras[i+1]
            # zTra.append([click,eye,len(eye),goal,action,mask,reward,nclick,neye,len(neye)])
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

            personList.append(person)
            npersonList.append(npersion)
        click,eye,goal,action,mask,reward,lastP,length,persion=self.newTras[-1]
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
        lengths=torch.stack(lengths,dim=0)
        nlenghts=torch.stack(nlenghts,dim=0)
        npersonList=torch.stack(npersonList,dim=0)
        personList=torch.stack(personList,dim=0)
        clickList=torch.stack(clickList,dim=0)
        nclickList=torch.stack(nclickList,dim=0)
        lastPList=torch.stack(lastPList,dim=0)
        nlastPList=torch.stack(nlastPList,dim=0)

        eyeList=torch.stack(eyeList,dim=0)
        neyeList=torch.stack(neyeList,dim=0)
        return clickList,eyeList,lastPList,lengths,personList,goalList,actionList,rewardList,maskList,nclickList,neyeList,nlastPList,nlenghts,npersonList

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
            click,eye,goal,action,mask,lastP,length,person=self.tras[i]
            # print(f'action {action} p {presents}')
            presents.append(present)
            if action!=12:
                present=action
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
            click,eye,goal,action,mask,lastP,length,person=self.tras[i]

            reward=self.rewardFun.getReward(action,goal,mask,presents[i],errorFlags[i])
            if traLen>1:
                if action==12:
                    if (i+1)/traLen<0.8:
                        self.noActionNumWithThreshold+=1
                    self.noActionNum+=1

            self.newTras.append([click,eye,goal,action,mask,reward,lastP,length,person])
    
    def push(self,*data):
        click,eye,goal,action,mask,lastP,length,person=dp(data)
        # traNum=len(self.tras)
        self.tras.append([click,eye,goal,action,mask,lastP,length,person])
    

    def setGoal(self,goal):
        for tra in self.tras:
            tra[2]=goal
    
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
            click,eye,goal,action,mask,reward,lastP,length,person=tra
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
            # print(f'present {present} goal{goal} action {action}')
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
            click,eye,goal,action,mask,reward,lastP,length,person=tra
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
        # print(self.tras,self.newTras)
        
        return len(self.tras),totalCorrect,endCorrect,totalReward,endReward,IPA2
            


    def getCorrects(self):
        return self.corrects

    def clear(self):
        self.tras=[]
        self.newTras=[]
        self.noActionNum=0.
        self.noActionNumWithThreshold=0.
        self.corrects=[]
        self.noAction=[]
        self.action=[]
        self.noActionCorrect=[]
    
    
    # def __del__(self):
    #     print(f'ave len: {np.mean(self.aveLen)} max len: {np.max(self.aveLen)} min len: {np.min(self.aveLen)} total: {np.sum(self.distribution)} counting: {self.counting}')
    #     print('distribution:')
    #     for i in range(25):
    #         for j in range(10):
    #             print(f'nums {i*10+j} {self.distribution[i*10+j]}',end=' ')
    #         print('')

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
    # TD(N) N->(0)


    # 
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
            # print(click,eye,goal,action,mask,reward)
            nclick,neye,ngoal,naction,nmask,nreward,nlastP,nlength,npersion,nscene=self.newTras[i+1]
            # zTra.append([click,eye,len(eye),goal,action,mask,reward,nclick,neye,len(neye)])
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

        lengths=torch.stack(lengths,dim=0)
        nlenghts=torch.stack(nlenghts,dim=0)

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
        return clickList,eyeList,lastPList,lengths,personList,sceneList,goalList,actionList,rewardList,maskList,nclickList,neyeList,nlastPList,nlenghts,npersonList,nsceneList

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
                present=action
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
        self.tras.append([click,eye,goal,action,mask,lastP,length,person,scene])
    

    def setGoal(self,goal):
        for tra in self.tras:
            tra[2]=goal
    
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
            # print(f'present {present} goal{goal} action {action}')
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
        # print(self.tras,self.newTras)
        
        return len(self.tras),totalCorrect,endCorrect,totalReward,endReward,IPA2
            


    def getCorrects(self):
        return self.corrects

    def clear(self):
        self.tras=[]
        self.newTras=[]
        self.noActionNum=0.
        self.noActionNumWithThreshold=0.
        self.corrects=[]
        self.noAction=[]
        self.action=[]
        self.noActionCorrect=[]
    
    
    # def __del__(self):
    #     print(f'ave len: {np.mean(self.aveLen)} max len: {np.max(self.aveLen)} min len: {np.min(self.aveLen)} total: {np.sum(self.distribution)} counting: {self.counting}')
    #     print('distribution:')
    #     for i in range(25):
    #         for j in range(10):
    #             print(f'nums {i*10+j} {self.distribution[i*10+j]}',end=' ')
    #         print('')


class DataTally(object):
    def __init__(self):
        ...

    
    # timeStamp, eyeX, eyeY, mouseX, mouseY, mouseS,predictX,predictY
    def tallyList(self,dataList):
        totalTime=0
        totalMove=0
        moveTimes=0
        lastX,lastY=-1,-1
        predictFlag=False
        for i in range(1,len(dataList)):
            x,y=int(dataList[i][3]),int(dataList[y])
            if x!=dataList[i-1][3] or y!=dataList[i-1][4]:
                if predictFlag:
                    if lastX!=x or lastY!=y:
                        moveTimes+=1
                        totalMove+=np.sqrt((x-lastX)**2+(y-lastY)**2)
                else:
                    moveTimes+=1
                    totalMove+=np.sqrt((x-dataList[i-1][3])**2+(y-dataList[i-1][4])**2)
                    totalTime+=dataList[i][0]-dataList[i-1][0]
            predictFlag=False
            if dataList[i][-1]!=0 and dataList[i][-2]!=0:
                lastX,lastY=dataList[i][-2],dataList[i][-1]
                predictFlag=True

        return totalTime,totalMove,moveTimes



    
    def tallyCSV(self,csvFilePath):
        df=pd.read_csv(csvFilePath,header=None)
        fileLen=len(df)
        totalTime=0
        totalMove=0
        moveTimes=0
        lastX,lastY=-1,-1
        predictFlag=False
        for i in range(1,fileLen):
            row=self.df.iloc[i]
            x,y=int(row[3]),int(row[4])
            if x!=df.iloc[i-1][3] or y!=df.iloc[i-1][4]:
                if predictFlag:
                    if lastX!=x or lastY!=y:
                        moveTimes+=1
                        totalMove+=np.sqrt((x-lastX)**2+(y-lastY)**2)
                else:
                    moveTimes+=1
                    totalMove+=np.sqrt((x-df.iloc[i-1][3])**2+(y-df.iloc[i-1][4])**2)
                    totalTime+=df.iloc[i][0]-df.iloc[i-1][0]
            predictFlag=False
            if df.iloc[i][-1]!=0 and df.iloc[i][-2]!=0:
                lastX,lastY=df.iloc[i][-2],df.iloc[i][-1]
                predictFlag=True

        return totalTime,totalMove,moveTimes


EYEAREAS=[[0,0,1300,1080],[1300,0,1920,390],[1300,390,1920,1080],
          [0+1920,0,560+1920,260],[0+1920,260,560+1920,670],[0+1920,670,560+1920,1080],#left
          [560+1920,0,1360+1920,680],[560+1920,680,975+1920,1080],[975+1920,680,1380+1920,1080],#mid
          [1380+1920,0,1920+1920,260],[1380+1920,260,1920+1920,680],[1380+1920,680,1920+1920,1080]]# right

CLICKAREAS=[[0,0,1300,1080],[1300,0,1920,390],[1300,390,1920,1080],
          [0+1920,0,560+1920,260],[0+1920,260,560+1920,670],[0+1920,670,560+1920,1080],#left
          [560+1920,0,1360+1920,680],[560+1920,680,975+1920,1080],[975+1920,680,1380+1920,1080],#mid
          [1380+1920,0,1920+1920,260],[1380+1920,260,1920+1920,680],[1380+1920,680,1920+1920,1080]]# right
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
            self.regionHeart=[0.5*(region[0]+region[2]),0.5*(region[1]+region[3])]
        self.counting=[0 for i in range(number+4)]
    
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

   

if __name__ == '__main__':
    print(getTimeStamp()[4:-6])
    # clickRegion=Region('click')
    # clickRegion.setRegions(CLICKAREAS)
    # # 2975.0, 488.0
    # print(clickRegion.judge(462.0, 996.0))