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
    
    def addClick(self,click):
        if len(self.click)<3:
            self.click.append(click)
            self.pList.append(click)
            return False,len(self.click)
        else:
            clickList=dp(self.click)
            self.click[0]=self.click[1]
            self.click[1]=self.click[2]
            self.click[2]=click
            return True,clickList
    
    def initP(self):
        self.pList=dp(self.click)

    def getP(self):
        return dp(self.pList)

    def addP(self,p):
        self.pList[0]=self.pList[1]
        self.pList[1]=self.pList[2]
        self.pList[2]=p
        

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

        self.device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        # train 
        self.agent=REMAgent(device=self.device,rnn_layer=5,embed_n=128)

        #predict
        self.predictModel=REMAgent(device=self.device,rnn_layer=5,embed_n=128)

        # 
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
        timestamp=getTimeStamp()
        self.storePath=os.path.join(OnlineConfig.ONLIEN_DATA_PATH.value,timestamp[0:-6],'model',timestamp[-6:])
        if not os.path.exists(self.storePath):
            os.makedirs(self.storePath)
        self.modelSavePath=os.path.join(self.storePath,'onlineModel.pt')
        self.rewardPath=os.path.join(self.storePath,'reward.txt')   
        self.onlineReward=os.path.join(self.storePath,'onlineReward.txt')    
        self.updateModel=os.path.join(self.storePath,'updateModel.txt')    
        self.update=False

        # self.dataCreater=DataCreater()
        self.stop=False
        self.tras=DQNRNNTrajectory()

        # for dataset
        self.dataSetBuffer=ReplayBufferRNN(2**19,self.device)
        # for test people data
        self.trainBuffer=ReplayBufferRNN(2**17,self.device)
        self.batchSize=8
        self.lr=OnlineConfig.LR.value
        self.agent.load(OnlineConfig.LOAD_PATH.value)
        self.predictModel.load(OnlineConfig.LOAD_PATH.value)
        self.baseStorePath=OnlineConfig.EYE_DATA_PATH.value
        self.dc=DataCreater()
        self.newFileNums=0
        self.remBlocskNum=5

    # multi threading
    def train(self):
        global dataRecorder
        trainer = torch.optim.Adam(lr=self.lr, params=self.agent.online.parameters())
        EPOSILON=0.02
        loss=nn.HuberLoss()
        best_scores=-np.inf
        envs=[]
        t_reward=[]
        trajectorys=[]
        pr=PresentsRecorder(20)
        t_reward_len=-1
        trainNums=[]
        skipSize=5
        for i in range(2,22):
            if i in OnlineConfig.SKIP_LIST.value:
                continue
            if i<10:
                envStr='0'+str(i)
            else:
                envStr=str(i)
            trainNums.append(i)
            envPath=os.path.join(OnlineConfig.DATASET_PATH.value,envStr,OnlineConfig.SCENE.value)
            envs.append(DQNRNNEnv(envPath))
            trajectorys.append(DQNRNNTrajectory())
        if os.path.exists(OnlineConfig.INDIVIDUAL.value):
            fileNum=len(os.listdir(OnlineConfig.INDIVIDUAL.value))
        else:
            fileNum=0
        onlineNum=int(np.ceil(fileNum/5))
        for i in range(onlineNum):
            envs.append(DQNRNNEnv(OnlineConfig.INDIVIDUAL.value))
            trajectorys.append(DQNRNNTrajectory())
            pr.addRecorder()
            trainNums.append(OnlineConfig.PERSON.value)
            # trajectorys.append(DQNTrajectory())
        for i in range(len(envs)):
            envs[i].load_dict()
            envs[i].get_file()
        for env in envs:
            env.state_mode=1
        K=0
        randomLow,randomHigh=0,19
        device=self.device
        while True:
            if self.stop:
                with modelConditioner:
                    print('stop to train')
                    modelConditioner.wait()
                print('awake to train')

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
            for steps in range(200):

                if self.stop:
                    with modelConditioner:
                        print('stop train')
                        modelConditioner.wait()
                if self.newFileNums%5==0 and self.newFileNums!=0:
                    updateTimes=int(np.ceil(self.newFileNums/5))
                    skipSize+=1
                    trainNums.append(OnlineConfig.PERSON.value)
                    for i in range(updateTimes):
                        envs.append(DQNRNNEnv(OnlineConfig.INDIVIDUAL.value))
                        envs[-1].load_dict()
                        envs[-1].get_file()
                        pr.addRecorder()
                        trajectorys.append(DQNRNNTrajectory())
                    
                    self.newFileNums=0

                # print('train2')
                eyeList=[]
                clickList=[]
                indexes=[]
                lastPList=[]
                lengthsList=[]
                personList=[]
                # ans -> eye click goal isEnd
                skipSize=np.minimum(15,skipSize)
                skipNum=np.random.randint(low=0,high=20,size=skipSize)
                for i in range(len(envs)):
                    if OnlineConfig.SKIP.value and i in skipNum:
                        continue
                    ans=envs[i].act()
                    if isinstance(ans,bool):
                        envs[i].load_dict()
                        if i >=OnlineConfig.PERSON_ENVS.value:
                            print(i,steps)
                        envs[i].get_file()
                        continue
                    else:
                        eye=torch.tensor(ans[0],dtype=torch.long)
                        click=torch.tensor(ans[1],dtype=torch.long).to(device)
                        lengths=torch.tensor(ans[4],dtype=torch.long)
                        if pr.flag[i]==False:
                            pr.init(i,ans[1])
                        lastP=torch.tensor(pr.getLastPresents(i),dtype=torch.long).to(device)
                        person=torch.tensor([trainNums[i]],dtype=torch.float32).to(device)
                        lastPList.append(lastP)
                        eyeList.append(eye)
                        clickList.append(click)
                        lengthsList.append(lengths)
                        personList.append(person)
                        # ans -> eye click goal isEnd
                        doneFlag=0
                        if ans[3]==1:
                            doneFlag=0
                        else:
                            doneFlag=1
                        trajectorys[i].push(click,eye,ans[2],0,doneFlag*UTIL.GAMMA,lastP,lengths,person)
                        indexes.append(i)
                clickCat=torch.stack(clickList,dim=0).to(device)
                lastPCat=torch.stack(lastPList,dim=0).to(device)
                lengthStack=torch.stack(lengthsList,dim=0)
                personStack=torch.stack(personList,dim=0).to(device).unsqueeze(1)   
                # for e in eyeList:
                #     print(f'e {e} len {len(e)}')
                eyeList=torch.stack(eyeList,dim=0).to(device)
                actions=self.agent.act(clickCat,eyeList,lastPCat,lengthStack,personStack)
                # print(f' aaaaa {actions}')
                for actS,index in zip(actions,indexes):
                    pr.add(index,actS)
                    prob=random.random()
                    if prob>EPOSILON:
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
                        if traInfo[0]>1 and index>=OnlineConfig.PERSON_ENVS.value:
                            traLenOverOne+=traInfo[0]
                            noActionNum+=traInfo[1]
                            noActionNumWithThreshold+=traInfo[2]
                        if index>=OnlineConfig.PERSON_ENVS.value:
                            lastWithNoAction+=traInfo[3]
                            totalIn+=traInfo[4]
                            totalOut+=traInfo[5]
                            endIn+=traInfo[6]
                            endOut+=int(traInfo[6])^1
                        if index>=OnlineConfig.PERSON_ENVS.value:
                            rewards.append(traInfo[7])
                            endAveReward.append(traInfo[8])
                            inNoAct+=traInfo[9]
                            outNoAct+=traInfo[10]
                        if traInfo[0]>=3 and index>=OnlineConfig.PERSON_ENVS.value:
                            traLenOverThree+=traInfo[0]
                            totalErrors+=traInfo[11]
                        TDZeroList=trajectorys[index].getComTraZero()
                        if traInfo[12]!=-1 and index>=OnlineConfig.PERSON_ENVS.value:
                            endInNoAct+=traInfo[12]
                            endOutNoAct+=int(traInfo[12])^1
                        
                        self.dataSetBuffer.push(TDZeroList)
                        if index>=OnlineConfig.PERSON_ENVS.value:
                            self.trainBuffer.push(TDZeroList)
        
                        trajectorys[index].clear()
                # print(f'end train {self.trainBuffer.holding} {self.dataSetBuffer.holding}')
                
                datasetBatchSize=int((1+(self.dataSetBuffer.getRatio()))*self.batchSize*4)
                if self.dataSetBuffer.holding<datasetBatchSize:
                    continue
                with bufferConditioner:
                    trainBatchSize=int((1+(self.trainBuffer.getRatio()))*self.batchSize)
                if trainBatchSize*2 > self.trainBuffer.capacity and self.batchSize<256:
                    self.batchSize*=2
                clickList,eyeList,lastPList,lengths,person,actionList,rewardList,maskList,nclickList,neyeList,nlastPList,nlengths,nperson = self.dataSetBuffer.sample(datasetBatchSize)
                clickListT,eyeListT,lastPListT,lengthsT,personT,actionListT,rewardListT,maskListT,nclickListT,neyeListT,nlastPListT,nlengthsT,npersonT = self.trainBuffer.sample(trainBatchSize)
                clickList=torch.cat([clickList,clickListT],dim=0)
                eyeList=torch.cat([eyeList,eyeListT],dim=0)
                lastPList=torch.cat([lastPList,lastPListT],dim=0)
                lengths=torch.cat([lengths,lengthsT],dim=0)
                actionList=torch.cat([actionList,actionListT],dim=0)
                rewardList=torch.cat([rewardList,rewardListT],dim=0)
                maskList=torch.cat([maskList,maskListT],dim=0)
                nclickList=torch.cat([nclickList,nclickListT],dim=0)
                neyeList=torch.cat([neyeList,neyeListT],dim=0)
                nlastPList=torch.cat([nlastPList,nlastPListT],dim=0)
                nlengths=torch.cat([nlengths,nlengthsT],dim=0)
                person=torch.cat([person,personT],dim=0)
                nperson=torch.cat([nperson,npersonT],dim=0)
                
                
        
                deltaP = torch.rand(self.remBlocskNum).to(device)
                deltaP =deltaP/ deltaP.sum()
                with torch.no_grad():
                    onlineValues=self.agent.online(clickList,eyeList,lastPList,lengths,person)
                    onlineValues=sum(onlineValues)/len(onlineValues)
                    yAction=torch.argmax(onlineValues,dim=-1,keepdim=True)
                    targetValues=self.agent.target(nclickList,neyeList,nlastPList,nlengths,nperson)
                    for i in range(self.remBlocskNum):
                        targetValues[i]=targetValues[i]*deltaP[i]
                    targetValues=sum(targetValues)
                    y=targetValues.gather(dim=-1,index=yAction)*maskList+rewardList
                values=self.agent.online(clickList,eyeList,lastPList,lengths,person)
                for i in range(self.remBlocskNum):
                    values[i]=values[i]*deltaP[i]
                values=sum(values)
                values=values.gather(dim=-1,index=actionList)
                l = loss(values, y)
                trainer.zero_grad()
                l.backward()
                trainer.step()
                if steps%10==0 and steps!=0:
                    self.agent.update()

            t_reward_len+=1
            if t_reward_len>=10:
                t_reward_len%=10
                t_reward[t_reward_len]=np.mean(rewards)
            else:
                t_reward.append(np.mean(rewards))

            if len(t_reward)>4 and np.mean(t_reward)>best_scores:
                print('save')
                with modelConditioner:
                    torch.save(self.agent.target.state_dict(), self.modelSavePath)
                    self.update=True
                best_scores=np.mean(t_reward)
                with open(self.updateModel,'a',encoding='UTF-8') as updateModel:
                    updateModel.write('eposides:'+str(K+1)+' ave_eposides_rewards:'+str(round(np.mean(t_reward),2))+\
                    ' ave reward '+str(round(t_reward[-1],2)) +'\n'+\
                    ' end_in: '+str(endIn)+' end_out: '+str(endOut)+' IPA2: '+str(round(endIn/(endOut+endIn),2))+\
                    ' end_in_no_action: '+str(endInNoAct)+' end_out_no_act: '+str(endOutNoAct)+' IPA2_NOACTION:'+str(round(endInNoAct/(endInNoAct+endOutNoAct),2))+'\n'+\
                    ' in: '+str(totalIn)+' _out:'+str(totalOut)+' IPA1:'+str(round(totalIn/(totalIn+totalOut),2))+'\n')
            #     with open(updataInfo,'a',encoding='UTF-8') as updateInfoFile:
            #         updateInfoFile.write('eposides:'+str(K+1)+' ave_eposides_rewards:'+str(round(np.mean(t_reward),2))+\
            #         ' ave reward '+str(round(t_reward[-1],2)) +'\n'+\
            #         ' end_in: '+str(endIn)+' end_out: '+str(endOut)+' train_end_acc: '+str(round(endIn/(endOut+endIn),2))+\
            #         ' end_in_no_action: '+str(endInNoAct)+' end_out_no_act: '+str(endOutNoAct)+' acc:'+str(round(endInNoAct/(endInNoAct+endOutNoAct),2))+'\n'+\
            #         ' in: '+str(totalIn)+' _out:'+str(totalOut)+' train_ave_acc:'+str(round(totalIn/(totalIn+totalOut),2))+\
            #         ' in_no_act:'+str(inNoAct)+' out_no_act:'+str(outNoAct)+' acc:'+str(round(inNoAct/(inNoAct+outNoAct),2))+'\n'+\
            #         ' len_tra_over_one: '+str(traLenOverOne)+' no_action_num:'+str(noActionNum)+' no_action_num_80:'+str(noActionNumWithThreshold)+\
            #         ' acc:'+str(round(noActionNum/traLenOverOne,2))+' acc2:'+str(round(noActionNumWithThreshold/traLenOverOne,2))+'\n'+\
            #         ' len_tra_over_three: '+str(traLenOverThree)+' total_errors: '+str(totalErrors)+' acc: '+str(round(totalErrors/traLenOverThree,2))+'\n')
            
            K+=1
            with open(self.rewardPath,'a',encoding='UTF-8') as rewardFile:
                rewardFile.write('eposides:'+str(K+1)+' ave_eposides_rewards:'+str(round(np.mean(t_reward),2))+\
                ' ave reward '+str(round(t_reward[-1],2)) +'\n'+\
                ' end_in: '+str(endIn)+' end_out: '+str(endOut)+' IPA2: '+str(round(endIn/(endOut+endIn),2))+\
                ' end_in_no_action: '+str(endInNoAct)+' end_out_no_act: '+str(endOutNoAct)+' IPA2_NOACTION:'+str(round(endInNoAct/(endInNoAct+endOutNoAct),2))+'\n'+\
                ' in: '+str(totalIn)+' _out:'+str(totalOut)+' IPA1:'+str(round(totalIn/(totalIn+totalOut),2))+'\n')
    

    def getAveTraAcc(self):
        aveAcc='file_IPA1: '+str(round(np.sum(self.traAcc)/np.sum(self.traLen),2))+' file_IPA2: '+str(round(np.mean(self.traEndAcc),2))+' file_PRR: '+str(round(np.mean(self.traPRR),2))
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
        with open(self.onlineReward,'a') as f:
            f.write('tras: '+str(self.predictCnts)+' tra_len: '+str(info[0])+' '+\
                    'total_correct: '+str(info[1]) +' end_correct: '+str(info[2])+'\n'+\
                    'total_reward: '+str(info[3])+' end_reward: '+str(info[4])+' ave_reward: '+str(round(info[4]/info[0],2))+'\n'+\
                    'IPA2: '+str(round(np.mean(self.onlineAcc),2))+' TOTALIPA1: '+str(round(float(np.sum(self.onlineAveAcc)/np.sum(self.onlineLen)),2))+\
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
        clickList,eyeList,length,lastP,clickRegion,state,person=data
        mask=UTIL.GAMMA
        if self.update:
            with modelConditioner:
                print('used the update')
                self.predictModel.load(self.modelSavePath)
                self.update=False

        clickTensor=torch.tensor([clickList],dtype=torch.long).unsqueeze(0).to(self.device)
        # print(f'eyeList {eyeList}')
        eyeTensor=torch.tensor([eyeList],dtype=torch.long).unsqueeze(0).to(self.device)
        lenTensor=torch.tensor([length],dtype=torch.long)
        pTensor=torch.tensor([lastP],dtype=torch.long).unsqueeze(0).to(self.device)
        personTensor=torch.tensor([[[person]]],dtype=torch.float32).to(self.device)
        ans=int(self.predictModel.act(clickTensor,eyeTensor,pTensor,lenTensor,personTensor))
        # click,eye,goal,action,mask
        mask=UTIL.GAMMA
        if state==0:
            mask=0
        self.tras.push(clickTensor.squeeze(),eyeTensor.squeeze(),0,ans,mask,pTensor.squeeze(),lenTensor.squeeze(),personTensor.squeeze(0).squeeze(0))
        if state==0:
            print(f'add')
            self.tras.setGoal(clickRegion)
            self.numberRecorder=0
            self.tras.getNewTras()
            TDZeroList=self.tras.getComTraZero()
            with bufferConditioner:
                self.trainBuffer.push(TDZeroList)
            self.writeReward()
            self.tras.clear()
            self.dc.refreshTra()
            mask=0
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


@app.route('/sendFile',methods=['post','get'])
def saveFile():
    print('send')
    global model
    data=request.files.get('file')
    fileName=request.files.get('fileName')
    data.save(os.path.join(OnlineConfig.EYE_DATA_PATH.value,fileName))
    model.dc.refresh()
    writeTxt=model.getAveTraAcc()
    model.writeTxt(writeTxt)
    print(f'this is txt {writeTxt}')
    model.traAccClear()
    model.writeTxt('\nend\n\n')
    responseDict=getResponse()
    responseJson=json.dumps(responseDict)
    response=make_response(responseJson)
    model.newFileNums+=1
    return response


@app.route('/sendData',methods=['post','get'])
def sendData():
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
    savePath=os.path.join(OnlineConfig.EYE_DATA_PATH.value,peopleNum,sceneNum)
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
        if model.trainBuffer.holding<model.batchSize:
            responseDict['data']=False
            responseDict['flag']=0
            responseDict['message']='short of data'+' train: '+str(model.trainBuffer.holding)
        else:
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
    responseJson=json.dumps(responseDict)
    response=make_response(responseJson)
    return response




@app.route('/getPredict',methods=['post','get'])
def getPredict():
    global model,STOP_TRAIN_FLAG
    resp=request.get_json()
    responseDict=getResponse()
    clickRegion,eyeRegion,mouseS,person=resp.get('clickRegion'),resp.get('eyeRegion'),resp.get('mouseS'),resp.get('person')
    eyeAns=model.dc.addEye(eyeRegion)
    clickAns=model.dc.addClick(clickRegion)
    if clickAns[0]==False:
        responseDict['data']=clickAns[1]
        responseDict['flag']=False
        responseDict['message']='click not enough'
    else:
        # clickList,eyeList,length,lastP,clickRegion,statefg
        lastP=model.dc.getP()
        ans=model.predict(clickAns[1],eyeAns[0],eyeAns[1],lastP,clickRegion,mouseS,person)
        responseDict['data']=ans[0]
        responseDict['flag']=True
        responseDict['message']=ans[1]
    responseJson=json.dumps(responseDict)
    response=make_response(responseJson)
    return response


# 启动程序
if __name__ == '__main__':
    app.run( host='0.0.0.0', port=7788)
    

