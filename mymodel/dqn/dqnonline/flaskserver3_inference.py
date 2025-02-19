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
import traceback
# warnings.filterwarnings("error") 

torch.autograd.set_detect_anomaly(True)
# to stop training
modelConditioner=threading.Condition(threading.Lock())
# for train
bufferConditioner=threading.Condition(threading.Lock())

addConditioner=threading.Condition(threading.Lock())
STOP_TRAIN_FLAG=False

# response sample 
# class Responser(object):
#     def __init__(self):
#         # type 0-> wanted prediect 1-> train data
#         self.type=0
#         self.data={}


REWARDS=20

# create data

# class DataCreater(object):
#     def __init__(self) -> None:
#         self.eye=[]
#         self.click=[]
#         self.pList=[]

#     def refreshTra(self):
#         self.eye=[]
#         # self.click=[]
#         # self.pList=[]
    
#     def refresh(self):
#         self.eye=[]
#         self.click=[]
#         self.pList=[]

#     # region_level data
#     # false -> not predict

#     def getEye(self):
#         eyeList=[]
#         if len(self.eye)<10:
#             eyeList=dp(self.eye)
#         else:
#             step=(len(self.eye)//10)
#             for i in range(9):
#                 eyeList.append(self.eye[i*step])
#             eyeList.append(self.eye[-1])
#         length=len(eyeList)
#         if len(eyeList)<10:
#             eyeList.extend([0 for i in range(10-length)])
#         return eyeList,length

#     def addEye(self,eye):
#         self.eye.append(eye)
#         return self.getEye()
    
#     def addP(self,p):
#         self.pList.append(p)
#         return dp(self.pList[-3:])

#     def addClick(self,click,mouseS):
#         clickFlag= True if len(self.click)>=3 else False
#         if clickFlag:
#             clickAns=dp(self.click[-3:])
#         else:
#             clickAns=len(self.click)
#         if mouseS==0:
#             if len(self.click)==0 or self.click[-1]!=click:
#                 self.click.append(click)
        
#         if len(self.click)>=3:
#             return 
#         return False,len(self.click)
    
#     def initP(self):
#         self.pList=dp(self.click)

        

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

class TrainModel(object):
    def __init__(self) -> None:
        

        # 1 iteration mode 2->no iteration mode 0->offline
        self.mode=0
        self.switchModeFlag=False

        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
        self.storePath=os.path.join(OnlineConfig.ONLIEN_DATA_PATH.value,self.timeStamp[0:-6],'model',self.timeStamp[-6:])
        if not os.path.exists(self.storePath):
            os.makedirs(self.storePath)
        self.modelSavePath=os.path.join(self.storePath,'onlineModel.pt')
        self.rewardPath=os.path.join(self.storePath,'reward.txt')   
        self.onlineReward=os.path.join(self.storePath,'onlineReward.txt')    
        self.updateModel=os.path.join(self.storePath,'updateModel.txt')    
        self.update=False
        self.checkpointBase=os.path.join(OnlineConfig.ONLIEN_DATA_PATH.value,self.timeStamp[0:-6],'checkpoint')

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
        # self.dc=DataCreater()
        self.newFileNums=0
        self.remBlocskNum=5
        self.best_scores=-np.inf
        # personID
        self.currentID=-1
        self.sceneID=-1
        self.addNum=[]
        self.addScene=[]
        self.addFlag=[]
        self.filesPath=[]


        self.error=False

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
        # self.isTraining=False

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
        self.storePath=os.path.join(OnlineConfig.ONLIEN_DATA_PATH.value,self.timeStamp[0:-6],'model',self.timeStamp[-6:])
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
        # self.dc=DataCreater()
        self.newFileNums=0
        self.remBlocskNum=5
        self.switchModeFlag=False     

    # multi threading

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
        eyeList,clickList,lastP,state,person,scene,length,goal,timeStamp=data
        self.currentID=person
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
        prob=self.predictModel.getProbs()
        # click,eye,goal,action,mask
        mask=UTIL.GAMMA
        if state==0:
            mask=0
        self.tras.push(clickTensor.squeeze(),eyeTensor.squeeze(),0,ans,mask,pTensor.squeeze(),\
                       lenTensor,personTensor.squeeze(0).squeeze(0),sceneTensor.squeeze(0).squeeze(0))
        self.tras.pushProbs(self.predictModel.getProbs())
        self.tras.pushTimeStamp(timeStamp)
        if state==0 and len(self.tras.tras)>0:
            print(f'add')
            self.tras.setGoal(goal)
            self.numberRecorder=0
            self.tras.getNewTras()

            
            self.tras.clear()

        return ans,prob,'success'

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

    model.traAccClear()

    responseDict=getResponse()
    responseJson=json.dumps(responseDict)
    response=make_response(responseJson)    

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
            # model.mode=mode
            if model.mode!=mode:
                # if model.mode!=0:
                #     model.switchModeFlag=False
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
    timeStamp=resp.get('timeStamp')
    print(f'scene {sceneId}')
    model.sceneID=sceneId
    try:
        ans=model.predict(eyeList,clickList,pList,mouseS,person,sceneId,length,goal,timeStamp)
    except:
        print('error4')
        print(eyeList,clickList,pList,mouseS,person,sceneId,length,goal,timeStamp)
        ans=12
    print(ans[0])
    responseDict['data']=ans[0]
    responseDict['flag']=True
    responseDict['message']=ans[2]
    responseDict['prob']=ans[1].tolist()
    responseJson=json.dumps(responseDict)
    response=make_response(responseJson)
    return response


# 启动程序
if __name__ == '__main__':
    app.run( host='0.0.0.0', port=7788)
    

