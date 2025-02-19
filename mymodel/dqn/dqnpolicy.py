import torch
from torch import nn
import os
import sys
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import Dataset
import numpy as np

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
from dqndataloader import *

def testFun(agent):
    testAllEnvs=[]
    testAllNum=[]
    testAllScene=[]
    testEnvs=['06', '11', '07', '04', '13']
    indexDict={}
    k=0
    for _ in testEnvs:
        indexDict[int(_)] =k
        k+=1
    for scene in range(1,5):
        for env in testEnvs:
            testEnvPath=os.path.join('/home/wu_tian_ci/eyedata/seperate',env,str(scene))
            testAllEnvs.append(DQNRNNEnv(testEnvPath,num=int(env),scene=int(scene),restrict=True))
            testAllNum.append(int(env))
            testAllEnvs[-1].shuffle=False
            testAllEnvs[-1].topN=0
            testAllEnvs[-1].eval=True

            testAllScene.append(scene)

    preTest=PresentsRecorder(len(testAllEnvs))
    stopFlags=[False for i in range(len(testAllEnvs))]
    surviveFlags=len(testAllEnvs)

    numFlag=surviveFlags//4
    # numFlag=surviveFlags
    testTras=[DQNRNNTrajectory2() for i in range(surviveFlags)]
    testEndInS=[[0 for i in range(5)] for i in range(numFlag)]
    testEndOutS=[[0 for i in range(5)] for i in range(numFlag)]
    testInS=[[0 for i in range(5)] for i in range(numFlag)]
    testOutS=[[0 for i in range(5)] for i in range(numFlag)]
    testErrorsS=[[0 for i in range(5)] for i in range(numFlag)]
    testLenS=[[0 for i in range(5)] for i in range(numFlag)]
    agent.eval()


    rewards=[]
    for i in range(len(testAllEnvs)):
        testAllEnvs[i].load_dict()
        testAllEnvs[i].get_file()
    # K=0
    while surviveFlags>0:
        # print(K)
        # K+=1
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
        # click,eye,lengths,person,scene
        testActions=agent.act(testClickStack,testEyeListStack,testLengthStack,testPersonListStack,testSceneStack)
        if testActions.shape==():
            testActions=[testActions]
        for testAction,testIndex in zip(testActions,testIndexesList):
            testTras[testIndex].tras[-1][3]=dp(testAction)
            if testTras[testIndex].tras[-1][4]==0:
                testTras[testIndex].getNewTras()
                traInfoTest=testTras[testIndex].getInfo()
                rewards.append(traInfoTest[7])

                testInS[indexDict[testAllNum[testIndex]]][testAllScene[testIndex]]+=\
                traInfoTest[4]

                testOutS[indexDict[testAllNum[testIndex]]][testAllScene[testIndex]]+=traInfoTest[5]
                testEndInS[indexDict[testAllNum[testIndex]]][testAllScene[testIndex]]+=traInfoTest[6]
                testEndOutS[indexDict[testAllNum[testIndex]]][testAllScene[testIndex]]+=int(traInfoTest[6])^1
                if traInfoTest[0]>=3:
                    testLenS[indexDict[testAllNum[testIndex]]][testAllScene[testIndex]]+=traInfoTest[0]
                    testErrorsS[indexDict[testAllNum[testIndex]]][testAllScene[testIndex]]+=traInfoTest[11]
                testTras[testIndex].clear()
    return testEndInS,testEndOutS,testInS,testOutS,testErrorsS,testLenS,np.mean(rewards)

def testFunPolicy(agent,testIter,device):
    agent.eval()
    loss = torch.nn.CrossEntropyLoss()
    lossTotal=[]
    # net.eval()

    for clickTensor,eyeTensor,goalTensor,lenTensor,personTensor,sceneTensor in testIter:
        clickTensor=clickTensor.to(device)
        eyeTensor=eyeTensor.to(device)
        goalTensor=goalTensor.to(device).squeeze()
        personTensor=personTensor.to(device)
        sceneTensor=sceneTensor.to(device)
        ans=agent.online(clickTensor,eyeTensor,lenTensor,personTensor,sceneTensor).squeeze()
        l=loss(ans,goalTensor)
        lossTotal.append(l.cpu().detach().numpy())
    return np.mean(lossTotal)

# agent, train_iter,args.lr, args.epochs,device,store_path=store_path
def train_epoch(agent,train_iter,testIter, lr, epochs, device,store_path=None):
    modelPath=os.path.join(store_path,'model.pt')
    testEnvs=['06', '11', '07', '04', '13']
    preTestInfos=[]
    for t in testEnvs:
        preTestInfo=os.path.join(store_path,t+'.txt')
        preTestInfos.append(preTestInfo)
        f=open(preTestInfo,'w')
        f.write('begin\n')
        f.close()
    trainer = torch.optim.Adam(lr=lr, params=agent.online.parameters())
    loss = torch.nn.CrossEntropyLoss()

    bestReward=np.inf
    for _ in tqdm(range(epochs)):
        agent.train()
        for clickTensor,eyeTensor,goalTensor,lenTensor,personTensor,sceneTensor in train_iter:
            clickTensor=clickTensor.to(device)
            eyeTensor=eyeTensor.to(device)
            goalTensor=goalTensor.to(device).squeeze()
            personTensor=personTensor.to(device)
            sceneTensor=sceneTensor.to(device)
            trainer.zero_grad()
            ans=agent.online(clickTensor,eyeTensor,lenTensor,personTensor,sceneTensor).squeeze()
            l=loss(ans,goalTensor)
            l.backward()
            trainer.step()
        testLoss=testFunPolicy(agent,testIter,device)
        if testLoss<bestReward:
            torch.save(agent.online.state_dict(), modelPath)
            bestReward=testLoss
            testEndInS,testEndOutS,testInS,testOutS,testErrorsS,testLenS,testRewards=testFun(agent)
            for j in range(5):
                testErrorR='\nerror: '
                testAccR='\nIPA1: '
                testEndAccR='\nIPA2: ' 
                for i in range(5):
                    testEndAccR+=str(i)+' '+str(round(testEndInS[j][i]/(testEndInS[j][i]+testEndOutS[j][i]) if (testEndInS[j][i]+testEndOutS[j][i])>0  else 0 ,2))+' '
                    testAccR+=str(i)+' '+str(round(testInS[j][i]/(testInS[j][i]+testOutS[j][i]) if (testInS[j][i]+testOutS[j][i])>0  else 0 ,2))+' '
                    testErrorR+=str(i)+' '+str(round(testErrorsS[j][i]/testLenS[j][i] if testLenS[j][i]>0  else 0 ,2))+' '
                with open(preTestInfos[j],'a',encoding='UTF-8') as f:
                    f.write('\npretest trainenv: '+'\n+env: '+testEnvs[i]+'  '+str(_)+'\n'+testErrorR+testAccR+testEndAccR+'\n')
           


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
    parser.add_argument('-cuda',type=str,default='cuda:0')
    parser.add_argument('-mode',type=int,default=1)
    parser.add_argument('-net',type=int,default=1)
    parser.add_argument('-lr',type=float,default= 0.0005)
    # 0.00005 0.0005
    parser.add_argument('-layers',type=int,default=5)
    parser.add_argument('-embed',type=int,default=128)
    parser.add_argument('-rems',type=int,default=5)
    parser.add_argument('-epochs',type=int,default=30)
    parser.add_argument('-batchsize',type=int,default=256)
    parser.add_argument('-restrict',type=str2bool,default=True)
    parser.add_argument('-path',type=str2bool,default=False)
    parser.add_argument('-train',type=int,default=1)
    parser.add_argument('-load',type=str2bool,default=False)
    parser.add_argument('-flag',type=int,default=1)
    args=parser.parse_args()
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    agent=PolicyAgent(device=device,rnn_layer=args.layers,embed_n=args.embed,flag=args.flag)
    store=UTIL.getTimeStamp()

    mainPath=os.path.join('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/',store[0:-6],'policy')
    if not os.path.exists(mainPath):
        os.makedirs(mainPath)
    fileNum=len(os.listdir(mainPath))
    store_path=os.path.join(mainPath,str(fileNum))
    print(store[-6:])
    envs=[]
    exclude=[]
    # envTrain=[12, 14, 18, 21, 8, 2, 5, 20, 19, 9, 16, 10, 3, 15, 17]
    envTrain=[6, 11, 7, 4, 13]
    for i in envTrain:
        if i<10:
            envs.append('0'+str(i))
        else:
            envs.append(str(i))
    print(device)
    # envs=['23']
    if args.path==True:
        if args.flag==1:
            store_path='/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20240110/policystorepath'
        elif args.flag==2:
            store_path='/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20240110/policystorepath2'
    if args.load:
        if args.flag==1:
            agent.load('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20240110/policystorepath/model.pt')
        elif args.flag==2:
            agent.load('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20240110/policystorepath2/model.pt')
    testPath='/home/wu_tian_ci/eyedata/dqn_policy/test.json'
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    if args.train!=1:
        testPath='/home/wu_tian_ci/eyedata/dqn_policy/test.json'
        trainPath=os.path.join('/home/wu_tian_ci/eyedata/dqn_policy/','train'+str(args.train)+'.json')
    else:
        testPath='/home/wu_tian_ci/eyedata/dqn_policy/test2.json'
        trainPath='/home/wu_tian_ci/eyedata/dqn_policy/train.json'
    with open(os.path.join(store_path,'envsinfo.txt'),'w') as f:
        f.write('TRAIN:' +trainPath+'\n'+'envs'+str(envs)+'\n trainEnvs:'+str(envs)+'\n'+' lr:'+str(args.lr)+' device:'+str(device)+'\n'+\
                str(args.path)+' '+str(args.load)+'\n'+store_path+'\n'+'flag: '+str(args.flag)+'\n')
    train_dataset=DQNDataLoader(trainPath)
    train_iter=torch.utils.data.DataLoader(dataset=train_dataset,shuffle=True,batch_size=256,num_workers=16,pin_memory=True)
    testDataset=DQNDataLoader(testPath)
    testIter=torch.utils.data.DataLoader(dataset=testDataset,shuffle=True,batch_size=256,num_workers=16,pin_memory=True)
    train_epoch(agent, train_iter,testIter,args.lr, args.epochs,device,store_path=store_path)