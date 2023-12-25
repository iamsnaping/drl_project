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

def testFun(agent,testAllEnvs,testAllNum,testAllScene):

    preTest=PresentsRecorder(len(testAllEnvs))
    stopFlags=[False for i in range(len(testAllEnvs))]
    surviveFlags=len(testAllEnvs)

    numFlag=surviveFlags//3
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
        


def train_epoch(agent:REMAgent2, device,testEnvs,topN=5,store_path=None,load=False,loadPath='',restrict=True):
    agent.online.to(device)
    agent.target.to(device)
    # ebuffer=ExampleBuffer(2**17)
 
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    preTestInfos=[]
    env_str=''
    for t in testEnvs:
        preTestInfo=os.path.join(store_path,t+'preAfterTest.txt')
        preTestInfos.append(preTestInfo)
        f=open(preTestInfo,'w')
        f.write(env_str+' '+t+'\n')
        f.close()

    testAllEnvs=[]
    testAllNum=[]
    testAllScene=[]

    for scene in range(1,5):
        if scene==3:
            continue

        for env in testEnvs:
            # testEnvPath=os.path.join('/home/wu_tian_ci/eyedata/seperate',env,str(scene))
            testEnvPath=os.path.join('/home/wu_tian_ci/eyedatanew',env,str(scene))
            print(testEnvPath)
            testAllEnvs.append(DQNRNNEnv(testEnvPath,num=int(env),scene=int(scene),restrict=restrict,MODE=0))
            testAllNum.append(int(env))
            testAllEnvs[-1].shuffle=False
            testAllEnvs[-1].topN=0
            testAllEnvs[-1].eval=True
            testAllScene.append(scene)


    for tt in testEnvs:
       # pre-test

        testEndInS,testEndOutS,testInS,testOutS,testErrorsS,testLenS=testFun(agent,testAllEnvs,testAllNum,testAllScene)
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
    parser.add_argument('-cuda',type=str,default='cuda:1')
    parser.add_argument('-sup',type=str,default='50')
    parser.add_argument('-preload',type=str2bool,default=True)
    parser.add_argument('-topN',type=int,default=5)
    parser.add_argument('-layers',type=int,default=5)
    parser.add_argument('-embed',type=int,default=128)
    parser.add_argument('-rems',type=int,default=5)
    # iteration flag true->iteration donot use before False->iteration use before
    parser.add_argument('-load',type=str2bool,default=False)
    parser.add_argument('-idFlag',type=str2bool,default=True)
    parser.add_argument('-restrict',type=str2bool,default=False)
    # TRUE -> skip the scene three

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
            actor_load='/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20231222/offline/5/dqnnetoffline.pt'
        agent.load(actor_load)
    # else:
    #     actor_load='/home/wu_tian_ci/drl_project/mymodel/ddpg/pretrain_data/ddpg_train/1last_move5/ActorNet.pt'
    mainPath=os.path.join('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/',store[0:-6],'test')
    if not os.path.exists(mainPath):
        os.makedirs(mainPath)
    fileNum=len(os.listdir(mainPath))
    store_path=os.path.join(mainPath,str(fileNum))


    print(args.sup)

    envs=[]
    for i in range(17,22):
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
    # testEnvs=['23']
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    with open(os.path.join(store_path,'envsinfo.txt'),'w') as f:
        f.write('testEnv: '+str(testEnvs)+' \n'+'time: '+store[0:-6]+' '+store[-6:]+'\n'+' actor_load: '+actor_load+'\n'+' restrict: '+str(args.restrict)\
                + 'idFlag: '+str(args.idFlag))
        if args.sup!='50':
            f.write('\n'+args.sup)
    loadBasePath='/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20231221/offline'
    restrict=[True,False,False,True]
    testEnvs=['23']
    print(testEnvs)
    for i in range(4):
        for env in testEnvs:
            store_path_=os.path.join(store_path,str(i),env)
            if restrict[i]==True:
                loadPath='/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20231221/trainallscene/restrict/dqnnetoffline.pt'
            else:
                loadPath='/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20231221/trainallscene/restrict2/dqnnetoffline.pt'
            if not os.path.exists(store_path_):
                os.makedirs(store_path_)
            agent.load(loadPath)
            train_epoch(agent,device,store_path=store_path_,testEnvs=[env],topN=args.topN,load=args.load,loadPath=actor_load,\
                        restrict=restrict[i])