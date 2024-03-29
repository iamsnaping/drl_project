import torch
from torch import nn
import os
import sys
from tqdm import tqdm
from torch.nn import functional as F
sys.path.append('/home/wu_tian_ci/drl_project')
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
            testAllEnvs[-1].topN=5
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
        testActions=agent.act(testClickStack,testEyeListStack,testLastPStack,testLengthStack,testPersonListStack,testSceneStack)
        if testActions.shape==():
            testActions=[testActions]
        for testAction,testIndex in zip(testActions,testIndexesList):
            testTras[testIndex].tras[-1][3]=dp(testAction)
            if testTras[testIndex].tras[-1][4]==0:
                testTras[testIndex].getNewTras()
                traInfoTest=testTras[testIndex].getInfo()
                rewards.append(traInfoTest[7])
                testInS[indexDict[testAllNum[testIndex]]][testAllScene[testIndex]]+=traInfoTest[4]
                testOutS[indexDict[testAllNum[testIndex]]][testAllScene[testIndex]]+=traInfoTest[5]
                testEndInS[indexDict[testAllNum[testIndex]]][testAllScene[testIndex]]+=traInfoTest[6]
                testEndOutS[indexDict[testAllNum[testIndex]]][testAllScene[testIndex]]+=int(traInfoTest[6])^1
                if traInfoTest[0]>=3:
                    testLenS[indexDict[testAllNum[testIndex]]][testAllScene[testIndex]]+=traInfoTest[0]
                    testErrorsS[indexDict[testAllNum[testIndex]]][testAllScene[testIndex]]+=traInfoTest[11]
                testTras[testIndex].clear()
    return testEndInS,testEndOutS,testInS,testOutS,testErrorsS,testLenS,np.mean(rewards)





def train_epoch(agent, lr, epochs, batch_size,device,mode,multienvs,remBlocskNum=5,store_path=None,restrict=False):
    modelPath=os.path.join(store_path,'model.pt')
    testEnvs=['06', '11', '07', '04', '13']
    preTestInfos=[]
    for t in testEnvs:
        preTestInfo=os.path.join(store_path,t+'.txt')
        preTestInfos.append(preTestInfo)
        f=open(preTestInfo,'w')
        f.write('begin\n')
        f.close()
    agent.online.to(device)
    agent.target.to(device)
    EPOSILON_DECAY=epochs
    EPOSILON_START=0.1
    EPOSILON_END=0.02
    # ebuffer=ExampleBuffer(2**17)
    trainer = torch.optim.Adam(lr=lr, params=agent.online.parameters())
    loss=nn.HuberLoss()
    agentBuffer=ReplayBufferRNN2(2**19,device)
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    t_reward=[]
    t_end_reward=[]
    t_reward_len=-1
    best_scores=-np.inf
    is_training=False
    envs=[]
    envFlags=[]
    envPaths=[]
    trajectorys=[]
    personNum=[]
    sceneNum=[]
    for scene in range(1,5):
        for env in multienvs:
            envPath=os.path.join('/home/wu_tian_ci/eyedata/seperate/',env,str(scene))
            envs.append(DQNRNNEnv(envPath,restrict=restrict))
            envs[-1].shuffle=False
            envs[-1].eval=False
            envs[-1].topN=5
            envFlags.append(False)
            envPaths.append(envPath)
            trajectorys.append(DQNRNNTrajectory2())
            personNum.append(int(env))
            sceneNum.append(int(scene))
    for i in range(len(envs)):
        envs[i].load_dict()
        envs[i].get_file()

    pr=PresentsRecorder(len(envs))
    beginFlags=[True for i in range(len(envs))]
    for K in tqdm(range(epochs)):

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
        # 100
        processReward=[]

        # scene
        endInS=[0 for i in range(5)]
        endOutS=[0 for i in range(5)]
        inS=[0 for i in range(5)]
        outS=[0 for i in range(5)]
        errorsS=[0 for i in range(5)]
        lenS=[0 for i in range(5)]
        agent.train()
        for steps in range(200):
            eyeList=[]
            clickList=[]
            indexes=[]
            lastPList=[]
            lengthsList=[]
            personList=[]
            sceneList=[]
            # ans -> eye click goal isEnd length
            for i in range(len(envs)):
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
                    person=torch.tensor([personNum[i]],dtype=torch.long).to(device)
                    scene=torch.tensor([sceneNum[i]],dtype=torch.long).to(device)
                    if pr.flag[i]==False:
                        pr.init(i,ans[1])
                    lastP=torch.tensor(pr.getLastPresents(i),dtype=torch.long).to(device)
                    lastPList.append(lastP)
                    eyeList.append(eye)
                    clickList.append(click)
                    lengthsList.append(lengths)
                    personList.append(person)
                    sceneList.append(scene)
                    doneFlag=0
                    if ans[3]==1:
                        doneFlag=0
                    else:
                        doneFlag=1
                    trajectorys[i].push(click,eye,ans[2],0,doneFlag*UTIL.GAMMA,lastP,lengths,person,scene)
                    indexes.append(i)
                    if ans[3]==1:
                        beginFlags[i]=True
            clickCat=torch.stack(clickList,dim=0).to(device)
            lastPCat=torch.stack(lastPList,dim=0).to(device)
            lengthStack=torch.stack(lengthsList,dim=0)
            personStack=torch.stack(personList,dim=0).to(device).unsqueeze(1)            
            sceneStack=torch.stack(sceneList,dim=0).to(device).unsqueeze(1)       
            eyeList=torch.stack(eyeList,dim=0).to(device)
            actions=agent.act(clickCat,eyeList,lastPCat,lengthStack,personStack,sceneStack)
            # print(f' aaaaa {actions}')
            for actS,index in zip(actions,indexes):
                pr.add(index,actS)
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
                    inS[sceneNum[index]]+=traInfo[4]
                    outS[sceneNum[index]]+=traInfo[5]

                    endIn+=traInfo[6]
                    endOut+=int(traInfo[6])^1
                    endInS[sceneNum[index]]+=traInfo[6]
                    endOutS[sceneNum[index]]+=int(traInfo[6])^1

                    rewards.append(traInfo[7])
                    endAveReward.append(traInfo[8])
                    processReward.append(traInfo[7]-traInfo[8])
                    inNoAct+=traInfo[9]
                    outNoAct+=traInfo[10]
                    if traInfo[0]>=3:
                        traLenOverThree+=traInfo[0]
                        totalErrors+=traInfo[11]
                        lenS[sceneNum[index]]+=traInfo[0]
                        errorsS[sceneNum[index]]+=traInfo[11]

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
                    is_training=True
                clickList,eyeList,lastPList,lengths,person,scene,actionList,rewardList,maskList,\
                    nclickList,neyeList,nlastPList,nlengths,nperson,nscene= agentBuffer.sample(dbatch_size)
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
                # clp.add(l.cpu().detach().numpy().tolist())
                if  steps%5==0 and steps!=0:
                    agent.update()

        # eye click goal isEnd
        t_reward_len+=1
        if t_reward_len>=20:
            t_reward_len%=20
            t_reward[t_reward_len]=np.mean(rewards)
            t_end_reward[t_reward_len]=np.mean(endAveReward)
        else:
            t_reward.append(np.mean(rewards))
            t_end_reward.append(np.mean(endAveReward))


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
        # errorR=[0 for i in range(5)]
        # accR=[0 for i in range(5)]
        # endAccR=[0 for i in range(5)]

        errorR='\nerror: '
        accR='\nIPA1: '
        endAccR='\nIPA2: '

        for i in range(1,5):
            endAccR+=str(i)+' '+str(round(endInS[i]/(endInS[i]+endOutS[i]) if (endInS[i]+endOutS[i])>0  else 0 ,2))+' '
            accR+=str(i)+' '+str(round(inS[i]/(inS[i]+outS[i]) if (inS[i]+outS[i])>0  else 0 ,2))+' '
            errorR+=str(i)+' '+str(round(errorsS[i]/lenS[i] if lenS[i]>0  else 0 ,2))+' '
    

        testEndInS,testEndOutS,testInS,testOutS,testErrorsS,testLenS,testRewards=testFun(agent)
        
        if testRewards>best_scores and is_training:
            torch.save(agent.target.state_dict(), modelPath)
            best_scores=testRewards

            for j in range(5):
                testErrorR='\nerror: '
                testAccR='\nIPA1: '
                testEndAccR='\nIPA2: ' 
                for i in range(5):
                    testEndAccR+=str(i)+' '+str(round(testEndInS[j][i]/(testEndInS[j][i]+testEndOutS[j][i]) if (testEndInS[j][i]+testEndOutS[j][i])>0  else 0 ,2))+' '
                    testAccR+=str(i)+' '+str(round(testInS[j][i]/(testInS[j][i]+testOutS[j][i]) if (testInS[j][i]+testOutS[j][i])>0  else 0 ,2))+' '
                    testErrorR+=str(i)+' '+str(round(testErrorsS[j][i]/testLenS[j][i] if testLenS[j][i]>0  else 0 ,2))+' '
                with open(preTestInfos[j],'a',encoding='UTF-8') as f:
                    f.write('\npretest trainenv: '+'\n+env: '+str(j+17)+'\n'+testErrorR+testAccR+testEndAccR+'\n')
           

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
    parser.add_argument('-cuda',type=str,default='cuda:0')
    parser.add_argument('-mode',type=int,default=1)
    parser.add_argument('-net',type=int,default=1)
    parser.add_argument('-preload',type=str2bool,default=False)
    parser.add_argument('-lr',type=float,default=0.00005)
    # 0.00005 0.0005
    parser.add_argument('-layers',type=int,default=5)
    parser.add_argument('-embed',type=int,default=128)
    parser.add_argument('-rems',type=int,default=5)
    parser.add_argument('-epochs',type=int,default=20)
    parser.add_argument('-batchsize',type=int,default=256)
    parser.add_argument('-restrict',type=str2bool,default=False)
    parser.add_argument('-appF',type=str2bool,default=False)
    parser.add_argument('-appP',type=str,default='restrict')
    parser.add_argument('-agentFlag',type=int,default=1)
    args=parser.parse_args()
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    agent=REMAgent2(device=device,rnn_layer=args.layers,embed_n=args.embed,flag=args.agentFlag)
    store=UTIL.getTimeStamp()
    loadPath=[
        '/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20240110/chooseway/4/model.pt',
        '/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20240110/chooseway/3/model.pt',
        '/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20240110/chooseway/2/model.pt',
        '/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20240110/chooseway/0/model.pt',
        '/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20240110/chooseway/1/model.pt'

    ]
    agent.load(loadPath[args.agentFlag-1])

    # if args.preload:
    #     agent.load('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20231220/trainallscene/0/dqnnetoffline.pt')
    mainPath=os.path.join('/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/',store[0:-6],'chooseway','smallbatch')
    if not os.path.exists(mainPath):
        os.makedirs(mainPath)
    fileNum=len(os.listdir(mainPath))
    if args.appF==False:
        store_path=os.path.join(mainPath,str(fileNum))
    else:
        store_path=os.path.join(mainPath,args.appP)
    print(store[-6:])
    envs=[]
    exclude=[]
    envTrain=[12, 14, 18, 21, 8, 2, 5, 20, 19, 9, 16, 10, 3, 15, 17]
    envTrain=[6, 11, 7, 4, 13]
    for i in envTrain:
        if i<10:
            envs.append('0'+str(i))
        else:
            envs.append(str(i))
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    with open(os.path.join(store_path,'envsinfo.txt'),'w') as f:
        f.write('envs'+str(envs)+'\n trainEnvs:'+str(envs)+'\n'+' lr:'+str(args.lr)+' preload: '+str(args.preload)+' device:'+str(device)\
                +' layers: '+str(args.layers)+" embded: "+str(args.embed)+' rems'+str(args.rems)+'\nexclude: '+str(exclude)+'\n')
        for flag in AgentFlag:
            if flag.value==args.agentFlag:
                f.write(flag.name+'\n')
                break
        
    train_epoch(agent, args.lr, args.epochs, args.batchsize,device,args.mode,store_path=store_path,multienvs=envs,\
               remBlocskNum=args.rems,restrict=args.restrict)