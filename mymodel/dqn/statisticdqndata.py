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
import pandas as pd


def tallyFile(filePath):
    df=pd.read_csv(filePath,header=None)
    clickRegion=Region('click')
    eyeRegion=Region('eye')
    clickRegion.setRegions(CLICKAREAS)
    eyeRegion.setRegions(EYEAREAS)
    tallyList=[0 for i in range(6)]
    skipArea=[0 ,1, 2 ,3 ,6, 9]
    hashDic=dict()
    for i,j in zip(range(6),skipArea):
        hashDic[j]=i
    lastArea=-1
    for i in range(len(df)):
        row=df.iloc[i]
        if np.isnan(row[5]) or np.isnan(row[1]) or np.isnan(row[2]) or np.isnan(row[3]) or np.isnan(row[4]):
            continue
        flag=row[5]        
        if flag==0:
            goal=clickRegion.judge(row[3]*1.25,row[4]*1.25)
            if goal not in skipArea:
                continue
            if lastArea==-1:
                lastArea=goal
                # print(hashDic[goal],goal)
                tallyList[hashDic[goal]]+=1
            else:
                if lastArea!=goal:
                    lastArea=goal
                    tallyList[hashDic[goal]]+=1
    return tallyList


def tally(filesPath,scenes,testEnvs):
    writePath='/home/wu_tian_ci/drl_project/mymodel/dqn/pretrain_data/offlinedqn/20231222'
    
    clickRegion=Region('click')
    eyeRegion=Region('eye')
    clickRegion.setRegions(CLICKAREAS)
    eyeRegion.setRegions(EYEAREAS)
  
    for i in range(len(testEnvs)):
        for j in range(len(scenes)):
            envPath=os.path.join(filesPath,testEnvs[i],str(scenes[j]))
            dirList=os.listdir(envPath)
            tallyList=np.zeros((len(dirList),6))
            for k in range(len(dirList)):
                dir=os.path.join(envPath,dirList[k])
                tallyList[k]=tallyFile(dir)
            print(f'env {testEnvs[i]} scene {scenes[j]} mean {np.mean(tallyList,axis=0)}')
            print(f'env {testEnvs[i]} scene {scenes[j]} std {np.std(tallyList,axis=0)}\n')




    





if __name__=='__main__':
    filePath='/home/wu_tian_ci/eyedata/seperate'
    scenes=[1,2,4]
    testEnvs=[]
    for i in range(17,22):
        testEnvs.append(str(i))
    # filePath='/home/wu_tian_ci/eyedatanew'
    # scenes=[1,2,4]
    # testEnvs=['23'] 
    tally(filePath,scenes,testEnvs)
