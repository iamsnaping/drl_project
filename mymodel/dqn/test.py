import random
import threading
import time

import numpy as np
import torch
import math
import os

import subprocess

import pandas as pd

# for i in [1,2,4]:
#     mainPath=os.path.join('/home/wu_tian_ci/eyedatanew/23',str(i))
#     dirList=os.listdir(mainPath)
#     newPath=os.path.join('/home/wu_tian_ci/eyedatanew/24',str(i))
#     if not os.path.exists(newPath):
#         os.makedirs(newPath)
#     for dir in dirList:
#         dirPath=os.path.join(mainPath,dir)
#         df=pd.read_csv(dirPath,header=None)
#         # print(len(df),print)
#         for i in range(len(df)):
#            df.iat[i,3]/=1.25
#            df.iat[i,4]/=1.25
#         newDirPath=os.path.join(newPath,dir)
#         df.to_csv(dirPath,header=None,index=False)


# hashDic={0:0,1:1,2:2,3:3,6:4,9:5}

# for i in [0,1,2,3,6,9]:
#     print(hashDic[i])

a=np.array([[4,5,6],[2,3,4],[1,2,3]])
print(np.mean(a,axis=0))