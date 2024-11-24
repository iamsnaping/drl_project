import random
import threading
import time

import numpy as np
import torch
import math
import os

import subprocess
from dqnutils import *
import dqnutils as UTIL
import networkx as nx
import matplotlib.pyplot as plt
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

# basePath='/home/wu_tian_ci/eyedata/seperate'


eyeRegion=Region('eye')
clickRegion=Region('click')
eyeRegion.setRegions(UTIL.EYEAREAS)
clickRegion.setRegions(UTIL.CLICKAREAS)

source=[]
target=[]
nameMap=dict()
for i in range(2,22):
    if i<10:
        people='0'+str(i)
    else:
        people=str(i)
    for j in range(1,5):
        filesPath=os.path.join(basePath,people,str(j))
        fileList=os.listdir(filesPath)
        for f in fileList:
            Flag=True
            filePath=os.path.join(filesPath,f)
            df=pd.read_csv(filePath,header=None)
            dfLen=len(df)
            node=''
            lastG=-1
            for k in range(dfLen):
                row=df.iloc[k]
                if np.isnan(row[5]) or np.isnan(row[1]) or np.isnan(row[2]) or np.isnan(row[3]) or np.isnan(row[4]):
                    continue
                flag=row[5]
                if flag!=0:
                    continue
                g=str(clickRegion.judge(row[3]*1.25,row[4]*1.25))
                if lastG==g:
                    continue
                lastG=g
                goal=';A:'+str(clickRegion.judge(row[3]*1.25,row[4]*1.25))
                subNode='S:'+str(j)+goal
                if Flag:
                    node=dp(subNode)
                    Flag=False
                    # print('1',node,subNode,node)
                else:
                    # print(node,subNode)
                    if nameMap
                    source.append(dp(node))
                    target.append(dp(subNode))
                    node=dp(subNode)
                    # breakpoint()

df=pd.DataFrame({"source":source,"target":target})
df.to_excel('relation2.xlsx',index=False)


# def getR(df,name):
#     source=df['source']
#     target=df['target']
#     sourceDict=dict()
#     G=nx.Graph()
#     counts=1
#     nodes=[]
#     edges=[]
#     for s,t in zip(source,target):
#         s_n=sourceDict.get(s)
#         t_n=sourceDict.get(t)
#         if s_n is None:
#             sourceDict[s]=counts
#             s_n=counts
#             nodes.append(s_n)
#             counts+=1
#         if t_n is None:
#             sourceDict[t]=counts
#             t_n=counts
#             nodes.append(t_n)
#             counts+=1
#         edges.append((s_n,t_n))

#     G.add_nodes_from(nodes)
#     G.add_edges_from(edges)
#     r=nx.degree_assortativity_coefficient(G)
#     nx.draw(G,with_labels=True)
#     plt.savefig(name+'.png')
#     print(name,':',r)

# df1=pd.read_excel('/home/wu_tian_ci/relation.xlsx',sheet_name='Sheet1',header=0)
# df2=pd.read_excel('/home/wu_tian_ci/relation2.xlsx',sheet_name='Sheet1',header=0)
# getR(df1,'relations1')
# getR(df2,'relations2')



