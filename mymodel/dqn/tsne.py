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
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data,Batch
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
from torch import nn
import numpy as np
from torch_geometric import nn as tnn
from torch_geometric.utils import dense_to_sparse
import numpy as np
from sklearn.manifold import TSNE


# # 固定随机种子，保证代码可复现性
# def weight_init(m):
#     random.seed(1114)
#     np.random.seed(1114)
#     torch.manual_seed(1114)
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_normal_(m.weight)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0.0)
#     elif isinstance(m, nn.LayerNorm):
#         nn.init.constant_(m.weight, 1.0)
#         nn.init.constant_(m.bias, 0.0)

# # GCN归一化
# class GCNChain(nn.Module):
#     def __init__(self,input_dim,hidden_dim):
#         super(GCNChain,self).__init__()
#         self.cov=GCNConv(input_dim,hidden_dim)
#         #
#         self.norm=tnn.MessageNorm(learn_scale=True)
#         self.activation=nn.GELU()
    
#     def forward(self,X,edge_index):
#         return self.activation(self.norm(X,self.cov(X,edge_index)))

# #  构建Deeper GCN
# class GCNFFN(nn.Module):
#     def __init__(self,input_dim,hidden_dim):
#         super(GCNFFN,self).__init__()
#         self.ffn1=GCNChain(input_dim,hidden_dim)
#         self.ffn2=GCNChain(hidden_dim,input_dim)
#         self.norm=nn.Sequential(tnn.GraphNorm(input_dim),nn.GELU())
#     def forward(self,X,edge_index):
#         X=X.transpose(0,1)    
#         f=self.ffn1(X,edge_index)
#         f=self.ffn2(f,edge_index)
#         return self.norm(f+X)
# # GCN
# class GCN(nn.Module):
#     def __init__(self, input_dim, hidden_dim,layers,threshold=0.7):
#         super(GCN, self).__init__()
#         self.gcn=nn.ModuleList()
#         self.input_dim=input_dim
#         self.hidden_dim=hidden_dim
#         self.threshold=threshold
#         for i in range(layers):
#             self.gcn.append(GCNFFN(input_dim,hidden_dim))

#     # 将数据处理成GCN能接受的形式
#     @torch.no_grad()
#     def pre_process(self,data,adj):

#         edge_index = torch.nonzero(adj).t().contiguous()

#         # 将节点特征和边索引转换为Data对象
#         data = Data(x=data, edge_index=edge_index)

#         # 构建 PyTorch Geometric 的数据对象
#         # data = Data(x=data, edge_index=edge_index, edge_attr=edge_attr)
#         return data

#     def forward(self, x,adj):
#         for layer in self.gcn:
#             x=layer(x,adj)
#         return x
# # embedding: 用于将各种操作映射如高维空间，更易于提取特征
# class MyModel(nn.Module):
#     def __init__(self,input_dim, hidden_dim,layers,threshold=0.7):
#         super().__init__()
#         self.gcn=GCN(input_dim, hidden_dim,layers,threshold)
#         self.embedding=nn.Embedding(200,128)
    
#     def forward(self,data):
#         X,adj = data.x, data.edge_index
#         return torch.mean(self.gcn(self.embedding(X),adj),dim=-1)


# if __name__=='__main__':
#     gcn=GCN(10,20,2)
#     X = torch.rand(2,10,10)

# # for i in [1,2,4]:
# #     mainPath=os.path.join('/home/wu_tian_ci/eyedatanew/23',str(i))
# #     dirList=os.listdir(mainPath)
# #     newPath=os.path.join('/home/wu_tian_ci/eyedatanew/24',str(i))
# #     if not os.path.exists(newPath):
# #         os.makedirs(newPath)
# #     for dir in dirList:
# #         dirPath=os.path.join(mainPath,dir)
# #         df=pd.read_csv(dirPath,header=None)
# #         # print(len(df),print)
# #         for i in range(len(df)):
# #            df.iat[i,3]/=1.25
# #            df.iat[i,4]/=1.25
# #         newDirPath=os.path.join(newPath,dir)
# #         df.to_csv(dirPath,header=None,index=False)


# # hashDic={0:0,1:1,2:2,3:3,6:4,9:5}

# # for i in [0,1,2,3,6,9]:
# #     print(hashDic[i])

# basePath='/home/wu_tian_ci/eyedata/seperate'


# eyeRegion=Region('eye')
# clickRegion=Region('click')
# eyeRegion.setRegions(UTIL.EYEAREAS)
# clickRegion.setRegions(UTIL.CLICKAREAS)

# nameMap=dict()
# batch_=[]
# def createMapAdj(target,source):
#     data=torch.zeros((13,1),dtype=torch.long)
#     adj=torch.zeros((2,len(target)),dtype=torch.long)
#     d=dict()
#     idx=1
#     for i in range(len(target)):
#         if d.get(source[i]) is None:
#             d[source[i]]=idx
#             idx+=1
#         if d.get(target[i]) is None:
#             d[target[i]]=idx
#             idx+=1
#     for value,idx in d.items():
#         data[idx,0]=value
#     # breakpoint()
#     counting=dict()
#     for i in range(len(target)):
#         counts=str(source[i])+str(target[i])
#         if counting.get(counts) is None:
#             counting[counts]=1
#             continue
#         adj[0,i]=d[source[i]]
#         adj[1,i]=d[target[i]]
#     data_=Data(x=data,edge_index=adj)
#     return data_

# datas=[]

# # 模型实例化，并初始化
# gcn=MyModel(128,128,1)
# gcn.apply(weight_init)
# for i in range(2,22):
#     if i<10:
#         people='0'+str(i)
#     else:
#         people=str(i)
#     for j in range(1,5):
#         filesPath=os.path.join(basePath,people,str(j))
#         fileList=os.listdir(filesPath)
#         length=0
#         source=[]
#         target=[]
#         for f in fileList:
#             Flag=True
#             filePath=os.path.join(filesPath,f)
#             df=pd.read_csv(filePath,header=None)
#             dfLen=len(df)
#             node=''
#             lastG=-1
#             for k in range(dfLen):
#                 row=df.iloc[k]
#                 if np.isnan(row[5]) or np.isnan(row[1]) or np.isnan(row[2]) or np.isnan(row[3]) or np.isnan(row[4]):
#                     continue
#                 flag=row[5]
#                 if flag!=0:
#                     continue
#                 g=str(clickRegion.judge(row[3]*1.25,row[4]*1.25))
#                 if lastG==g:
#                     continue
#                 lastG=g
#                 goal=';A:'+str(clickRegion.judge(row[3]*1.25,row[4]*1.25))
#                 if i>=10:
#                     subNode='P:'+str(i-1)+'S:'+str(j)+goal
#                 else:
#                     subNode='P:0'+str(i-1)+'S:'+str(j)+goal
#                 if Flag:
#                     node=dp(subNode)
#                     Flag=False
#                     # print('1',node,subNode,node)
#                 else:
#                     # print(node,subNode)
#                     if nameMap.get(node) is None:
#                         nameMap[node]=length
#                         length+=1
#                     if nameMap.get(subNode) is None:
#                         nameMap[subNode]=length
#                         length+=1
      
#                     source.append(nameMap[node])
#                     target.append(nameMap[subNode])
#                     node=dp(subNode)
#         data_=createMapAdj(source,target)
#         ans=gcn(data_)
#         datas.append(ans.squeeze().detach().numpy())
#                     # breakpoint()
#     print(i)
# datas=np.array(datas)
# np.save('datas1',datas)
datas=np.load('datas1.npy')
c=[] 
cmps=plt.colormaps()
for i in range(20):
    c.append(i)
    c.append(i)
    c.append(i)
    c.append(i)

plt.clf()
# TSNE 实例化
# 这4个参数的意义分别为，数据将降至的维数、空间聚类的邻居数、学习率为，初始化状态。
tsne=TSNE(n_components=2,perplexity=80,learning_rate=50,random_state=0)
# x_embedded:降维后的数据 datas:原始数据
x_embedded=tsne.fit_transform(datas)
# 可视化
plt.scatter(x_embedded[:,0,],x_embedded[:,1],c=c,cmap=plt.cm.get_cmap('tab20',20),s=30)
plt.savefig('figs/relation'+'k3'+'.png')
# tsne=TSNE(n_components=2,random_state=5)
# x_embedded=tsne.fit_transform(datas)
# plt.clf()
# plt.scatter(x_embedded[:,0],x_embedded[:,1],c=c,s=10)
# plt.savefig('figs/relation.png')


