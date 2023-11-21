import os
import time

import numpy as np
import pandas as pd
import csv

# dataList=[[1,1,1],[2,2,2],[3,3,3]]
# df=pd.DataFrame(dataList)
# df.to_csv('test3.csv',header=None,index=False)
#
# # b=40
# # c=-1
# #
# # print(np.clip(b,0,20),np.clip(c,0,20))
#
# print(os.listdir('D:\\tobii\\tobii'))
import torch
from  torch import  Tensor
# a=torch.tensor([[1,2,3]])
# b=torch.tensor([[2,3,4]])
# print(torch.cat([a,b],dim=0).shape)
# k=torch.tensor([[1,2,3],[2,3,4],[3,4,5]])
# print(k[0],k[[0,1]])


# k=(0,1,2,3,4,5)
# def fun(b):
#     c,d,e,f,g=b
#     print(c,d,e,f,g)
# print(k[1:-1])
# fun(k[1:])
# a=torch.tensor([[1,2,3]]).unsqueeze(0)
# print(a.shape,a)
# print(a.squeeze(1).shape,a.squeeze(1))


# a=torch.tensor([[[1]]])
# print(a.shape)
# print(a.squeeze().shape)
# print(a.device)
# b=a.reshape((2,2))
# print(a,b)

handList=np.array([[1,2],[2,3],[3,4]])

ave = handList.mean(axis=0)
scores=np.sqrt(np.mean(np.sum(np.square(handList-ave),axis=-1)))
print(scores)