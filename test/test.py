
import torch
from torch import nn
import numpy as np

import random
from torch.distributions import Normal
from torch.nn import functional as F

import pandas as pd

loss=torch.nn.CosineEmbeddingLoss(margin=0.5)
import os
import argparse
# torch.nn.KLDivLoss
import math
import collections
from torch.nn import functional as F


'''
multilayer lstm or gru

h0 -> batch_size -> num_layer
sequence_len -> input_batch_size

-> 0 -> 0 ->
   ~   ~   
   |   |   
-> 0 -> 0 ->
'''
# parser=argparse.ArgumentParser()
# parser.add_argument('--test',type=int)
# a=parser.parse_args()
# print(a.test)

import json


data={}
a={2:{1:['13'],2:[2,3]}}
b={3:{1:['3'],2:[1,3]}}
c={4:{1:['1'],2:[1,2,3]}}
d={1:{1:['123'],2:[1,2,3]}}
e={}
e[5]={1:['123'],2:[1,2,3]}
data.update(d)
data.update(a)
data.update(b)
data.update(c)
data.update(e)

f=open(r'/home/wu_tian_ci/eyedata/mixed/5_policy_1last_move5/s1/train1.json','w')
json.dump(data,f,indent=2)