import random
import threading
import time

import numpy as np
import torch
import math


rnn=torch.nn.LSTM(10,20,5)
a=torch.ones(3).long()
b=torch.ones(4).long()
embeding=torch.nn.Embedding(2,10)
c=torch.nn.utils.rnn.pad_sequence([b,a])
# 
dd=embeding(c)
print(dd.shape)
packed=torch.nn.utils.rnn.pack_padded_sequence(dd,torch.tensor([4,3]))
h0=torch.zeros(5,2,20)
h1=torch.zeros(5,2,20)
print(h0.shape)
ans,(hn, cn) =rnn(packed,(h0,h1))
unpack,_=torch.nn.utils.pad_packed_sequence(ans)

unpack=unpack.permute(1,0,2)
sub=torch.ones(10)
print(len(sub),sub.shape[0])
# print(unpack)
# print(unpack[[0,1],[3,2],:])
# print(unpack)
# print(unpack,_)
# print(unpack[[i for i in range(len(_))],_-1,:])

# b=torch.randn(1,2,3)
# print(b)
# b=b.repeat(3,1,1)
# print(b)

# k=torch.randn(2,5)
# print(k)

# print(k.flatten())

a=torch.tensor([1,2,3])
b=torch.tensor([2,3,4])
c=torch.tensor([3,4,5])
d=torch.stack([a,b,c],dim=0)
e=torch.tensor([[2,2,2,]])
print(d,d+e)


# e=np.random.sample(d,2)
# print(e[:,2])

# print(torch.stack([a,b],dim=0))
# c=torch.tensor(4)
# b=torch.tensor(3)
# print(torch.stack([b,c],dim=0))

# print(torch.arange(0,10).shape)

# k=torch.tensor([[1,2,3],[2,3,4]])
# print(k[:,2].unsqueeze())
