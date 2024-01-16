import torch
import os
import glob
from torch.utils.data import Dataset
import sys
os.chdir(sys.path[0])
import time
import json


class DQNDataLoader(Dataset):
    def __init__(self, data_path):
        super(DQNDataLoader, self).__init__()
        f=open(data_path,'r')
        self.data=json.load(f)

    def __getitem__(self, item):
        data = self.data.get(str(item))
        # eye,click,goal,length
        click =data.get('1')
        eye=data.get('0')
        goal=data.get('2')
        length=data.get('3')
        person=data.get('4')
        scene=data.get('5')
        # print()
        clickTensor=torch.tensor([click],dtype=torch.long)
        eyeTensor=torch.tensor([eye],dtype=torch.long)
        lenTensor=torch.tensor(length,dtype=torch.long)
        personTensor=torch.tensor([[person]],dtype=torch.long)
        sceneTensor=torch.tensor([[scene]],dtype=torch.long)
        goalTensor=torch.tensor([[goal]],dtype=torch.long)

        return clickTensor,eyeTensor,goalTensor,lenTensor,personTensor,sceneTensor



    def __len__(self):
        return len(self.data.keys())




if __name__=='__main__':
    train_dataset=DQNDataLoader('/home/wu_tian_ci/eyedata/dqn_policy/train.json')
    train_iter=torch.utils.data.DataLoader(dataset=train_dataset,shuffle=True,batch_size=256,num_workers=16,pin_memory=True)
    t=0
    for clickTensor,eyeTensor,goalTensor,lenTensor,personTensor,sceneTensor in train_iter:
       t+=1
    print(t)



