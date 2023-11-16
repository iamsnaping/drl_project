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
        # d_path=os.path.join(data_path,'*.txt')
        # self.data_path = glob.glob(d_path)
        f=open(data_path,'r')
        self.data=json.load(f)

    def __getitem__(self, item):
        data = self.data.get(str(item))

        #click,eye,goal,action,mask,reward,nclick,neye
        click =data.get('0')
        eye=data.get('1')
        goal=data.get('2')
        action=data.get('3')
        num=data.get('4')
        mask=data.get('5')
        reward=data.get('6')
        nclick=data.get('7')
        neye=data.get('8')
        nnum=data.get('9')
        click=torch.tensor(click,dtype=torch.int).reshape(1,-1)
        eye=torch.tensor(eye,dtype=torch.int).reshape(1,-1)
        goal=torch.tensor(goal,dtype=torch.float32).reshape(1,-1)
        action=torch.tensor(action,dtype=torch.int64).reshape(1,-1)
        num=torch.tensor(num,dtype=torch.float32).reshape(-1,1)
        mask=torch.tensor(mask,dtype=torch.float32).reshape(1,-1)
        reward=torch.tensor(reward,dtype=torch.float32).reshape(1,-1)
        nclick=torch.tensor(nclick,dtype=torch.int).reshape(1,-1)
        neye=torch.tensor(neye,dtype=torch.int).reshape(1,-1)
        nnum=torch.tensor(nnum,dtype=torch.float32).reshape(-1,1)
        return click,eye,goal,action,num,mask,reward,nclick,neye,nnum



    def __len__(self):
        return len(self.data.keys())




if __name__=='__main__':
    train_dataset=DQNDataLoader('/home/wu_tian_ci/eyedata/dqn/1/s1/train1.json')
    test_dataset=DQNDataLoader('/home/wu_tian_ci/eyedata/dqn/1/s1/test1.json')
    train_iter=torch.utils.data.DataLoader(dataset=train_dataset,shuffle=True,batch_size=12,num_workers=16,pin_memory=True)
    test_iter=torch.utils.data.DataLoader(dataset=test_dataset,shuffle=True,batch_size= 12,num_workers=16,pin_memory=True)
    for click,eye,goal,action,seq,mask,reward,nclick,neye,nseq in test_iter:
        print(seq.shape,nseq.shape)



