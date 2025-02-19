import torch
import os
import glob
from torch.utils.data import Dataset
import random
import sys
os.chdir(sys.path[0])
import time

def trans_num(nums):
    number=0.
    flag=1
    # print(nums)
    nums=nums.replace('\n','')
    for i in nums:
        if i=='':
            continue
        if i=='-':
            flag=-1
        else:
            number*=10
            number+=int(i)
    return flag*number


def trans_nums(nums):
    n = []
    for num in nums:
        if num=='':
            continue
        n.append(trans_num(num))
    return n

class dataset_loader(Dataset):
    def __init__(self, data_path):
        super(dataset_loader, self).__init__()
        d_path=os.path.join(data_path,'*.txt')
        self.data_path = glob.glob(d_path)

    def __getitem__(self, item):
        txt_path = self.data_path[item]
        nums = []
        with open(txt_path) as f:
            lines = f.readlines()
            for line in lines:
                line=line.replace('\n','')
                nums.append(trans_nums(line.split(' ')))
        eye_list =nums[0]
        next_eye_list=nums[1]
        last_goal=nums[2]
        if len(last_goal)==2:
            print(eye_list)
            print(last_goal)
            print(nums[2])
            print(txt_path)
            breakpoint()
        goal=nums[3]
        flag=nums[4]
        # s_=nums[2]
        return torch.tensor(eye_list,dtype=torch.float32).reshape(1,-1),torch.tensor(next_eye_list,dtype=torch.float32).reshape(1,-1),\
        torch.tensor(last_goal,dtype=torch.float32).reshape(1,-1),\
               torch.tensor(goal,dtype=torch.float32).reshape(1,-1),torch.tensor([flag],dtype=torch.float32)


    def __len__(self):
        return len(self.data_path)

if __name__=='__main__':
    od3 = dataset_loader('/home/wu_tian_ci/eyedata/mixed/pretrain2/s1/train')
    dl3=torch.utils.data.DataLoader(dataset=od3,shuffle=False,batch_size=1,num_workers=0)
    k=0
    a=torch.ones(1,1,6)
    b=torch.ones(1,1,9)
    c=torch.ones(1,1,3)
    d=torch.ones(1,1,1)
    # print(a.shape)
    for v,w,x,y,z in dl3:
        if a.shape!=v.shape or a.shape!=w.shape or b.shape!=x.shape or c.shape!=y.shape or d.shape!=z.shape:
            print(v,w,x,y,z)
        # print(v,w,x,y,z)
        # time.sleep(0.1)
