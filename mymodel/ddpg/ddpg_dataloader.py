import torch
import os
import glob
from torch.utils.data import Dataset
import random
import sys
os.chdir(sys.path[0])
import time
import json

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
        # next_eye_list=nums[1]
        last_goal=nums[1]
        if len(last_goal)==2:
            print(eye_list)
            print(last_goal)
            print(nums[2])
            print(txt_path)
            breakpoint()
        goal=nums[2]
        flag=nums[3]
        # s_=nums[2]
        return torch.tensor(eye_list,dtype=torch.float32).reshape(1,-1),\
        torch.tensor(last_goal,dtype=torch.float32).reshape(1,-1),\
               torch.tensor(goal,dtype=torch.float32).reshape(1,-1)


    def __len__(self):
        return len(self.data_path)




class dataset_loader_critic(Dataset):
    def __init__(self, data_path):
        super(dataset_loader_critic, self).__init__()
        # d_path=os.path.join(data_path,'*.txt')
        # self.data_path = glob.glob(d_path)
        f=open(data_path,'r')
        self.data=json.load(f)

    def __getitem__(self, item):
        data = self.data.get(str(item))
        nums = []
        state=data.get('0')
        near_state=data.get('1')
        last_action=data.get('2')
        n_state=data.get('3')
        n_near_state=data.get('4')
        n_last_action=data.get('5')
        last_goal=data.get('6')
        begin_=data.get('7')
        goal=data.get('8')
        is_end=data.get('9')
        epochs=data.get('10')
        # print(data)
        # last_goal,state,near_state,last_action,n_state,n_near_state,n_last_action,begin_,goal,is_end
        return torch.tensor(last_goal,dtype=torch.float32).reshape(1,-1),\
        torch.tensor(state,dtype=torch.float32).reshape(1,-1),\
               torch.tensor(near_state,dtype=torch.float32).reshape(1,-1),\
                torch.tensor(last_action,dtype=torch.float32).reshape(1,-1),\
                    torch.tensor(n_state,dtype=torch.float32).reshape(1,-1),\
                        torch.tensor(n_near_state,dtype=torch.float32).reshape(1,-1),\
                            torch.tensor(n_last_action,dtype=torch.float32).reshape(1,-1),\
                                torch.tensor(begin_,dtype=torch.float32).reshape(1,-1),\
                                    torch.tensor(goal,dtype=torch.float32).reshape(1,-1),\
                                        torch.tensor(is_end,dtype=torch.float32).reshape(1,-1),\
                                            torch.tensor(epochs,dtype=torch.float32).reshape(1,-1)
    def __len__(self):
        return len(self.data.keys())


class dataset_loader_policy(Dataset):
    def __init__(self, data_path):
        super(dataset_loader_policy, self).__init__()
        # d_path=os.path.join(data_path,'*.txt')
        # self.data_path = glob.glob(d_path)
        f=open(data_path,'r')
        self.data=json.load(f)

    def __getitem__(self, item):
        data = self.data.get(str(item))
        nums = []
        #states,near_state,last_goal,last_action,self.goal,[isend]
        #last_goal,state_n,n_state_n,mouse_n,goal,[is_end]
        state =data.get('0')
        near_state=data.get('1')
        last_goal=data.get('2')
        last_action=data.get('3')
        goal=data.get('4')
        is_end=data.get('5')
        return torch.tensor(last_goal,dtype=torch.float32).reshape(1,-1),\
        torch.tensor(state,dtype=torch.float32).reshape(1,-1),\
            torch.tensor(near_state,dtype=torch.float32).reshape(1,-1),\
               torch.tensor(last_action,dtype=torch.float32).reshape(1,-1),\
                torch.tensor(goal,dtype=torch.float32).reshape(1,-1)



    def __len__(self):
        return len(self.data.keys())



if __name__=='__main__':
    od3 = dataset_loader_policy('/home/wu_tian_ci/eyedata/mixed/5_policy_1last_move5/s1/train1.json')
    dl3=torch.utils.data.DataLoader(dataset=od3,shuffle=False,batch_size=5120,num_workers=18)
    # for a,b,c,d,e,f,g,h,i,j,k in dl3:
    #     # print(a,b,c,d,e,f,g,h,i,j,k)
    #     pass
    for a,b,c,d,e in dl3:
        print(a,b,c,d,e)
    # k=0
    # a=torch.ones(1,1,6)
    # b=torch.ones(1,1,9)
    # c=torch.ones(1,1,3)
    # d=torch.ones(1,1,1)
    # # print(a.shape)
    # for a,b,c,d in dl3:
    #     print(a.shape,b.shape,c.shape,d.shape)




