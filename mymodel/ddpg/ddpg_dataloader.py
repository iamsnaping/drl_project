import torch
import os
import glob
from torch.utils.data import Dataset
import random
import sys
os.chdir(sys.path[0])
import time
import json




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

        #states,near_state,last_goal,last_action,self.goal,[isend]
        #last_goal,state_n,n_state_n,mouse_n,goal,[is_end]
        state =data.get('0')
        near_state=data.get('1')
        last_goal=data.get('2')
        last_action=data.get('3')
        goal=data.get('4')
        is_end=data.get('5')
        goal_=data.get('6')
        return torch.tensor(last_goal,dtype=torch.float32).reshape(1,-1),\
        torch.tensor(state,dtype=torch.float32).reshape(1,-1),\
            torch.tensor(near_state,dtype=torch.float32).reshape(1,-1),\
               torch.tensor(last_action,dtype=torch.float32).reshape(1,-1),\
                torch.tensor(goal,dtype=torch.float32).reshape(1,-1),\
                    torch.tensor(goal_,dtype=torch.float32).reshape(1,-1)



    def __len__(self):
        return len(self.data.keys())



class data_prefetcher():
    def __init__(self,loader):
        self.loader=loader
        self.stream=torch.cuda.Stream()
        self.preload()
    
    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        # with torch.cuda.stream(self.stream):
        #     self.next_data = self.next_data.cuda(non_blocking=True)
            
    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data



if __name__=='__main__':
    od3 = dataset_loader_policy('/home/wu_tian_ci/eyedata/mixed/5_policy_1last_move5/s1/train1.json')
    dl3=torch.utils.data.DataLoader(dataset=od3,shuffle=False,batch_size=30720,num_workers=12,pin_memory=True)
    # for a,b,c,d,e,f,g,h,i,j,k in dl3:
    #     # print(a,b,c,d,e,f,g,h,i,j,k)
    #     pass
    begin_=time.process_time()
    print(begin_)
    for i in range(5):
        for a,b,c,d,e,f in dl3:
            ...
    print(time.process_time()-begin_)





