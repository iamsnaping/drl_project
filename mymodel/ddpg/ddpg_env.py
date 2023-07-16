import os
import random
import shutil
import sys
sys.path.append('/home/wu_tian_ci/drl_project')
sys.path.append('/home/wu_tian_ci/drl_project/wtcutils')
import pandas as pd

import wtcutils.constant as wtuc
from mymodel.ddpg.ddpg_base_net import *

s_path='/home/wu_tian_ci/eyedata/seperate'
m_path='/home/wu_tian_ci/eyedata/mixed'

root_path='/home/wu_tian_ci/eyedata'

scheme=['s1','s2','s3','s4']

s1,s2,s3,s4=[],[],[],[]

origin_list=os.listdir(root_path)


'''
seperate/train\test/
miexd/train\test/

'''
import os

import numpy as np
import pandas as pd
import random
from tqdm import tqdm

class DDPGEnv:
    #所有指针都指向下一个元素
    def __init__(self,base_path):
        self.file_finish = False
        self.files=[]
        self.files_len=0
        self.df=None
        self.nums=0
        self.bwidth=120
        self.bheight=120
        self.state_space=60
        self.current_file=-1
        self.cfile_len=0
        self.trans_nums=0
        self.trans_path=''
        self.last_goal_list=[]
        self.goal=[0.,0.,0.]
        self.move_list=[]
        self.state_list=[]
        self.near_states_list=[]
        self.goal_nums=0
        self.next_state=[]
        self.total_num=0
        self.isend_total=0
        self.k1=0.
        self.k2=0
        self.mode=3
        self.states_idx=0
        self.states_len=0
        self.files_path=[]
        self.base_path=base_path
        self.begin_idx=0
        self.end_idx=0
        self.verify_list=[[] for i in range(7)]
        self.verify_dict=dict()
        self.verify_num=0
        self.action_list=[]
        # 1-> origin  2-> not origin
        self.state_mode=1
        
    def load_dict(self):
        dirs=os.listdir(self.base_path)
        for dir in dirs:
            self.files_path.append(os.path.join(self.base_path,dir))
        self.current_dir_len=len(self.files_path)
        random.shuffle(self.files_path)
        random.shuffle(self.files_path)

    def get_file(self, appoint_path=None):
        if appoint_path is not None:
            # self.df = pd.read_feather(appoint_path)
            self.df=pd.read_csv(self.files_path[self.current_file],header=None)
            return True
        self.current_file += 1
        if self.current_file >= self.current_dir_len:
            self.finish = True
            return False
        # self.df=pd.read_feather(self.files_path[self.current_file])
        self.df=pd.read_csv(self.files_path[self.current_file],header=None)
        current_path=self.files_path[self.current_file]
        # if self.verify_dict.get(current_path) is None:
            # self.verify_dict[current_path]=self.verify_num
            # self.verify_num+=1
        self.round_refresh()
        self.cfile_len = len(self.df)
        # print(self.files_path[self.current_file])
        # print(self.current_dir_len)
        return True


    def round_refresh(self):
        self.file_finish = False
        self.files=[]
        self.files_len=0
        self.nums=0
        self.cfile_len=0
        self.trans_nums=0
        self.last_goal_list=[]
        self.goal=[0.,0.,0.]
        self.move_list=[]
        self.state_list=[]
        self.near_states_list=[]
        self.goal_nums=0
        self.next_state=[]
        self.total_num=0
        self.states_idx=0
        self.states_len=0
        self.begin_idx=0
        self.end_idx=0
        self.action_list=[]
    
    def refresh(self):
        self.file_finish = False
        self.files=[]
        self.files_len=0
        self.df=None
        self.nums=0
        self.bwidth=120
        self.bheight=120
        self.state_space=60
        self.current_file=-1
        self.cfile_len=0
        self.trans_nums=0
        self.trans_path=''
        self.last_goal_list=[]
        self.goal=[0.,0.,0.]
        self.move_list=[]
        self.state_list=[]
        self.goal_nums=0
        self.next_state=[]
        self.total_num=0
        self.isend_total=0
        self.k1=0.
        self.k2=0
        self.mode=3
        self.states_idx=0
        self.states_len=0
        self.begin_idx=0
        self.end_idx=0
        self.action_list=[]
        self.near_states_list=[]

    def get_goal(self):
        flag=-1
        while self.goal_nums<self.cfile_len:
            row=self.df.iloc[self.goal_nums]
            if np.isnan(row[5]) or np.isnan(row[1]) or np.isnan(row[2]) or np.isnan(row[3]) or np.isnan(row[4]):
                self.goal_nums+=1
                continue  
            flag=row[5]
            if flag==0:
                break
            self.goal_nums+=1
        while flag ==0 and self.goal_nums<self.cfile_len:
            row=self.df.iloc[self.goal_nums]
            if np.isnan(row[5]) or np.isnan(row[1]) or np.isnan(row[2]) or np.isnan(row[3]) or np.isnan(row[4]):
                break
            flag=int(row[5])
            if flag !=0:
                break
            self.goal=[row[3],row[4]]
            self.goal_nums+=1
    
    def get_1round(self):
        if len(self.last_goal_list)<9:
            self.get_goal()
            self.last_goal_list.append(self.goal[0])
            self.last_goal_list.append(self.goal[1])
            # self.last_goal_list.append(self.goal[2])
            self.get_goal()
            self.last_goal_list.append(self.goal[0])
            self.last_goal_list.append(self.goal[1])
            # self.last_goal_list.append(self.goal[2])
            self.get_goal()
            self.last_goal_list.append(self.goal[0])
            self.last_goal_list.append(self.goal[1])
            # self.last_goal_list.append(self.goal[2])
            self.begin_idx=self.goal_nums
        else:
            self.last_goal_list.append(self.goal[0])
            self.last_goal_list.append(self.goal[1])
            # self.last_goal_list.append(self.goal[2])
        flag=True
        self.move_list.clear()
        self.move_list=[]
        self.state_list=[]
        self.near_states_list=[]
        # while flag:
        self.get_goal()
        if self.end_idx!=0:
            self.begin_idx=self.end_idx
        self.end_idx=self.goal_nums
        x,y=-1,-1
        xx,yy=-1,-1
        last_goal=self.last_goal_list[-6:].copy()
        states=[]
        near_states=[]
        self.move_list=[]
        last_action=[0.,0.]
        # self.k1=0
        self.k2+=1
        for i in range(self.begin_idx,self.end_idx):
            row=self.df.iloc[i]
            increase_=False
            if len(row)<5:
                self.nums+=1
                continue
            if np.isnan(row[5]) or np.isnan(row[1]) or np.isnan(row[2]) or np.isnan(row[3]) or np.isnan(row[4]):
                self.nums+=1
                continue
            if x==-1 and y==-1:
                x,y=int(row[1]/120+0.5),int(row[2]/120+0.5)
                self.move_list.append([x,y])
                xx,yy=row[3],row[4]
                increase_=True
            else:
                x1,y1=int(row[1]/120+0.5),int(row[2]/120+0.5)
                if self.state_mode==1:
                    self.move_list.append([x1,y1])
                if x!=x1 or y!=y1:
                    last_action[0],last_action[1]=xx,yy
                    xx,yy=row[3],row[4]
                    x=x1
                    y=y1
                    if self.state_mode==2:
                        self.move_list.append([x,y])
                    increase_=True
                    self.k1+=1
            states,near_states=self.get_s()
            # isend=0
            if increase_:
                self.action_list.append(last_action)
                self.state_list.append(states)
                self.near_states_list.append(near_states)
            if (i==self.end_idx-1) and (self.end_idx>=self.cfile_len-5):
                break
        if self.end_idx>=self.cfile_len:
            self.file_finish=True
        self.action_list.append([-1,-1])
        self.state_list.append([-1 for i in range(len(self.state_list[-1]))])
        self.near_states_list.append([-1 for i in range(len(self.near_states_list[-1]))])
        self.states_idx=0



    '''
    sn-1 sn
    idx  states_len-1   


    '''
    def get_states(self):
        self.states_len=len(self.state_list)
        # if self.states_len==1 and self.states_idx<self.states_len:
        #     states_idx=self.states_idx
        #     self.states_idx+=1
        #     return self.state_list[states_idx],self.state_list[states_idx],self.last_goal_list[-9:],int(self.states_idx>=self.states_len-1)
        if self.states_idx>=self.states_len:
            return False
        states_idx=self.states_idx
        self.states_idx+=1
        # state near_state last_goal_list,last_goal,goal,isend
        # print(self.states_idx,self.states_len-1,int(self.states_idx>=(self.states_len-1)))
        if self.states_idx>=(self.states_len):
            return self.state_list[states_idx],self.near_states_list[states_idx],[-1 for i in range(6)],\
                [-1,-1],[-1,-1],int(self.states_idx>=(self.states_len))
        return self.state_list[states_idx],self.near_states_list[states_idx],self.last_goal_list[-6:],self.last_goal_list[-2:],self.goal,int(self.states_idx>=(self.states_len))
        

    def act(self):
        ans=self.get_states()
        if isinstance(ans,bool):
            if self.file_finish:
                flag=self.get_file()
                if flag==False:
                    return False
                self.get_1round()
                ans=self.get_states()
                # self.verify_list[self.verify_dict[self.files_path[self.current_file]]].append(ans)
                return ans
            else:
                self.get_1round()
                ans=self.get_states()
                # self.verify_list[self.verify_dict[self.files_path[self.current_file]]].append(ans)
                return ans
        # self.verify_list[self.verify_dict[self.files_path[self.current_file]]].append(ans)
        return ans
        


    def get_s(self):
        m_list=[]
        near_m_list=[]
        if len(self.move_list)>=10:
            m_list.append(self.move_list[0][0])
            m_list.append(self.move_list[0][1])
            for i in range(8):
                m_list.append(self.move_list[len(self.move_list)//9*(i+1)][0])
                m_list.append(self.move_list[len(self.move_list)//9*(i+1)][1])
            m_list.append(self.move_list[-1][0])
            m_list.append(self.move_list[-1][1])
        else:
            put_list=[(10//len(self.move_list)) for i in range(len(self.move_list))]
            k=(10-sum(put_list))//len(put_list)
            for i in range(int(len(put_list)//2)):
                put_list[i]+=k
                put_list[len(put_list)-i-1]+=k
            put_list[int(len(put_list)/2+0.5)-1]+=10-sum(put_list)
            # put_list[(len(put_list)-1)//2]+=10-sum(put_list)
            j=0
            for pt in put_list:
                for i in range(pt):
                    m_list.append(self.move_list[j][0])
                    m_list.append(self.move_list[j][1])
                j+=1
        near_=self.move_list[-(max(len(self.move_list)//3,1)):]
        if len(near_)>=5:
            near_m_list.append(near_[0][0])
            near_m_list.append(near_[0][1])
            for i in range(3):
                near_m_list.append(near_[len(near_)//4*(i+1)][0])
                near_m_list.append(near_[len(near_)//4*(i+1)][1])
            near_m_list.append(near_[-1][0])
            near_m_list.append(near_[-1][1])
        else:
            put_list=[(5//len(near_)) for i in range(len(near_))]
            k=(5-sum(put_list))//len(put_list)
            for i in range(int(len(put_list)//2)):
                put_list[i]+=k
                put_list[len(put_list)-i-1]+=k
            put_list[int(len(put_list)/2+0.5)-1]+=5-sum(put_list)
            # put_list[(len(put_list)-1)//2]+=10-sum(put_list)
            j=0
            for pt in put_list:
                for i in range(pt):
                    near_m_list.append(near_[j][0])
                    near_m_list.append(near_[j][1])
                j+=1
        return m_list,near_m_list





# def verify(ft1,ft2):
#     files_list=os.listdir('/home/wu_tian_ci/eyedata/seperate/01/1')
#     for fl in files_list:
#         fp=os.path.join('/home/wu_tian_ci/eyedata/seperate/01/1',fl)
#         list1=ft1.verify_list[ft1.verify_dict[fp]]
#         list2=ft2.verify_list[ft2.verify_dict[fp]]
#         if len(list1) != list2:
#             print('not equal')
#             return
#         for states1,states2 in zip(list1,list2):
#             for state1,states2 in zip(states1,states2):
#                 if isinstance(state1,states2):


        


      

if __name__ == '__main__':
    # env=DDPGEnv('/home/wu_tian_ci/eyedata/feather/seperate/01/1')
    # env.load_dict()
    # env.get_file()
    # t=0
    # while True:
    #     ans=env.act()
    #     if isinstance(ans,bool):
    #         break
    #         # env.refresh()
    #     else:
    #         state=torch.tensor([ans[0]],dtype=torch.float32).unsqueeze(0)
    #         last_goal=torch.tensor([ans[1]],dtype=torch.float32).unsqueeze(0)
    #         t+=1
    # print(t)
    t=0

    env2=DDPGEnv('/home/wu_tian_ci/eyedata/seperate/02/1')
    env2.load_dict()
    env2.get_file()
    while True:
        ans=env2.act()
        if isinstance(ans,bool):
            break
            # env.refresh()
        else:
            state=torch.tensor([ans[0]],dtype=torch.float32).unsqueeze(0)
            last_goal=torch.tensor([ans[1]],dtype=torch.float32).unsqueeze(0)
            print(ans[4])
            t+=1
    print(t)
        
    # print(t)
    # print(len(os.listdir('/home/wu_tian_ci/eyedata/mixed/3_policy3/s1/train')))

    




