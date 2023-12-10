import os
import random
import shutil
import sys
sys.path.append('/home/wu_tian_ci/drl_project')
sys.path.append('/home/wu_tian_ci/drl_project/wtcutils')
import pandas as pd
from dqnutils import *
import dqnutils as UTIL
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


class DQNRNNEnv:
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

        self.lengths=[]
        self.lastEye=-1

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
        self.shuffle=True
        self.regularPath=[]
        self.topN=-1
        self.isRegular=[]
        self.eyeRegion=Region('eye')
        self.clickRegion=Region('click')
        self.eyeRegion.setRegions(UTIL.EYEAREAS)
        self.clickRegion.setRegions(UTIL.CLICKAREAS)
        self.eval=False

    def regular(self,paths):
        for p in paths:
            self.regularPath.append(p)

    def loadRegular(self):
        for p in self.regular:
            self.files_path.append(p)

    def load_dict(self):
        self.refresh()
        if self.isRegular:
            self.regular()
        elif self.eval==True:
            dirs=os.listdir(self.base_path)
            for dir in dirs:
                self.files_path.append(os.path.join(self.base_path,dir))
            self.files_path=self.files_path[self.topN:-1]
        else:
            dirs=os.listdir(self.base_path)
            for dir in dirs:
                self.files_path.append(os.path.join(self.base_path,dir))
            if self.topN!=-1:
                self.files_path=self.files_path[0:self.topN]
        if self.shuffle:
            random.shuffle(self.files_path)
            random.shuffle(self.files_path)
        self.current_dir_len=len(self.files_path)


    def get_file(self, appoint_path=None):
        if appoint_path is not None:
            self.df=pd.read_csv(self.files_path[self.current_file],header=None)
            return True
        self.current_file += 1
        if self.current_file >= self.current_dir_len:
            self.finish = True
            return False
        self.df=pd.read_csv(self.files_path[self.current_file],header=None)
        current_path=self.files_path[self.current_file]
        self.round_refresh()
        self.cfile_len = len(self.df)
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

        self.lengths=[]
        self.lastEye=-1
    
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
        self.goal=0.
        self.move_list=[]
        self.state_list=[]
        self.goal_nums=0
        self.next_state=[]
        self.total_num=0
        self.isend_total=0
        self.files_path=[]
        self.k1=0.
        self.k2=0
        self.states_idx=0
        self.states_len=0
        self.begin_idx=0
        self.end_idx=0
        self.action_list=[]
        self.near_states_list=[]

        self.lastEye=-1
        self.lengths=[]

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
            self.goal=self.clickRegion.judge(row[3],row[4])
            self.goal_nums+=1
    
    def get_1round(self):
        t=0
        increased=False
        self.lengths=[]
        self.move_list=[]
        self.state_list=[]
        while True:
            while len(self.last_goal_list)<3:
                self.get_goal()
                if len(self.last_goal_list)==0 or self.last_goal_list[-1]!=self.goal:
                    self.last_goal_list.append(self.goal)
                self.begin_idx=self.goal_nums
            else:
                if t==0 and self.last_goal_list[-1]!=self.goal:
                    self.last_goal_list.append(self.goal)
            # while flag:
            self.get_goal()
            if self.end_idx!=0:
                self.begin_idx=self.end_idx
            self.end_idx=self.goal_nums
            self.move_list=[]
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
                if self.lastEye==-1:
                    self.lastEye=self.eyeRegion.judge(row[1],row[2])
                    self.move_list.append(self.lastEye)
                    increase_=True
                else:
                    eyeRegion=self.eyeRegion.judge(row[1],row[2])
                    if eyeRegion!=self.lastEye:
                        self.lastEye=eyeRegion
                        self.move_list.append(eyeRegion)
                        increase_=True
                    else:
                        if i==self.begin_idx and self.goal!=self.last_goal_list[-1]:
                            self.lastEye=eyeRegion
                            self.move_list.append(eyeRegion)
                            increase_=True
                states,length=self.get_s()
                if increase_ and (not increased):
                    increased=True
                if increase_:
                    self.state_list.append(states)
                    self.lengths.append(length)
                if (i==self.end_idx-1) and (self.end_idx>=self.cfile_len-5):
                    break
            if self.end_idx>=self.cfile_len:
                self.file_finish=True
            self.states_idx=0
            t+=1
            if self.file_finish:
                break
            if increased:
                break



    '''
    sn-1 sn
    idx  states_len-1   


    '''
    def get_states(self):
        self.states_len=len(self.state_list)
        if self.states_idx>=self.states_len:
            return False
        states_idx=self.states_idx
        self.states_idx+=1
        # state near_state last_goal_list,last_goal,goal,isend
        return self.state_list[states_idx],self.last_goal_list[-3:],\
            self.goal,int(self.states_idx>=(self.states_len)),self.lengths[states_idx]
        

    def act(self):
        ans=self.get_states()
        if isinstance(ans,bool):
            if self.file_finish:
                flag=self.get_file()
                if flag==False:
                    return False
                self.get_1round()
                ans=self.get_states()
                return ans
            else:
                self.get_1round()
                ans=self.get_states()
                return ans
        return ans
        


    def get_s(self):
        m_list=[]

        if len(self.move_list)<10:
            m_list=dp(self.move_list)
        else:
            step=len(self.move_list)//10
            for i in range(9):
                m_list.append(self.move_list[i*step])
            m_list.append(self.move_list[-1])


        length=len(m_list)
        if length<10:
            m_list.extend([0 for i in range(10-length)])
        return m_list,length



      

if __name__ == '__main__':

    env2=DQNRNNEnv('/home/wu_tian_ci/eyedatanew/01/1')
    env2.load_dict()
    env2.get_file()
    k=0
    while True:
        ans=env2.act()
        k+=1
        if isinstance(ans,bool):
            env2.load_dict()
            env2.get_file()
            print(k)


    




