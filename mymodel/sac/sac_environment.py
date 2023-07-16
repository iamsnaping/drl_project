import functools
import os

import numpy as np
import pandas as pd
import random


class EyeEnvironment:
    def __init__(self,root_path,scheme):
        self.root_path=root_path
        self.scheeme=scheme
        self.dirs_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15',
                          '16', '17', '18', '19', '20', '21']
        # self.dirs_list = ['01', '02', '03', '04', '05']
        self.bwidth = 120
        self.bheight = 120
        self.nums = 0
        self.file_len = 0.
        self.file_list=[]
        self.trans_nums=0.
        self.finish = False
        self.current_file = -1
        self.current_dir_len = 0
        self.file_finish = False
        self.state_space=10
        self.action_space=3
        self.total_move=0.
        self.move_list=[0 for i in range(10)]
        self.last_goal=[0.,0.,0.]
        self.goal=[0.,0.,0.]
        self.goal_idx=0
        self.last_action=[0,0,0]
        self.gamma=0.1


    def load_dict(self):
        dirs=os.listdir(self.root_path)
        for dir in dirs:
            self.file_list.append(os.path.join(self.root_path,dir))
        self.current_dir_len=len(self.file_list)
        random.shuffle(self.file_list)
        random.shuffle(self.file_list)




    def get_file(self, appoint_path=None):
        if appoint_path is not None:
            self.df = pd.read_csv(appoint_path, header=None)
            return True
        self.current_file += 1
        if self.current_file >= self.current_dir_len:
            self.finish = True
            return False
        self.df=pd.read_csv(self.file_list[self.current_file], header=None)
        self.file_len = len(self.df)
        return True

    def get_state(self):
        flag=True
        while flag and self.nums<self.file_len:
            row = self.df.iloc[self.nums]
            if np.isnan(row[5]) or np.isnan(row[1]) or np.isnan(row[2]) or np.isnan(row[3]) or np.isnan(row[4]):
                flag=True
                self.nums+=1
                continue
            else:
                flag=False
            goal = [int(row[3] / self.bwidth + 0.5), int(row[4] / self.bheight + 0.5),int(row[5])]
        if flag==True:
            self.file_finish=self.nums>=self.file_len
            return -1,[]
        self.move_list.append(int(row[1] / self.bwidth + 0.5))
        self.move_list.append(int(row[2] / self.bwidth + 0.5))
        self.nums+=1
        self.file_finish=self.nums>=self.file_len
        return self.move_list[-10:], goal

    def get_goal(self):
        idx=self.nums
        flag=True
        while idx<self.file_len:
            row=self.df.iloc[idx]
            if np.isnan(row[5]) or np.isnan(row[1]) or np.isnan(row[2]) or np.isnan(row[3]) or np.isnan(row[4]):
                flag=True
                idx+=1
                continue
            else:
                flag=False
            if int(row[5])==0:
                self.last_goal=self.goal.copy()
                self.goal=[int(row[3]/self.bwidth+0.5),int(row[4]/self.bheight+0.5),int(row[5])]
                break
            idx+=1
    
    def get_reward(self,point):
        
        if self.goal_idx<=self.nums:
            dis_p=np.sqrt((self.goal[0]-point[0])**2+(self.goal[1]-point[1])**2)
            dis_l=np.sqrt((self.last_action[0]-self.goal[0])**2+(self.last_action[1]-self.goal[1])**2)
            dis=np.exp(-dis_p)
            v1=[point[0]-self.goal[0],point[1]-self.goal[1]]
            v2=[self.last_goal[0]-self.goal[0],self.last_goal[1]-self.goal[1]]
            if (np.linalg.norm(v1)*np.linalg.norm(v2))!=0:
                penalty=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
            else:
                penalty=0
            F=np.exp(-dis_p)-np.exp(-dis_l)
            self.last_action=point.copy()
            return self.gamma*(penalty*0.1+F+dis)
        dis_p=np.sqrt((self.goal[0]-point[0])**2+(self.goal[1]-point[1])**2)
        self.last_action=point.copy()
        return 1.5-np.exp(dis_p)



    def act(self):
        state = self.get_state()
        return self.file_finish, state

    def reset(self):
        self.nums = 0
        self.trans_nums=0
        self.last_goal=[0.,0.,0.]
        self.goal=[0.,0.,0.]
        self.move_list=[0 for i in range(10)]
        flag = self.get_file()
        if flag == False:
            return flag, []
        self.file_finish = False
        return flag, self.get_state()
    
    def refresh(self):
        self.nums = 0
        self.file_len = 0.
        self.file_list=[]
        self.trans_nums=0.
        self.finish = False
        self.current_file = -1
        self.current_dir_len = 0
        self.file_finish = False
        self.total_move=0.
        self.move_list=[0 for i in range(10)]
        self.last_goal=[0.,0.,0.]
        self.goal=[0.,0.,0.]







if __name__ == '__main__':
    env=EyeEnvironment('/home/wu_tian_ci/eyedata/seperate/01/1','1')
    env.load_dict()
    env.get_file()
    flag,state=env.reset()
    print(flag)
    while flag:
        kflag=False
        while not kflag:
            kflag,state=env.act()
            # print(env.nums)
            # print(state)
            if len(state[1])<3:
                break
            # if state[0]==-1:
            #     break
            if len(state[1])<3:
                print(state)
                break
        flag,state=env.reset()
        print('reset',flag)