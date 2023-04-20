import os
import random
import shutil

import pandas as pd



s_path='D:\\eyedata\\seperate'
m_path='D:\\eyedata\\mixed'

root_path='D:\\eyedata'

scheme=['s1','s2','s3','s4']

s1,s2,s3,s4=[],[],[],[]

origin_list=os.listdir(root_path)


'''
seperate/train\test/
miexd/train\test/

'''
def move_files():
    for origin in origin_list:
        if len(origin)>5:
            continue
        origin_path=os.path.join(root_path,origin)
        file_list=os.listdir(origin_path)
        f_len=len(file_list)
        train_nums=int(0.8*f_len+0.5)
        t=0
        random.shuffle(file_list)
        for f in file_list:
            if f=='record.csv':
                continue
            # print(f)
            t+=1
            print(file_list,origin_path)
            mode=int(f[2])
            factor=random.random()
            s_train_path=os.path.join(s_path,'train')
            s_test_path=os.path.join(s_path,'test')
            s_mpath=os.path.join(m_path,scheme[mode-1])
            if not os.path.exists(s_mpath):
                os.makedirs(s_mpath)
            m_train_path = os.path.join(s_mpath, 'train')
            m_test_path = os.path.join(s_mpath, 'test')
            file_path=os.path.join(origin_path,f)
            if not os.path.exists(m_train_path):
                os.makedirs(m_train_path)
            if not os.path.exists(m_test_path):
                os.makedirs(m_test_path)
            if t>train_nums:
                new_file_path=os.path.join(m_test_path,f)
            else:
                new_file_path=os.path.join(m_train_path,f)
            shutil.copy(file_path,new_file_path)


import functools
import os

import numpy as np
import pandas as pd
import random

class FileToTxt:
    def __init__(self):
        self.file_finish = False
        self.files=[]
        self.files_len=0
        self.df=None
        self.nums=0
        self.bwidth=120
        self.bheight=120
        self.state_space=60
        self.current_file=0
        self.cfile_len=0
        self.trans_nums=0
        self.trans_path=''


    def load_files(self,files_path,trans_path):
        self.files=files_path
        self.files_len=len(files_path)
        self.trans_path=trans_path
        self.current_file=0

    def load_df(self):
        self.df=pd.read_csv(self.files[self.current_file],header=None)
        self.cfile_len=len(self.df)
        self.nums=0

    def act(self):
        state,goal=self.get_state()
        flag=self.file_finish
        isend=False
        if flag:
            self.current_file+=1
            if self.current_file>=self.files_len:
                isend=True
            else:
                self.load_df()
            self.file_finish=False
        return flag,isend,state,goal


    def get_state(self):
        state = []
        # print(1,len(self.df), self.nums, self.current_file, len(self.train_files),self.mode)
        # print(2,len(self.df), self.nums, self.train_files[self.current_file], self.current_file, len(self.train_files),self.mode)
        row = self.df.iloc[self.nums]
        last_state = [int(row[1] / self.bwidth + 0.5), int(row[2] / self.bheight + 0.5)]
        state.append(int(row[1] / self.bwidth + 0.5))
        state.append(int(row[2] / self.bheight + 0.5))
        goal = [int(row[3] / self.bwidth + 0.5), int(row[4] / self.bheight + 0.5)]
        begin_goal=[goal[0],goal[1]]

        while (self.nums < self.cfile_len) and (len(state) < self.state_space):
            row = self.df.iloc[self.nums]
            x, y = int(row[1] / self.bwidth + 0.5), int(row[2] / self.bheight + 0.5)
            if x != last_state[0] or y != last_state[1]:
                state.append(x)
                state.append(y)
                goal[0], goal[1] = int(row[3] / self.bwidth + 0.5), int(row[4] / self.bheight + 0.5)
            self.nums += 1
            last_state[0],last_state[1]=x,y
        if self.nums >= self.cfile_len:
            self.file_finish = True
        self.file_finish=(self.nums>=self.cfile_len)
        if len(state) < self.state_space:
            state.extend([0 for i in range(self.state_space - len(state))])
        # self.total_move+=np.sqrt((b_move[0]-goal[0])**2+(b_move[1]-goal[1])**2)
        begin_goal.extend(goal)
        return state, begin_goal

    def trans_file(self,points,goal,flag,points_=None,intrain=False):
        save_path=os.path.join(self.trans_path,str(int(self.trans_nums))+'.txt')
        with open(save_path,'w') as f:
            for point in points:
                f.write(str(point)+' ')
                # print(point)
            f.write('\n')
            flag = 1 if flag==True else 0
            f.write(str(goal[0])+' '+str(goal[1])+' '+str(goal[2])+' '+str(goal[3])+' '+str(flag))
            if intrain:
                f.write('\n')
                for point in points_:
                    f.write(str(point)+' ')
        f.close()
        self.trans_nums+=1



def get_file_paths(p=None):
    dirs=[]
    root_path=os.path.join('D:\\eyedata\\mixed',p)
    for dir in os.listdir(root_path):
        d=os.path.join(root_path,dir)
        dirs.append(d)
    return dirs

def pretrain_save_file():
    schemes=['s1','s2','s3','s4']
    modes=['train','test']
    for scheme in schemes:
        for mode in modes:
            _=os.path.join(scheme,mode)
            dirs=get_file_paths(_)
            ft=FileToTxt()
            p='D:\\eyedata\\mixed\\pretrain\\'
            p=os.path.join(p,_)
            if not os.path.exists(p):
                os.makedirs(p)
            ft.load_files(dirs,p)
            ft.load_df()
            while True:
                flag,isend,state,goal=ft.act()
                ft.trans_file(state,goal,flag)
                if isend:
                    break

def intrain_save_file():
    dirs = get_file_paths('train')


if __name__ == '__main__':
    pretrain_save_file()
    # intrain_save_file()
    # move_files()




