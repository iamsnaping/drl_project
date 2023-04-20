import functools
import os

import numpy as np
import pandas as pd
import random


class EyeEnvironment:
    def __init__(self, verify_mode='random', verify_num=0.3):
        self.p='s4'
        self.root_path = os.path.join('D:\\eyedata\\seperate\\',self.p)
        self.dirs_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15',
                          '16', '17', '18', '19', '20', '21']
        # self.dirs_list = ['01', '02', '03', '04', '05']
        self.bwidth = 120
        self.bheight = 120
        self.nums = 0
        self.file_len = 0.
        self.train_files = []
        self.test_files = []

        self.trans_files=[]
        self.trans_nums=0.

        self.verify_mode = verify_mode
        self.verify_num = verify_num
        self.finish = False
        self.current_file = -1
        self.mode = 'train'
        self.current_dir_len = 0
        self.file_finish = False
        self.state_space=60
        self.action_space=2

        self.load_dict2()
        self.total_move=0.
        self.total_steps=0

    def load_dict(self):
        for dir in self.dirs_list:
            dir_path = os.path.join(self.root_path, dir)
            dir_list = os.listdir(dir_path)
            if 'record.csv' in dir_list:
                dir_list.pop(0)
            train_num = int(len(dir_list) * self.verify_num)
            if self.verify_mode == 'random':
                random.shuffle(dir_list)
            for i in range(len(dir_list)):
                f_path = os.path.join(dir_path, dir_list[i])
                if i < len(dir_list) - train_num:
                    self.train_files.append(f_path)
                else:
                    self.test_files.append(f_path)
        if self.mode == 'train':
            self.current_dir_len = len(self.train_files)
        elif self.mode == 'test':
            self.current_dir_len = len(self.test_files)
        random.shuffle(self.train_files)

    def load_dict2(self):
        root_path = os.path.join('D:\\eyedata\\mixed\\pretrain',self.p)
        for dir in self.dirs_list:
            new_path=os.path.join(root_path,dir)
            dir_path = os.path.join(self.root_path, dir)
            dir_list = os.listdir(dir_path)
            if 'record.csv' in dir_list:
                dir_list.pop(0)
            train_num = int(len(dir_list) * self.verify_num)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            for i in range(len(dir_list)):
                f_path = os.path.join(dir_path, dir_list[i])
                # new_f_path=os.path.join(new_path,dir_list[i])
                self.train_files.append(f_path)
                self.trans_files.append(new_path)

                # if i < len(dir_list) - train_num:
                #     self.train_files.append(f_path)
                # else:
                #     self.test_files.append(f_path)


    def set_mode(self, mode):
        self.mode = mode
        self.finish = False
        self.current_file = -1
        self.file_finish = False
        if self.mode == 'train':
            self.current_dir_len = len(self.train_files)
        elif self.mode == 'test':
            self.current_dir_len = len(self.test_files)

    def get_file(self, appoint_path=None):
        if appoint_path is not None:
            self.df = pd.read_csv(appoint_path, header=None)
            return True
        # print(6,self.file_finish)
        self.current_file += 1

        # print(f'current file num {self.current_file} self.mode {self.mode}')
        if self.current_file >= self.current_dir_len:
            self.finish = True
            return False
        if self.mode == 'train':
            self.df = pd.read_csv(self.train_files[self.current_file], header=None)
        elif self.mode == 'test':
            self.df = pd.read_csv(self.test_files[self.current_file], header=None)
        self.file_len = len(self.df)
        return True

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
        b_move=[int(row[3] / self.bwidth + 0.5), int(row[4] / self.bheight + 0.5)]
        self.total_steps+=1
        while (self.nums < self.file_len) and (len(state) < self.state_space):
            row = self.df.iloc[self.nums]
            x, y = int(row[1] / self.bwidth + 0.5), int(row[2] / self.bheight + 0.5)
            if x != last_state[0] or y != last_state[1]:
                state.append(x)
                state.append(y)
                goal[0], goal[1] = int(row[3] / self.bwidth + 0.5), int(row[4] / self.bheight + 0.5)
            self.nums += 1
            last_state[0],last_state[1]=x,y
        if self.nums >= self.file_len:
            self.file_finish = True
        self.file_finish=self.nums>=self.file_len
        if len(state) < self.state_space:
            state.extend([0 for i in range(self.state_space - len(state))])
        self.total_move+=np.sqrt((b_move[0]-goal[0])**2+(b_move[1]-goal[1])**2)
        # print(3,self.file_finish, self.nums, self.file_len,len(self.df), self.nums, self.current_file, len(self.train_files),self.mode)
        begin_goal.extend(goal)
        return state, begin_goal

    def act(self):
        state = self.get_state()
        # print(4,self.file_len,self.file_finish)
        return self.file_finish, state

    def reset(self):
        self.nums = 0
        self.trans_nums=0
        flag = self.get_file()
        if flag == False:
            return flag, []
        self.file_finish = False
        # print(f'reset state')
        return flag, self.get_state()

    def trans_file(self,points,goal,flag,points_=None,intrain=False):
        # print(f'trainfiles len {len(self.train_files)}')
        # print(f'transfiles len {len(self.trans_files)}')
        # print(self.nums)
        save_path=os.path.join(self.trans_files[self.current_file],str(int(self.trans_nums))+'.txt')
        with open(save_path,'w') as f:
            for point in points:
                f.write(str(point)+' ')
                # print(point)
            f.write('\n')
            f.write(str(goal[0])+' '+str(goal[1])+' '+str(goal[2])+' '+str(goal[3])+' '+str(flag))
            if intrain:
                f.write('\n')
                for point in points_:
                    f.write(str(point)+' ')
        f.close()
        self.trans_nums+=1


def pretrain_save_file():
    env = EyeEnvironment()
    env.set_mode('train')
    flag, _ = env.reset()
    t = 0
    state = _[0]
    goal = _[1]
    while True:
        flag, _ = env.act()
        if not flag:
            env.trans_file(state, goal, 0)
        if flag == True:
            env.trans_file(state, goal, 1)
            flag, s = env.reset()
            if flag:
                state = s[0]
                goal = s[1]
                env.trans_file(state, goal, 0)
                continue
            if flag == False and env.mode == 'train':
                break
                env.set_mode('test')
                t += 1
            elif flag == False and env.mode == 'test':
                break
                env.set_mode('train')
    print(env.total_move / env.total_steps)

def intrain_save_file():
    env = EyeEnvironment()
    env.set_mode('train')
    flag, _ = env.reset()
    t=0
    state = _[0]
    goal = _[1]
    while True:
        flag, _ = env.act()
        state_=_[0]
        goal_=_[1]
        if not flag:
            env.trans_file(state,goal,0,state_,intrain=True)
        if flag == True:
            env.trans_file(state, goal, 1, state_,intrain=True)
            flag, s = env.reset()
            if flag:
                state = s[0]
                goal = s[1]
                continue
            if flag == False and env.mode == 'train':
                break
                env.set_mode('test')
                t+=1
            elif flag == False and env.mode == 'test':
                break
                env.set_mode('train')
        state=state_
        goal=goal_
    print(env.total_move/env.total_steps)

if __name__ == '__main__':
    pretrain_save_file()
    # intrain_save_file()
