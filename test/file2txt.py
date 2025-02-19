import os
import random
import shutil

import pandas as pd



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
from tqdm import tqdm

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
        self.last_goal=[0,0,0]
        self.move_list=[]


    def load_files(self,files_path,trans_path):
        self.files=files_path
        self.files_len=len(files_path)
        self.trans_path=trans_path
        self.origin_trans_path=trans_path
        self.current_file=0

    def load_df(self):
        if self.current_file==0:
            self.trans_path=os.path.join(self.origin_trans_path,'train')
            if not os.path.exists(self.trans_path):
                os.makedirs(self.trans_path)
        if self.current_file>self.files_len*0.7:
            self.nums=0
            self.trans_path=os.path.join(self.origin_trans_path,'test')
            if not os.path.exists(self.trans_path):
                os.makedirs(self.trans_path)
        print(f'current files {self.files[self.current_file]}')
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

    def act2(self):
        state,goal=self.get_state2()
        flag=self.file_finish
        isend=False
        if flag:
            self.current_file+=1
            if self.current_file>=self.files_len:
                isend=True
            else:
                self.load_df()
            self.file_finish=False
        return [int(flag)],isend,state,goal

    def act3(self):
        state,begin_goal,goal=self.get_state3()
        flag=self.file_finish
        isend=False
        self.move_list.clear()
        if flag:
            # print(self.current_file,self.files_len,self.files[self.current_file])
            self.current_file+=1
            # print(f'next files {self.current_file} {self.files_len} {self.files[self.current_file]}')
            if self.current_file>=self.files_len:
                isend=True
            else:
                self.last_goal=[0.,0.,0.]
                self.move_list.clear()
                self.load_df()
            self.file_finish=False
        return flag,isend,state,begin_goal,goal


    def get_state2(self):
        state = []
        goal_list=[0 for i in range(45)]
        # print(1,len(self.df), self.nums, self.current_file, len(self.train_files),self.mode)
        # print(2,len(self.df), self.nums, self.train_files[self.current_file], self.current_file, len(self.train_files),self.mode)
        row = self.df.iloc[self.nums]
        last_state = [int(row[1] / self.bwidth + 0.5), int(row[2] / self.bheight + 0.5)]
        state.append(int(row[1] / self.bwidth + 0.5))
        state.append(int(row[2] / self.bheight + 0.5))
        # goal = [int(row[3] / self.bwidth + 0.5), int(row[4] / self.bheight + 0.5)]
        goal = [int(row[3]), int(row[4]),int(row[5])]
        begin_goal=[]
        goal_list.pop(0)
        goal_list.pop(0)
        goal_list.pop(0)
        goal_list.append(int(row[3]))
        goal_list.append(int(row[4]))
        goal_list.append(int(row[5]))
        while (self.nums < self.cfile_len) and (len(state) < self.state_space):
            row = self.df.iloc[self.nums]
            x, y = int(row[1] / self.bwidth + 0.5), int(row[2] / self.bheight + 0.5)
            if x != last_state[0] or y != last_state[1]:
                state.append(x)
                state.append(y)
                # goal[0], goal[1] = int(row[3] / self.bwidth + 0.5), int(row[4] / self.bheight + 0.5)
                goal[0], goal[1],goal[2] = int(row[3]), int(row[4]),int(row[5])
                goal_list.pop(0)
                goal_list.pop(0)
                goal_list.pop(0)
                goal_list.append(int(row[3]))
                goal_list.append(int(row[4]))
                goal_list.append(int(row[5]))
                if len(state)==30:
                    begin_goal=goal_list.copy()
            self.nums += 1
            last_state[0],last_state[1]=x,y
        if self.nums >= self.cfile_len:
            self.file_finish = True
        self.file_finish=(self.nums>=self.cfile_len)
        if len(state) < self.state_space:
            state.extend([0 for i in range(self.state_space - len(state))])
        if len(begin_goal)<10:
            begin_goal=goal_list.copy()
        # self.total_move+=np.sqrt((b_move[0]-goal[0])**2+(b_move[1]-goal[1])**2)
        return state, begin_goal,goal

    def get_state3(self):
        move_=[0,0,0]
        row = self.df.iloc[self.nums]
        x, y = int(row[1] / self.bwidth + 0.5), int(row[2] / self.bheight + 0.5)
        self.move_list.append([x,y])
        while (self.nums < self.cfile_len):
            row = self.df.iloc[self.nums]
            print(row)
            if len(row)<5:
                self.nums+=1
                continue
            if np.isnan(row[5]) or np.isnan(row[1]) or np.isnan(row[2]) or np.isnan(row[3]) or np.isnan(row[4]):
                self.nums+=1
                continue
            # print(row,np.isnan(row[]))
            x, y = int(row[1] / self.bwidth + 0.5), int(row[2] / self.bheight + 0.5)
            if x != self.move_list[-1][0] or y != self.move_list[-1][1]:
                self.move_list.append([x,y])
            flag=int(row[5])
            if flag==0:
                move_=self.last_goal.copy()
                self.last_goal=[int(row[3]/self.bwidth+0.5),int(row[4]/self.bheight+0.5),int(row[5])]
                break
            self.nums+=1
        if self.nums<self.cfile_len:
            row=self.df.iloc[self.nums]
            flag=int(row[5])
            if flag==0:
                while flag==0 and self.nums<self.cfile_len:
                    row = self.df.iloc[self.nums]
                    flag = int(row[5])
                    self.nums+=1
                self.nums += 1
        #extend
        m_list=[]
        if len(self.move_list)>=5:
            m_list.append(self.move_list[0][0])
            m_list.append(self.move_list[0][1])
            for i in range(3):
                m_list.append(self.move_list[len(self.move_list)//4*(i+1)][0])
                m_list.append(self.move_list[len(self.move_list)//4*(i+1)][1])
            # m_list.append(self.move_list[len(self.move_list)//2][0])
            # m_list.append(self.move_list[len(self.move_list)//2][1])
            m_list.append(self.move_list[-1][0])
            m_list.append(self.move_list[-1][1])
        else:
            put_list=[(5//len(self.move_list)) for i in range(len(self.move_list))]
            put_list[len(put_list)//2]+=5-sum(put_list)
            # m_list.append(self.move_list[len(self.move_list) // 2][0])
            # m_list.append(self.move_list[len(self.move_list) // 2][1])
            j=0
            for pt in put_list:
                for i in range(pt):
                    m_list.append(self.move_list[j][0])
                    m_list.append(self.move_list[j][1])
                j+=1
            # print(sum(put_list),put_list)
            # m_list.append(self.move_list[-1][0])
            # m_list.append(self.move_list[-1][1])
        if self.nums >= self.cfile_len:
            self.file_finish = True
        self.file_finish=(self.nums>=self.cfile_len)

        #move_ : last_click self.last_goal:current click
        if len(m_list)>10:
            # print(m_list)
            breakpoint()
        return m_list,move_,self.last_goal



    def get_state(self):
        state = []
        # print(1,len(self.df), self.nums, self.current_file, len(self.train_files),self.mode)
        # print(2,len(self.df), self.nums, self.train_files[self.current_file], self.current_file, len(self.train_files),self.mode)
        row = self.df.iloc[self.nums]
        last_state = [int(row[1] / self.bwidth + 0.5), int(row[2] / self.bheight + 0.5)]
        state.append(int(row[1] / self.bwidth + 0.5))
        state.append(int(row[2] / self.bheight + 0.5))
        # goal = [int(row[3] / self.bwidth + 0.5), int(row[4] / self.bheight + 0.5)]
        goal = [int(row[3]), int(row[4])]
        begin_goal=[goal[0],goal[1]]

        while (self.nums < self.cfile_len) and (len(state) < self.state_space):
            row = self.df.iloc[self.nums]
            x, y = int(row[1] / self.bwidth + 0.5), int(row[2] / self.bheight + 0.5)
            if x != last_state[0] or y != last_state[1]:
                state.append(x)
                state.append(y)
                # goal[0], goal[1] = int(row[3] / self.bwidth + 0.5), int(row[4] / self.bheight + 0.5)
                goal[0], goal[1] = int(row[3] ), int(row[4])
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

    def trans_file(self,*points_list):
        save_path=os.path.join(self.trans_path,str(int(self.trans_nums))+'.txt')
        with open(save_path,'w') as f:
            f_1=False
            f_2=False
            for points in points_list:
                if f_1==False:
                    f_1=True
                else:
                    f.write('\n')
                f_2=False
                for point in points:
                    if f_2==False:
                        f_2=True
                    else:
                        f.write(' ')
                    f.write(str(int(point)))

        #     for point in points:
        #         f.write(str(point)+' ')
        #         # print(point)
        #     f.write('\n')
        #     flag = 1 if flag==True else 0
        #     ll=len(last_goal)
        #     for i in range(ll):
        #
        #     for s in goal:
        #         f.write(str(s)+' ')
        #     f.write(str(flag))
        #     if intrain:
        #         f.write('\n')
        #         for point in points_:
        #             f.write(str(point)+' ')
        # f.close()
        self.trans_nums+=1



def get_file_paths(p=None):
    dirs=[]
    root_path=os.path.join('/home/wu_tian_ci/eyedata/mixed/rl',p)
    for dir in os.listdir(root_path):
        d=os.path.join(root_path,dir)
        dirs.append(d)
    return dirs

def pretrain_save_file():
    schemes=['s1','s2','s3','s4']
    for scheme in tqdm(schemes):
        print(f'schemes {scheme}')
        dirs=get_file_paths(scheme)
        random.shuffle(dirs)
        # print(dirs)
        ft=FileToTxt()
        p='/home/wu_tian_ci/eyedata/mixed/pretrain14/'
        p=os.path.join(p,scheme)
        if not os.path.exists(p):
            os.makedirs(p)
        ft.load_files(dirs,p)
        ft.load_df()
        while True:
            flag,isend,state,last_goal,goal=ft.act3()
            flag=0 if flag==False else 1
            ft.trans_file(state,last_goal,goal,[flag])
            if isend:
                break





def intrain_save_file():
    dirs = get_file_paths('train')
    schemes = ['s1', 's2', 's3', 's4']
    modes = ['train', 'test']





def trans_file(trans_path,num,*points_list):
    save_path=os.path.join(trans_path,str(int(num))+'.txt')
    with open(save_path,'w') as f:
        f_1=False
        f_2=False
        for points in points_list:
            if f_1==False:
                f_1=True
            else:
                f.write('\n')
            f_2=False
            for point in points:
                if f_2==False:
                    f_2=True
                else:
                    f.write(' ')
                f.write(str(int(point)))

def get_mixed_data(save_path,paths,schemes):
    nums=0
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for path,scheme in tqdm(zip(paths,schemes)):
        df = pd.read_csv(path, header=None)
        cfile_len = len(df)
        x_c_list=[0 for i in range(135)]
        y_c_list=[0 for i in range(135)]
        s_c_list=[0 for i in range(135)]
        x_e_list=[0 for i in range(45)]
        y_e_list=[0 for i in range(45)]
        # x_c_list=[]
        # y_c_list=[]
        # s_c_list=[]
        # x_e_list=[]
        # y_e_list=[]
        for i in range(cfile_len):
            row=df.iloc[i]
            x_e_list.append(int(row[1]/120+0.5))
            y_e_list.append(int(row[2]/120+0.5))
            x_c_list.append(int(row[3]/120+0.5))
            y_c_list.append(int(row[4] / 120 + 0.5))
            s_c_list.append(int(row[5] / 120 + 0.5))
            flag=1 if i==cfile_len-1 else 0
            # if len(x_c_list)>=135:
            eye_movement = x_e_list[-45:]+y_e_list[-45:]
            click_past=x_c_list[-135:-15]+y_c_list[-135:-15]
            click_past_s=s_c_list[-135:-15]
            click_=x_c_list[-15:]+y_c_list[-15:]
            click_s=s_c_list[-15:]
            trans_file(save_path,nums,eye_movement,click_past,click_past_s,click_,click_s,[scheme for i in range(120)],[flag])
            nums+=1


def save_file_to():
    p = '/home/wu_tian_ci/eyedata/mixed/pretrain9'
    schemes = ['s1', 's2', 's3', 's4']
    modes = ['train', 'test']
    train_list,train_schemes=[],[]
    test_list,test_schemes=[],[]
    for scheme in schemes:
        for mode in modes:
            _ = os.path.join(scheme, mode)
            files = get_file_paths(_)
            # print(files)
            if mode=='train':
                train_list.extend(files)
                train_schemes.extend([int(scheme[-1]) for i in range(len(files))])
            else:
                test_list.extend(files)
                test_schemes.extend([int(scheme[-1]) for i in range(len(files))])
    # print(train_list)
    # print(test_list)
    get_mixed_data(os.path.join(p,'train'),train_list,train_schemes)
    get_mixed_data(os.path.join(p,'test'),test_list,test_schemes)


def move_files_mixed_rl(rl_root_path):
    for origin in origin_list:
        if len(origin)>=5:
            continue
        # print(origin,len(origin))
        origin_path=os.path.join(root_path,origin)
        file_list=os.listdir(origin_path)
        f_len=len(file_list)
        # random.shuffle(file_list)
        t=0
        for f in file_list:
            if f=='record.csv':
                continue
            t+=1
            if len(f)<5:
                continue
            mode=int(f[2])
            s_mpath=os.path.join(rl_root_path,scheme[mode-1])
            if not os.path.exists(s_mpath):
                os.makedirs(s_mpath)
            file_path=os.path.join(origin_path,f)
            shutil.copy(file_path,s_mpath)



def move_files_to_seperate(root_path,new_path):
    people_n=[(i+1) for i in range(1,21)]
    for p_n in people_n:
        p_f_n=str(p_n)
        if p_n<10:
            p_f_n='0'+p_f_n
        file_path=os.path.join(root_path,p_f_n)
        n_f_path=os.path.join(new_path,p_f_n)
        files_list=os.listdir(file_path)
        for file_n in files_list:
            f_path=os.path.join(file_path,file_n)
            if file_n=='record.csv':
                continue
            mode=int(file_n[2])
            nfp_=os.path.join(n_f_path,file_n[2])
            if not os.path.exists(nfp_):
                os.makedirs(nfp_)
            nfp=os.path.join(nfp_,file_n)
            shutil.copy(f_path,nfp)


def move_files_to_mixed(root_path,new_path):
    people_n=[(i+1) for i in range(1,21)]
    for p_n in people_n:
        p_f_n=str(p_n)
        if p_n<10:
            p_f_n='0'+p_f_n
        file_path=os.path.join(root_path,p_f_n)
        n_f_path=os.path.join(new_path,p_f_n)
        files_list=os.listdir(file_path)
        for file_n in files_list:
            f_path=os.path.join(file_path,file_n)
            if file_n=='record.csv':
                continue
            mode=int(file_n[2])
            nfp_=os.path.join(n_f_path,file_n[2])
            if not os.path.exists(nfp_):
                os.makedirs(nfp_)
            nfp=os.path.join(nfp_,file_n)
            shutil.copy(f_path,nfp)




def get_max_min():
    root_path='/home/wu_tian_ci/eyedata/mixed/rl'
    p_dirs=['s1','s2','s3','s4']
    max_x,max_y,min_x,min_y=0,0,np.inf,np.inf
    for dirs in p_dirs:
        r_p=os.path.join(root_path,dirs)
        files=os.listdir(r_p)
        for f in files:
            f_p=os.path.join(r_p,f)
            csv_file=pd.read_csv(f_p,header=None)
            csv_len=len(csv_file)
            for i in range(csv_len):
                row=csv_file.iloc[i]
                if np.isnan(row[5]) or np.isnan(row[1]) or np.isnan(row[2]) or np.isnan(row[3]) or np.isnan(row[4]):
                    continue
                x,y=int(row[3]/120+0.5),int(row[4]/120+0.5)
                max_x=max(x,max_x)
                max_y=max(max_y,y)
                min_x=min(x,min_x)
                min_y=min(y,min_y)
    print(max_x,max_y,min_x,min_y)
                

if __name__ == '__main__':
    # move_files_to_mixed('/home/wu_tian_ci/eyedata','/home/wu_tian_ci/eyedata/mixed')
    # move_files_mixed_rl()
    # save_file_to()
    # pretrain_save_file()
    # intrain_save_file()
    # move_files()
    # move_files_mixed_rl('/home/wu_tian_ci/eyedata/mixed/rl')
    move_files_to_seperate('/home/wu_tian_ci/eyedata','/home/wu_tian_ci/eyedata/seperate')
    # get_max_min()




