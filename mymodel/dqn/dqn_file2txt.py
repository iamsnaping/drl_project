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
        self.last_goal_list=[]
        self.goal=[0.,0.,0.]
        self.move_list=[]
        self.goal_nums=0
        self.next_state=[]
        self.total_num=0
        self.isend_total=0
        self.k1=0.
        self.k2=0


    def load_files(self,files_path,trans_path):
        self.files=files_path
        self.files_len=len(files_path)
        self.trans_path=trans_path
        self.origin_trans_path=trans_path
        self.current_file=0

    def load_df(self,idx=None):
        if idx is not None:
            self.current_file=idx
        if self.current_file==0:
            self.trans_path=os.path.join(self.origin_trans_path,'train')
            if not os.path.exists(self.trans_path):
                os.makedirs(self.trans_path)
        # if self.current_file>self.files_len*0.7:
        #     self.nums=0
        #     self.goal_nums=0
        #     self.trans_path=os.path.join(self.origin_trans_path,'test')
        #     if not os.path.exists(self.trans_path):
        #         os.makedirs(self.trans_path)
        # print(f'current files {self.files[self.current_file]}')
        self.df=pd.read_csv(self.files[self.current_file],header=None)
        self.cfile_len=len(self.df)
        self.nums=0
        self.goal_nums=0
        self.last_goal=[0.,0.,0.]
        self.goal=[0.,0.,0.]

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
            self.goal=[int(row[3]/120+0.5),int(row[4]/120+0.5),0]
            self.goal_nums+=1
        # print(f'goal nums {self.goal_nums}')
    
    def get_states(self):
        self.get_goal()
        self.last_goal_list.append(self.goal[0])
        self.last_goal_list.append(self.goal[1])
        self.last_goal_list.append(self.goal[2])
        self.get_goal()
        self.last_goal_list.append(self.goal[0])
        self.last_goal_list.append(self.goal[1])
        self.last_goal_list.append(self.goal[2])
        self.get_goal()
        self.last_goal_list.append(self.goal[0])
        self.last_goal_list.append(self.goal[1])
        self.last_goal_list.append(self.goal[2])
        begin_idx=self.goal_nums
        flag=True
        self.move_list.clear()
        self.move_list=[]

        while flag:
            self.get_goal()
            end_idx=self.goal_nums
            x,y=-1,-1
            last_goal=self.last_goal_list[-9:].copy()
            last_state=[]
            next_state=[]
            self.move_list=[]
            # self.k1=0
            self.k2+=1
            for i in range(begin_idx,end_idx):
                row=self.df.iloc[i]
                last_state=next_state.copy()
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
                    increase_=True
                else:
                    x1,y1=int(row[1]/120+0.5),int(row[2]/120+0.5)
                    if x!=x1 or y!=y1:
                        x=x1
                        y=y1
                        self.move_list.append([x,y])
                        increase_=True
                        self.k1+=1
                next_state=self.get_s()
                isend=0
                if (i==end_idx-1) and (end_idx>=self.cfile_len-5):
                    isend=1
                if isend or (increase_ and len(last_state)!=0):
                    if len(last_state)!=0:
                        self.isend_total+=isend
                        trans_file(self.trans_path,self.total_num,last_state,next_state,last_goal,self.goal,[isend])
                        self.total_num+=1
                        # print(isend,i,end_idx,self.cfile_len,self.move_list,self.files[self.current_file])
            if end_idx>=self.cfile_len-5:
                break
            begin_idx=end_idx
            self.last_goal_list.append(self.goal[0])
            self.last_goal_list.append(self.goal[1])
            self.last_goal_list.append(self.goal[2])
            # print(end_idx,self.cfile_len)
            if end_idx>=self.cfile_len:
                flag=False


    def get_s(self):
        m_list=[]
        # if len(self.move_list)>=3:
        m_list.append(self.move_list[0][0])
        m_list.append(self.move_list[0][1])
        m_list.append(self.move_list[len(self.move_list)//2][0])
        m_list.append(self.move_list[len(self.move_list)//2][1])
        m_list.append(self.move_list[-1][0])
        m_list.append(self.move_list[-1][1])
        # else:
        return m_list
        if len(self.move_list)>=5:
            m_list.append(self.move_list[0][0])
            m_list.append(self.move_list[0][1])
            for i in range(3):
                m_list.append(self.move_list[len(self.move_list)//4*(i+1)][0])
                m_list.append(self.move_list[len(self.move_list)//4*(i+1)][1])
            m_list.append(self.move_list[-1][0])
            m_list.append(self.move_list[-1][1])
        else:
            put_list=[(5//len(self.move_list)) for i in range(len(self.move_list))]
            put_list[len(put_list)//2]+=5-sum(put_list)
            j=0
            for pt in put_list:
                for i in range(pt):
                    m_list.append(self.move_list[j][0])
                    m_list.append(self.move_list[j][1])
                j+=1
        return m_list

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
    for scheme in schemes:
        # print(f'schemes {scheme}')
        dirs=get_file_paths(scheme)
        random.shuffle(dirs)
        ft=FileToTxt()
        p='/home/wu_tian_ci/eyedata/mixed/pretrain2/'
        p=os.path.join(p,scheme)
        if not os.path.exists(p):
            os.makedirs(p)
        ft.load_files(dirs,p)
        file_len=ft.files_len
        train_=int(file_len*0.7)
        ft.trans_path=os.path.join(p,'train')
        ft.isend_total=0
        for i in tqdm(range(ft.files_len)):
            ft.load_df(i)
            ft.get_states()
            # print(ft.total_num)
            # if i>10:
            #     breakpoint()
            if i>=train_:
                ft.total_num=0
                ft.trans_path=os.path.join(p,'test')
        print(ft.isend_total,ft.files_len,ft.k1,ft.k2,ft.k1/ft.k2)
            
        # while True:
        #     flag,isend,state,next_state,last_goal,goal=ft.act()
        #     flag=0 if flag==False else 1
        #     ft.trans_file(state,next_state,last_goal,goal,[flag])
        #     if isend:
        #         break
        #     if ft.total_num>=10:
        #         breakpoint()

def trans_file(trans_path,num,*points_list):
    if not os.path.exists(trans_path):
        os.makedirs(trans_path)
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



def get_all_info(root_path):
    # dir_list=os.listdir(root_path)
    # f=open('/home/wu_tian_ci/drl_project/mymodel/dqn/trans.txt','w')
    # f.close()
    # for dir in dir_list:
    #     path_=os.path.join(root_path,dir)
    #     df=pd.read_csv(path_,header=None)
    #     for i in range(len(df)):
    #         row=df.iloc[i]
    #         if np.isnan(row[5]) or np.isnan(row[1]) or np.isnan(row[2]) or np.isnan(row[3]) or np.isnan(row[4]):
    #             continue  
    #         print(int(row[1]/120+0.5),int(row[2]/120+0.5),int(row[3]/120+0.5),int(row[4]/120+0.5),int(row[5]))

    files_par=os.listdir(root_path)
    total_len=0.
    for fp in files_par:
        f_p=os.path.join(root_path,fp)
        df=pd.read_csv(f_p,header=None)
        total_len+=len(df)
    print(total_len)

      

if __name__ == '__main__':
    pretrain_save_file()
    # get_all_info('/home/wu_tian_ci/eyedata/mixed/rl/s1')





