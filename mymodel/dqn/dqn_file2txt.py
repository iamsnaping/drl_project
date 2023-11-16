import os
import random
import shutil
import sys
sys.path.append('/home/wu_tian_ci/drl_project')
sys.path.append('/home/wu_tian_ci/drl_project/wtcutils')
import pandas as pd
import json
from dqnutils import *
import dqnutils as UTIL


'''
seperate/train\test/
miexd/train\test/

'''
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
        self.goal=0.
        self.move_list=[]
        self.goal_nums=0
        self.next_state=[]
        self.total_num=0
        self.isend_total=0
        self.k1=0.
        self.k2=0
        self.max_x=0.
        self.max_y=0.
        self.a_max_x=0.
        self.a_max_y=0.
        self.json_data={}
        self.eyeRegion=Region('eye')
        self.eyeRegion.setRegions(UTIL.EYEAREAS)
        self.clickRegion=Region('click')
        self.clickRegion.setRegions(UTIL.CLICKAREAS)
        self.trajectorys=DQNTrajectory()
        self.rewardFun=DQNReward()
        # self.next_state_list=[]


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
            self.trans_path=os.path.join(self.origin_trans_path,'train1.json')

        self.df=pd.read_csv(self.files[self.current_file],header=None)
        self.cfile_len=len(self.df)
        self.nums=0
        self.goal_nums=0
        self.last_goal=[0.,0.,0.]
        self.goal=0.

    def get_goal(self):
        flag=-1
        while self.goal_nums<self.cfile_len:
            row=self.df.iloc[self.goal_nums]
            if np.isnan(row[5]) or np.isnan(row[1]) or np.isnan(row[2]) or np.isnan(row[3]) or np.isnan(row[4]):
                self.goal_nums+=1
                continue  
            row=row.tolist()
            flag=row[5]
            if flag==0:
                break
            self.goal_nums+=1
        while flag ==0 and self.goal_nums<self.cfile_len:
            row=self.df.iloc[self.goal_nums]
            if np.isnan(row[5]) or np.isnan(row[1]) or np.isnan(row[2]) or np.isnan(row[3]) or np.isnan(row[4]):
                break
            row=row.tolist()
            flag=int(row[5])
            if flag !=0:
                break
            self.goal=self.clickRegion.judge(row[3],row[4])
            self.goal_nums+=1
    
    def get_states4critic(self):
        self.get_goal()
        self.last_goal_list.append(self.goal)
        self.get_goal()
        self.last_goal_list.append(self.goal)
        self.get_goal()
        self.last_goal_list.append(self.goal)
        begin_idx=self.goal_nums
        flag=True
        self.move_list.clear()
        self.move_list=[]
        move_list=[]
        k=0
        while flag:
            self.trajectorys.clear()
            self.get_goal()
            end_idx=self.goal_nums
            regionEye=-1
            states=[]
            self.move_list=[]
            move_list=[]
            self.trajectorys.clear()
            for i in range(begin_idx,end_idx):
                row=self.df.iloc[i]
                increase_=False
                if len(row)<5:
                    self.nums+=1
                    continue
                if np.isnan(row[5]) or np.isnan(row[1]) or np.isnan(row[2]) or np.isnan(row[3]) or np.isnan(row[4]):
                    self.nums+=1
                    continue
                row=row.tolist()
                if regionEye==-1:
                    regionEye=self.eyeRegion.judge(row[1],row[2])
                    if regionEye==None:
                        continue
                    self.move_list.append(regionEye)
                    increase_=True
                else:
                    regionSub=self.eyeRegion.judge(row[1],row[2])
                    if regionSub==None:
                        print(f'row1 {row[1]} row2 {row[2]}')
                        continue
                    if regionSub!=regionEye:
                        regionEye=regionSub
                        self.move_list.append(regionSub)
                        increase_=True
                        self.k1+=1
                states=self.get_s()
                if increase_:
                    move_list.append(states)
                # if (i==end_idx-1) and (end_idx>=self.cfile_len-5):
                #     isend=1
            move_list_len=len(move_list)
            if move_list_len>=0:
                for i in range(move_list_len):
                    flag=0
                    if i==move_list_len-1:
                        flag=1
                    eye=move_list[i]
                    click=self.last_goal_list[-3:]
                    goal=self.goal
                    if i<move_list_len*0.8:
                        action=12
                    else:
                        action=goal
                    self.trajectorys.push(click,eye,goal,action,UTIL.GAMMA*flag)
                self.trajectorys.getNewTras()
                TDZero=self.trajectorys.getComTraZero()
                # print(len(TDZero))
                for i in range(1,5):
                    TDone=self.trajectorys.getComTraN(N=i)
                    TDZero.extend(TDone)

                # print(f'tdone {len(TDone)}')
                for td in TDZero:
                    click,eye,goal,action,seq,mask,reward,nclick,neye,nseq=td
                    self.trans_file(click,eye,goal,action,seq,mask,reward,nclick,neye,nseq)
                    self.total_num+=1

            if end_idx>=self.cfile_len-5:
                break
            begin_idx=end_idx
            self.last_goal_list.append(self.goal)
            if end_idx>=self.cfile_len:
                flag=False

    def get_s(self):
        m_list=[]

        if len(self.move_list)>=3:
            m_list=self.move_list[-3:]
        else:
            m_list.append(self.move_list[0])
            m_list.append(self.move_list[0])
            m_list.append(self.move_list[-1])
        # near_m_list=[]
        # if len(self.move_list)>=10:
        #     m_list.append(self.move_list[0][0])
        #     for i in range(8):
        #         m_list.append(self.move_list[len(self.move_list)//9*(i+1)][0])
        #         m_list.append(self.move_list[len(self.move_list)//9*(i+1)][1])
        #     m_list.append(self.move_list[-1][0])
        #     m_list.append(self.move_list[-1][1])
        # else:
        #     put_list=[(10//len(self.move_list)) for i in range(len(self.move_list))]
        #     k=(10-sum(put_list))//len(put_list)
        #     for i in range(int(len(put_list)//2)):
        #         put_list[i]+=k
        #         put_list[len(put_list)-i-1]+=k
        #     put_list[int(len(put_list)/2+0.5)-1]+=10-sum(put_list)
        #     # put_list[(len(put_list)-1)//2]+=10-sum(put_list)
        #     j=0
        #     for pt in put_list:
        #         for i in range(pt):
        #             m_list.append(self.move_list[j][0])
        #             m_list.append(self.move_list[j][1])
        #         j+=1
        # near_=self.move_list[-(max(len(self.move_list)//3,1)):]
        # if len(near_)>=5:
        #     near_m_list.append(near_[0][0])
        #     near_m_list.append(near_[0][1])
        #     for i in range(3):
        #         near_m_list.append(near_[len(near_)//4*(i+1)][0])
        #         near_m_list.append(near_[len(near_)//4*(i+1)][1])
        #     near_m_list.append(near_[-1][0])
        #     near_m_list.append(near_[-1][1])
        # else:
        #     put_list=[(5//len(near_)) for i in range(len(near_))]
        #     k=(5-sum(put_list))//len(put_list)
        #     for i in range(int(len(put_list)//2)):
        #         put_list[i]+=k
        #         put_list[len(put_list)-i-1]+=k
        #     put_list[int(len(put_list)/2+0.5)-1]+=5-sum(put_list)
        #     # put_list[(len(put_list)-1)//2]+=10-sum(put_list)
 
        #     j=0
        #     for pt in put_list:
        #         for i in range(pt):
        #             near_m_list.append(near_[j][0])
        #             near_m_list.append(near_[j][1])
        #         j+=1
        return m_list

    def trans_file(self,*points_list):
        # if not os.path.exists(self.trans_path):
        #     os.makedirs(self.trans_path)
        save_path=os.path.join(self.trans_path,str(int(self.trans_nums))+'.txt')
        d={}
        i=0
        for points in points_list:
            d[str(i)]=points
            i+=1
        data={}
        data[str(self.trans_nums)]=d
        self.json_data.update(data)
        self.trans_nums+=1



def get_file_paths(p=None):
    dirs=[]
    root_path=os.path.join('/home/wu_tian_ci/eyedata/mixed/rl',p)
    for dir in os.listdir(root_path):
        d=os.path.join(root_path,dir)
        dirs.append(d)
    return dirs

# mode ,the length of eye moving  
def pretrain_save_file(name,comple):
    # schemes=['s1','s2','s3','s4']
    schemes=['s1']
    for scheme in schemes:
        dirs=get_file_paths(scheme)
        random.shuffle(dirs)
        ft=FileToTxt()

        p=os.path.join('/home/wu_tian_ci/eyedata/dqn',name)
        # p=os.path.join()
        p=os.path.join(p,scheme)
        if not os.path.exists(p):
            os.makedirs(p)
        with open(os.path.join(p,'info.txt'),'w') as f:
            f.write(comple)
        if not os.path.exists(p):
            os.makedirs(p)
        ft.load_files(dirs,p)
        file_len=ft.files_len
        train_=int(file_len*0.7)
        ft.trans_path=os.path.join(p,'train1.json')
        ft.isend_total=0
        store_flag=True
        for i in tqdm(range(ft.files_len)):
            ft.load_df(i)
            ft.get_states4critic()
            if i>=train_ and store_flag:
                f=open(ft.trans_path,'w')
                json.dump(ft.json_data,f,indent=2)
                f.close()
                ft.trans_nums=0
                ft.total_num=0
                ft.json_data.clear()
                ft.trans_path=os.path.join(p,'test1.json')
                store_flag=False
        f=open(ft.trans_path,'w')
        json.dump(ft.json_data,f,indent=2)
        f.close()
        # breakpoint()
        ft.json_data.clear()
        ft.trans_nums=0
        ft.total_num=0

def trans_file(trans_path,num,*points_list):
    if not os.path.exists(trans_path):
        print(trans_path)
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
    
    # for i,j in zip(x,y):
    #     t+=1
    #     if t>m:
    #         k+=1
    #         m+=section[k]*section[k]
    #     print(f'x:{i} y:{j} begin_r:{radium[k]} {radium[k+1]} r{np.sqrt(i**2+j**2)}')

    # pretrain_save_file('3_critic',wtuc.CRITIC,3)
    # pretrain_save_file('5_critic',wtuc.CRITIC,5)

    # pretrain_save_file('200pixels',wtuc.POLICY,5)
    
    # pretrain_save_file('5_critic_1last_move_not_origin',wtuc.CRITIC,5)
    # pretrain_save_file('direct_goal',wtuc.POLICY_,5)
    # pretrain_save_file('5_critic_1last_move',wtuc.CRITIC,5)
    # pretrain_save_file('5_policy_1last_move_',wtuc.POLICY_,5)
    # pretrain_save_file('5_critic_1last_move_',wtuc.CRITIC_,5)
 
    # pretrain_save_file('3_policy',wtuc.POLICY,3)
    pretrain_save_file('1',comple='action:len3 eye:len3')