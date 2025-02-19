import os
import random
import shutil
import sys
sys.path.append('/home/wu_tian_ci/drl_project')
sys.path.append('/home/wu_tian_ci/drl_project/wtcutils')
import pandas as pd
import json
import wtcutils.constant as wtuc

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
        print(origin_list)
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
        self.mode=3
        self.max_x=0.
        self.max_y=0.
        self.a_max_x=0.
        self.a_max_y=0.
        self.json_data={}
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
            # if not os.path.exists(self.trans_path):
            #     os.makedirs(self.trans_path)
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
            self.goal=row[3],row[4]
            self.goal_nums+=1
        # print(f'goal nums {self.goal_nums}')
    
    def get_states3policy(self):
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
        begin_idx=self.goal_nums
        flag=True
        # self.move_list.clear()
        self.move_list=[]
        while flag:
            states_list,near_state_list,last_goal_list,last_action_list,goal_list,is_end_list=[],[],[],[],[],[]
            self.get_goal()
            end_idx=self.goal_nums
            x,y=-1,-1
            xx,yy=-1,-1
            last_goal=self.last_goal_list[-6:].copy()
            states=[]
            near_state=[]
            self.move_list=[]
            # self.k1=0
            self.k2+=1
            last_action=[0.,0.]
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
                self.move_list.append([int(row[1]/120+0.5),int(row[2]/120+0.5)])
                # self.move_list.append()
                self.max_x=max(self.max_x,row[1])
                self.max_y=max(self.max_y,row[2])
                self.a_max_x=max(self.a_max_x,row[3])
                self.a_max_y=max(self.a_max_y,row[4])
                if x==-1 and y==-1:
                    x,y=int(row[1]/120+0.5),int(row[2]/120+0.5)
                    xx,yy=row[3],row[4]
                    self.move_list.append([x,y])
                    increase_=True
                else:
                    x1,y1=int(row[1]/120+0.5),int(row[2]/120+0.5)
                    self.move_list.append([x1,y1])
                    if x!=x1 or y!=y1:
                        x=x1
                        y=y1
                        last_action[0],last_action[1]=xx,yy
                        xx,yy=row[3],row[4]
                        increase_=True
                        self.k1+=1
                states,near_state=self.get_s()
                isend=0
                if (i==end_idx-1) and (end_idx>=self.cfile_len-5):
                    isend=1
                if isend or (increase_ and len(states)!=0):
                    if len(states)!=0:
                        self.isend_total+=isend
                        states_list.append(states)
                        near_state_list.append(near_state)
                        if len(last_goal)>6:
                            breakpoint()
                        last_goal_list.append(last_goal)
                        last_action_list.append(last_action)
                        goal_list.append(self.goal)
                        is_end_list.append([isend])
            list_len=len(states_list)
            if list_len==0:
                continue
            begin_=self.last_goal_list[-2:]
            p_x,p_y=self.get_gaussian_points(begin_,goal_list[-1],list_len*7)
            # print(len(p_x),list_len)
            points_idx=0
            upper_=int(list_len*0.2)
            if upper_<1:
                upper_=1
            for i in range(list_len):
                if i>upper_:
                    max_dis=200
                else:
                    max_dis=50
                a=states_list[i]
                b=near_state_list[i]
                c=last_goal_list[i]
                f=is_end_list[i]
                e=goal_list[i]
                aa,bb,cc,ee=[],[],[],[]
                for j in range(len(a)):
                    if j&1==0:
                        aa.append((a[j]-16)/16)
                    else:
                        aa.append((a[j]-4.5)/4.5)

                for j in range(len(b)):
                    if j&1==0:
                        bb.append((b[j]-16)/16)
                    else:
                        bb.append((b[j]-4.5)/4.5)

                for j in range(len(c)):
                    if j&1==0:
                        cc.append((c[j]-1920)/1920)
                    else:
                        cc.append((c[j]-540)/540)
                
                for j in range(len(e)):
                    if j&1==0:
                        ee.append((e[j]-1920)/1920)
                    else:
                        ee.append((e[j]-540)/540)


                for k in range(7):
                    d=[p_x[points_idx],p_y[points_idx]]
                    goal_=[e[0]-d[0],e[1]-d[1]]
                    if np.linalg.norm(goal_)>max_dis:
                        goal_=(np.array(goal_)/np.linalg.norm(goal_)*max_dis).tolist()
                    if np.linalg.norm(goal_)<50:
                        goal_[0],goal_[1]=0,0
                    points_idx+=1
                    dd=[(d[0]-1920)/1920,(d[1]-540)/540]
                    goal_[0],goal_[1]=goal_[0]+d[0],goal_[1]+d[1]
                    # goal_[0]=(goal_[0]-1920)/1920
                    # goal_[1]=(goal_[1]-540)/540
                    self.trans_file(aa,bb,cc,dd,goal_,f,e)
                    self.total_num+=1

            radi=np.sqrt((self.goal[0]-self.last_goal_list[-2:][0])**2+(self.goal[1]-self.last_goal_list[-2:][1])**2)/2
            p_x,p_y=self.get_gaussian_points_begin(self.goal,max(50,radi),15)
            for i in range(15):
                a=states_list[-1]
                b=near_state_list[-1]
                c=last_goal_list[-1]
                f=is_end_list[-1]
                e=goal_list[-1]
                d=[p_x[i],p_y[i]]
                goal_=[e[0]-d[0],e[1]-d[1]]

                aa,bb,cc,ee=[],[],[],[]
                for j in range(len(a)):
                    if j&1==0:
                        aa.append((a[j]-16)/16)
                    else:
                        aa.append((a[j]-4.5)/4.5)

                for j in range(len(b)):
                    if j&1==0:
                        bb.append((b[j]-16)/16)
                    else:
                        bb.append((b[j]-4.5)/4.5)

                for j in range(len(c)):
                    if j&1==0:
                        cc.append((c[j]-1920)/1920)
                    else:
                        cc.append((c[j]-540)/540)
                
                for j in range(len(e)):
                    if j&1==0:
                        ee.append((e[j]-1920)/1920)
                    else:
                        ee.append((e[j]-540)/540)




                if np.linalg.norm(goal_)>200:
                    goal_=(np.array(goal_)/np.linalg.norm(goal_)*200).tolist()
                if np.linalg.norm(goal_)<50:
                    goal_[0],goal_[1]=0,0
                goal_[0],goal_[1]=goal_[0]+d[0],goal_[1]+d[1]
                dd=[(d[0]-1920)/1920,(d[1]-540)/540]
                # goal_[0]=(goal_[0]-1920)/1920
                # goal_[1]=(goal_[1]-540)/540
                self.trans_file(aa,bb,cc,dd,goal_,1,e)
                self.total_num+=1

            if end_idx>=self.cfile_len:
                flag=False
            if end_idx>=self.cfile_len-5:
                break
            self.last_goal_list.append(self.goal[0])
            self.last_goal_list.append(self.goal[1])
            begin_idx=end_idx


    def get_states3policy_(self):
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
        begin_idx=self.goal_nums
        flag=True
        # self.move_list.clear()
        self.move_list=[]
        while flag:
            states_list,near_state_list,last_goal_list,last_action_list,goal_list,is_end_list=[],[],[],[],[],[]
            self.get_goal()
            end_idx=self.goal_nums
            x,y=-1,-1
            xx,yy=-1,-1
            last_goal=self.last_goal_list[-6:].copy()
            states=[]
            near_state=[]
            self.move_list=[]
            # self.k1=0
            self.k2+=1
            last_action=[0.,0.]
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
                self.move_list.append([int(row[1]/120+0.5),int(row[2]/120+0.5)])
                # self.move_list.append()
                self.max_x=max(self.max_x,row[1])
                self.max_y=max(self.max_y,row[2])
                self.a_max_x=max(self.a_max_x,row[3])
                self.a_max_y=max(self.a_max_y,row[4])
                if x==-1 and y==-1:
                    x,y=int(row[1]/120+0.5),int(row[2]/120+0.5)
                    xx,yy=row[3],row[4]
                    self.move_list.append([x,y])
                    increase_=True
                else:
                    x1,y1=int(row[1]/120+0.5),int(row[2]/120+0.5)
                    self.move_list.append([x1,y1])
                    if x!=x1 or y!=y1:
                        x=x1
                        y=y1
                        last_action[0],last_action[1]=xx,yy
                        xx,yy=row[3],row[4]
                        # self.move_list.append([x,y])
                        increase_=True
                        self.k1+=1
                states,near_state=self.get_s()
                isend=0
                if (i==end_idx-1) and (end_idx>=self.cfile_len-5):
                    isend=1
                if isend or (increase_ and len(states)!=0):
                    if len(states)!=0:
                        self.isend_total+=isend
                        states_list.append(states)
                        near_state_list.append(near_state)
                        if len(last_goal)>6:
                            breakpoint()
                        last_goal_list.append(last_goal)
                        last_action_list.append(last_action)
                        goal_list.append(self.goal)
                        is_end_list.append([isend])
            list_len=len(states_list)
            if list_len==0:
                continue
                # list_len+=1
            begin_=self.last_goal_list[-2:]
            radi=np.sqrt((self.goal[0]-self.last_goal_list[-2:][0])**2+(self.goal[1]-self.last_goal_list[-2:][1])**2)/2


            low_dis=list_len*0.2
            if low_dis<1:
                low_dis=0
            for i in range(list_len):
                a=states_list[i]
                b=near_state_list[i]
                c=last_goal_list[i]
                f=is_end_list[i]
                e=goal_list[i]
                p_x,p_y=self.get_random_points(begin_,goal_list[-1])
                dis_max=200

                aa,bb,cc=[],[],[]
                for i in range(len(a)):
                    if i&1==0:
                        aa.append((a[i]-16)/16)
                    else:
                        aa.append((a[i]-4.5)/4.5)

                for i in range(len(b)):
                    if i&1==0:
                        bb.append((b[i]-16)/16)
                    else:
                        bb.append((b[i]-4.5)/4.5)

                for i in range(len(c)):
                    if i&1==0:
                        cc.append((c[i]-1920)/1920)
                    else:
                        cc.append((c[i]-540)/540)


                if low_dis>i:
                    dis_max=50
                for k in range(len(p_x)):
                    d=[p_x[k],p_y[k]]
                    dd=[(d[0]-1920)/1920,(d[1]-540)/540]
                    goal_=[e[0]-d[0],e[1]-d[1]]
                    if np.linalg.norm(goal_)>dis_max:
                        goal_=(np.array(goal_)/np.linalg.norm(goal_)*dis_max).tolist()
                        goal_[0]/=300
                        goal_[1]/=300
                    self.trans_file(aa,bb,cc,dd,goal_,f)
                    self.total_num+=1

            if end_idx>=self.cfile_len:
                flag=False
            if end_idx>=self.cfile_len-5:
                break
            self.last_goal_list.append(self.goal[0])
            self.last_goal_list.append(self.goal[1])
            begin_idx=end_idx



    def get_random_points(self,begin_goal,end_goal):
        begin_goal=np.array(begin_goal)
        end_goal=np.array(end_goal)
        mid_points=((end_goal+begin_goal)/2).tolist()
        # 半径 扇区 每个扇区点的数目
        radium=[0,200,500,1000,2000]
        section=[16,8,8,8]
        nums=[3,3,3,3]
        x=[]
        y=[]
        for i in range(4):
            for j in range(section[i]):
                points_r=(np.random.rand(nums[i])*(radium[i+1]-radium[i])+radium[i]).tolist()
                angles=(np.random.rand(nums[i])*(np.pi*2/section[i])+j*np.pi*2/section[i]).tolist()
                points_x,points_y=[],[]
                for k in range(nums[i]):
                    points_x.append(float(np.cos(angles[k])*points_r[k]))
                    points_y.append(float(np.sin(angles[k])*points_r[k])*0.5)
                x.extend(points_x)
                y.extend(points_y)
        return x,y




    def get_gaussian_points(self,begin_goal,end_goal,points_num,ratio=0.5):
        begin_goal=np.array(begin_goal)
        end_goal=np.array(end_goal)
        if np.linalg.norm(begin_goal-end_goal)==0:
            radi=np.sqrt((self.goal[0]-self.last_goal_list[-2:][0])**2+(self.goal[1]-self.last_goal_list[-2:][1])**2)/2
            p_x,p_y=self.get_gaussian_points_begin(self.goal,max(50,radi),points_num)
            return p_x,p_y
        sigma1=np.linalg.norm(begin_goal-end_goal)*ratio
        sigma2=sigma1/2
        x_axis=np.array([1,0])
        conv=[[sigma1**2,0],[0,sigma2**2]]
        v1=(end_goal-begin_goal)/np.linalg.norm(end_goal-begin_goal)
        sin1=np.cross(v1,x_axis)
        cos1=np.dot(v1,x_axis)
        if v1[1]>0:
            rotation=np.array([[cos1,-sin1],[sin1,cos1]])
        else:
            rotation=np.array([[cos1,sin1],[-sin1,cos1]])
        x1,y1=np.random.multivariate_normal([0,0],conv,points_num).T
        xy=[x1,y1]
        xy=np.matmul(rotation,xy)
        mid_points=(end_goal+begin_goal)/2
        x1,y1=xy[0]+mid_points[0],xy[1]+mid_points[1]
        return x1.tolist(),y1.tolist()





    def get_gaussian_points_begin(self,begin_goal,rad,points_num):
        begin_goal=np.array(begin_goal)
        sigma1=rad
        sigma2=rad
        conv=[[sigma1**2,0],[0,sigma2**2]]
        x1,y1=np.random.multivariate_normal([0,0],conv,points_num).T
        x1,y1=x1+begin_goal[0],y1+begin_goal[1]
        return x1.tolist(),y1.tolist()

    def get_states4critic(self):
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
        begin_idx=self.goal_nums
        flag=True
        self.move_list.clear()
        self.move_list=[]
        move_list=[]
        near_list=[]
        k=0
        while flag:
            self.get_goal()
            end_idx=self.goal_nums
            x,y=-1,-1
            last_goal=self.last_goal_list[-6:].copy()
            states=[]
            near_state=[]
            self.move_list=[]
            move_list=[]
            near_list=[]
            last_action_list=[]
            lastaction=[-1,-1]
            xx,yy=-1,-1
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
                if x==-1 and y==-1:
                    x,y=int(row[1]/120+0.5),int(row[2]/120+0.5)
                    self.move_list.append([x,y])
                    xx,yy=row[3],row[4]
                    increase_=True
                else:
                    x1,y1=int(row[1]/120+0.5),int(row[2]/120+0.5)
                    self.move_list.append([x1,y1])
                    if x!=x1 or y!=y1:
                        lastaction[0],lastaction[1]=xx,yy
                        xx,yy=row[3],row[4]
                        x=x1
                        y=y1
                        # self.move_list.append([x,y])
                        increase_=True
                        self.k1+=1
                states,near_state=self.get_s()
                isend=0
                if increase_:
                    # print(move_list)
                    last_action_list.append(lastaction)
                    move_list.append(states)
                    near_list.append(near_state)
                if (i==end_idx-1) and (end_idx>=self.cfile_len-5):
                    isend=1
            last_action_list.append([-1 for i in range(len(last_action_list[-1]))])
            move_list.append([-1 for i in range(len(move_list[-1]))])
            near_list.append([-1 for i in range(len(near_list[-1]))])
            move_list_len=len(move_list)
            isend_=0
            points_idx=0

            p_x,p_y=self.get_gaussian_points(self.last_goal_list[-2:],self.goal,move_list_len*7)
            for i in range(move_list_len):
                if i==move_list_len-1:
                    isend_=1
                for k in range(7):
                    p_x[points_idx],p_y[points_idx]=p_x[points_idx],p_y[points_idx]
                    if i!=(move_list_len-1):
                        self.trans_file(move_list[i],near_list[i],[p_x[points_idx],p_y[points_idx]],move_list[i+1],\
                        near_list[i+1],[-1,-1],self.last_goal_list[-6:],self.last_goal_list[-2:],self.goal,[0],[move_list_len])
                    else:
                        self.trans_file(move_list[i],near_list[i],[p_x[points_idx],p_y[points_idx]],[-1 for i in range(len(move_list[i]))],\
                       [-1 for i in range(len(near_list[i]))],[-1,-1],[-1 for i in range(6)],[-1,-1],[-1,-1],[1],[move_list_len])
                    points_idx+=1
                    self.total_num+=1
            radi=np.sqrt((self.goal[0]-self.last_goal_list[-2:][0])**2+(self.goal[1]-self.last_goal_list[-2:][1])**2)/2
            p_x,p_y=self.get_gaussian_points_begin(self.goal,max(100,radi),15)
            for i in range(15):
                self.trans_file(move_list[-1],near_list[-1],[p_x[i],p_y[i]],[-1 for i in range(len(move_list[-1]))],\
                       [-1 for i in range(len(near_list[-1]))],[-1,-1],[-1 for i in range(6)],[-1,-1],[-1,-1],[1],[move_list_len])
                self.total_num+=1    

            if end_idx>=self.cfile_len-5:
                break
            begin_idx=end_idx
            self.last_goal_list.append(self.goal[0])
            self.last_goal_list.append(self.goal[1])
            if end_idx>=self.cfile_len:
                flag=False

    def get_states4critic_(self):
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
        begin_idx=self.goal_nums
        flag=True
        self.move_list.clear()
        self.move_list=[]
        move_list=[]
        near_list=[]
        k=0
        while flag:
            self.get_goal()
            end_idx=self.goal_nums
            x,y=-1,-1
            last_goal=self.last_goal_list[-6:].copy()
            states=[]
            near_state=[]
            self.move_list=[]
            move_list=[]
            near_list=[]
            last_action_list=[]
            lastaction=[-1,-1]
            xx,yy=-1,-1
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
                if x==-1 and y==-1:
                    x,y=int(row[1]/120+0.5),int(row[2]/120+0.5)
                    self.move_list.append([x,y])
                    xx,yy=row[3],row[4]
                    increase_=True
                else:
                    x1,y1=int(row[1]/120+0.5),int(row[2]/120+0.5)
                    self.move_list.append([x1,y1])
                    if x!=x1 or y!=y1:
                        lastaction[0],lastaction[1]=xx,yy
                        xx,yy=row[3],row[4]
                        x=x1
                        y=y1
                        increase_=True
                        self.k1+=1
                states,near_state=self.get_s()
                isend=0
                if increase_:
                    # print(move_list)
                    last_action_list.append(lastaction)
                    move_list.append(states)
                    near_list.append(near_state)
                if (i==end_idx-1) and (end_idx>=self.cfile_len-5):
                    isend=1
            last_action_list.append([-1 for i in range(len(last_action_list[-1]))])
            # move_list.append([-1 for i in range(len(move_list[-1]))])
            # near_list.append([-1 for i in range(len(near_list[-1]))])
            move_list_len=len(move_list)
            isend_=0
            radi=np.sqrt((self.goal[0]-self.last_goal_list[-2:][0])**2+(self.goal[1]-self.last_goal_list[-2:][1])**2)/2
            p_x,p_y=self.get_gaussian_points_begin(self.goal,max(100,radi),15)
            for i in range(15):
                p_x[i],p_y[i]=p_x[i],p_y[i]
                self.trans_file(move_list[-1],near_list[-1],[p_x[i],p_y[i]],[-1 for i in range(len(move_list[-1]))],\
                       [-1 for i in range(len(near_list[-1]))],[-1,-1],[-1 for i in range(6)],[-1,-1],[-1,-1],[1],[move_list_len])
                self.total_num+=1          

            if end_idx>=self.cfile_len-5:
                break
            begin_idx=end_idx
            self.last_goal_list.append(self.goal[0])
            self.last_goal_list.append(self.goal[1])
            if end_idx>=self.cfile_len:
                flag=False


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

    def trans_file(self,*points_list):
        # print(points_list)
        # print(points_list)
        # if not os.path.exists(self.trans_path):
        #     os.makedirs(self.trans_path)
        # save_path=os.path.join(self.trans_path,str(int(self.trans_nums))+'.txt')
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
def pretrain_save_file(name,type_,mode=None):
    # schemes=['s1','s2','s3','s4']
    schemes=['s1']
    for scheme in schemes:
        # print(f'schemes {scheme}')
        dirs=get_file_paths(scheme)
        random.shuffle(dirs)
        ft=FileToTxt()
        ft.mode=mode

        p=os.path.join('/home/wu_tian_ci/eyedata/mixed/',name+str(mode))

        # p=os.path.join()
        p=os.path.join(p,scheme)
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
            if type_==wtuc.POLICY:
                ft.get_states3policy()
            elif type_==wtuc.CRITIC:
                ft.get_states4critic()
            elif type_==wtuc.CRITIC_:
                ft.get_states4critic_()
            elif type_==wtuc.POLICY_:
                ft.get_states3policy_()
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

    # pretrain_save_file('5_policy_1last_move_not_origin',wtuc.POLICY,5)
    # pretrain_save_file('5_critic_1last_move_not_origin',wtuc.CRITIC,5)
 
    pretrain_save_file('5_policy_1last_move',wtuc.POLICY,5)
    # pretrain_save_file('5_critic_1last_move',wtuc.CRITIC,5)
    # pretrain_save_file('5_policy_1last_move_',wtuc.POLICY_,5)
    # pretrain_save_file('5_critic_1last_move_',wtuc.CRITIC_,5)
 
    # pretrain_save_file('3_policy',wtuc.POLICY,3)