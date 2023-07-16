import functools
import os

import numpy as np
import pandas as pd
from mymodel.enviroment import *


root_path='D:\\eyedata'

list_dirs=os.listdir(root_path)
last_ordinate=[0,0]
#flag ==False -> 全相同 状态
flag=False
sequence_len=1
max_len=0
idx_x,idx_y=0,0
id_path=''
num_list=[]
total_len=0
section_total=[0 for i in range(40)]
class ordinate_len:
    def __init__(self,x,y,o_len,flag,p):
        self.x=x
        self.y=y
        self.o_len=o_len
        self.flag=flag
        self.p=p
    def __str__(self):
        return ('x:%d y:%d olen:%d flag:%d path:%s'%(self.x,self.y,self.o_len,self.flag,self.p))

def pending_ordinate(x,y,p):
    global sequence_len,flag,num_list,total_len
    if (x==last_ordinate[0] )and (y==last_ordinate[1]):
        if flag==False:
            sequence_len+=1
        else:
            if sequence_len>=1000:
                total_len+=1
            num_list.append(ordinate_len(last_ordinate[0],last_ordinate[1],sequence_len,flag,p))
            last_ordinate[0],last_ordinate[1]=x,y
            sl=sequence_len
            sequence_len=1
            flag==False
            return sl
    else:
        if flag==False:
            if sequence_len>=1000:
                total_len+=1
            num_list.append(ordinate_len(last_ordinate[0], last_ordinate[1], sequence_len, flag, p))
            last_ordinate[0], last_ordinate[1] = x, y
            sl = sequence_len
            sequence_len = 1
            flag == True
            return sl
        else:
            sequence_len+=1
    return -1


env=EyeEnvironment()

for dirs in list_dirs:
    if len(dirs)>3:
        continue
    d_path=os.path.join(root_path,dirs)
    f_list=os.listdir(d_path)
    max_count=0.
    for f in f_list:
        if len(f)<20:
            continue
        cvs_f=os.path.join(d_path,f)
        env.get_file(appoint_path=cvs_f)
        for i in range(1,env.file_len):
            max_count=max(max_count,env.state_list[i])
            if env.state_list[i]==env.state_list[i-1]:
                if i==env.file_len-1:
                    section_total[env.state_list[i]//50]+=1
            else:
                section_total[env.state_list[i-1]//50]+=1

        print(f'dir {dirs} file {f} seciton total {section_total} max_count {max_count} {max_count//50}')



def cmp1(a,b):
    if a.o_len>b.o_len:
        return 1
    return -1
# num_list.sort(key=functools.cmp_to_key(cmp1))
# for i in num_list:
#     if i.o_len<=50:
#         section_total[int(i.o_len)]+=1
    # section_total[int(i.o_len/50)]+=1
# print(len(num_list))
# t=1
# for i in section_total:
#     print(f't : {t} i:{i}')
#     t+=1
t=1
total=0
#886046
for i in section_total:
    # total+=i
    print(f't : {t*50-50} ~ {t*50} i:{i}')
    t+=1
# for i in section_total:
#     total+=i
#     print(f't : {t} i:{total}')