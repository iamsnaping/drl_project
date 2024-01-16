import os
import random
import shutil
import sys
sys.path.append('/home/wu_tian_ci/drl_project')
sys.path.append('/home/wu_tian_ci/drl_project/wtcutils')
import pandas as pd
import json


'''
seperate/train\test/
miexd/train\test/

'''
import os

import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from dqnenv import *

class FileToTxt:
    def __init__(self,savePath,envs):
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        self.jsonPath=os.path.join(savePath,'test2.json')
        self.envs=envs
        self.jsonData={}
        self.transNums=0
    

    def getData(self):
        t=0
        for env in self.envs:
            for scene in range(1,5):
                envPath=os.path.join('/home/wu_tian_ci/eyedata/seperate',env,str(scene))
                dqnenv=DQNRNNEnv(base_path=envPath)
                # if int(env) in [6, 11, 7, 4, 13]:
                #     t+=1
                #     print(f'continue {t} {int(env)} {scene}')
                #     dqnenv.topN=5
                #     dqnenv.eval=True
                dqnenv.shuffle=False
                dqnenv.load_dict()
                dqnenv.get_file()
                while True:
                    # eye,click,goal,finish,length
                    ans=dqnenv.act()
                    if isinstance(ans,bool):
                        break
                    eye,click,goal,finish,length=ans
                    self.trans_file(eye,click,goal,length,int(env),scene)
        f=open(self.jsonPath,'w')
        json.dump(self.jsonData,f,indent=2)
        f.close()





    def trans_file(self,*points_list):
        d={}
        i=0
        for points in points_list:
            d[str(i)]=points
            i+=1
        data={}
        data[str(self.transNums)]=d
        self.jsonData.update(data)
        self.transNums+=1

      

if __name__ == '__main__':
    # envNum1=[12, 14, 18, 21, 8, 2, 5, 20, 19, 9, 16, 10, 3, 15, 17]
    envNum=[6, 11, 7, 4, 13]
    envs=[]
    # envNum=[ i for i in range(2,22)]
    for i in envNum:
        if i<10:
            envs.append('0'+str(i))
        else:
            envs.append(str(i))
    ft=FileToTxt('/home/wu_tian_ci/eyedata/dqn_policy',envs)
    ft.getData()