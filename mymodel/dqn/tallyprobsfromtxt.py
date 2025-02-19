import re
import os
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from pylab import mpl
# 设置显示中文字体
# mpl.rcParams["font.sans-serif"] = ["Noto Serif CJK HK"]

# mpl.rcParams["axes.unicode_minus"] = False

dirsPaths=[
'/home/wu_tian_ci/drl_project/mymodel/dqn/dqnonline/onlinedata/20231228/model/191200/23/3/1/trajectory',
'/home/wu_tian_ci/drl_project/mymodel/dqn/dqnonline/onlinedata/20231228/model/191200/24/3/1/trajectory',
'/home/wu_tian_ci/drl_project/mymodel/dqn/dqnonline/onlinedata/20231228/model/214020/25/3/1/trajectory',
'/home/wu_tian_ci/drl_project/mymodel/dqn/dqnonline/onlinedata/20231228/model/214020/26/3/1/trajectory',
'/home/wu_tian_ci/drl_project/mymodel/dqn/dqnonline/onlinedata/20231228/model/231223/27/3/1/trajectory',
'/home/wu_tian_ci/drl_project/mymodel/dqn/dqnonline/onlinedata/20231228/model/231223/28/3/1/trajectory',
'/home/wu_tian_ci/drl_project/mymodel/dqn/dqnonline/onlinedata/20231229/model/003342/28/3/1/trajectory',
'/home/wu_tian_ci/drl_project/mymodel/dqn/dqnonline/onlinedata/20231229/model/000620/27/3/2/trajectory'
]

right=[]
fault=[]
endRight=[]
endFault=[]


def getFlag(wordsList):
    goal=int(wordsList[15])
    action=int(wordsList[16])
    prob=float(wordsList[27+action])
    mask=float(wordsList[17])
    endFlag=False
    # print(mask)
    if int(mask+0.5)==0:
        endFlag=True
    if action==12:
        action=int(wordsList[20])
    if action==goal:
        right.append(prob)
        if endFlag:
            endRight.append(prob)
    else:
        fault.append(prob)
        if endFlag:
            endFault.append(prob)
wordsDict=dict()
for filesPath in dirsPaths:
    fileList=os.listdir(filesPath)
    for fpath in fileList:
        filePath=os.path.join(filesPath,fpath)
        f=open(filePath,'r')
        ans=f.readlines()
        words=[]
        for l in ans:
            # print(l)
            pattern = r'[-+]?\b\d+(?:\.\d*)?(?:[eE][-+]?\d+)?\b'
            flag=l.find('tensor')
            if flag==-1 and len(l)<4:
                continue
            matches = re.findall(pattern, l)
            for m in matches:
                words.append(m)
            if len(words)==41:
                getFlag(words)
                words=[]
        # breakpoint()
print(np.mean(right),np.max(right),np.min(right),np.std(right),len(right))
print(np.mean(fault),np.max(fault),np.min(fault),np.std(fault),len(fault))
print(np.mean(endRight),np.max(endRight),np.min(endRight),np.std(endRight),len(endRight))
print(np.mean(endFault),np.max(endFault),np.min(endFault),np.std(endFault),len(endFault))
title=[
    'right mean: '+str(np.round(np.mean(right),2))+'  std: '+str(np.round(np.std(right),2)),
    'wrong mean: '+str(np.round(np.mean(fault),2))+'  std: '+str(np.round(np.std(fault),2)),
    'end right mean: '+str(np.round(np.mean(endRight),2))+'  std: '+str(np.round(np.std(endRight),2)),
    'end wrong mean: '+str(np.round(np.mean(endFault),2))+'  std: '+str(np.round(np.std(endFault),2))
]

pngName=['right.png','fault.png','endRight.png','endFault.png']


import matplotlib.ticker as mtick

formatter=mtick.PercentFormatter(xmax=len(right))

# 绘制直方图和拟合的高斯分布
plt.figure()
plt.hist(right, bins=10, density=False,  color='g')
xmin, xmax = plt.xlim()
plt.title(title[0])
# formatter=mtick.PercentFormatter(xmax=len(right))
# plt.gca().yaxis.set_major_formatter(formatter)
plt.savefig(pngName[0])

plt.figure()
plt.hist(fault, bins=10, density=False,  color='g')
xmin, xmax = plt.xlim()
plt.title(title[1])
# formatter=mtick.PercentFormatter(xmax=len(fault))
# plt.gca().yaxis.set_major_formatter(formatter)
plt.savefig(pngName[1])

plt.figure()
plt.hist(endRight, bins=10, density=False,  color='g')
xmin, xmax = plt.xlim()
plt.title(title[2])
# formatter=mtick.PercentFormatter(xmax=len(endRight))
# plt.gca().yaxis.set_major_formatter(formatter)
plt.savefig(pngName[2])

plt.figure()
plt.hist(endFault, bins=10, density=False, color='g')
xmin, xmax = plt.xlim()
plt.title(title[3])
# formatter=mtick.PercentFormatter(xmax=len(endFault))
# plt.gca().yaxis.set_major_formatter(formatter)
plt.savefig(pngName[3])