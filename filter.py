import PIL
# import pynput
import pandas
from PIL import ImageGrab
import numpy as np
# import pynput.keyboard
import os
import collections

from copy import deepcopy as dp

GAMMA=0

AREAS=[[1360, 460, 150, 70],
[1580, 460, 150, 70],
[1360, 600, 150, 70],
[1580, 600, 150, 70],
[1360, 740, 150, 70],
[1580, 740, 150, 70],
[1360, 880, 150, 70],
[1580, 880, 150, 70],
# 8 button
[2020, 70, 450, 170],
[2010, 272, 460, 390],
[575, 135, 785, 527],
[1398, 281, 430, 385],
[90, 688, 460, 335],
[587, 708, 375, 328],
[988, 695, 375, 328],
[1402, 698, 430, 322],
#8 areas
[2031, 160, 91, 41],
[2141, 160, 91, 41],
[2251, 160, 91, 41],
[2361, 160, 91, 41],
[2041, 448, 181, 60],
[2251, 448, 181, 61],
[2041, 508, 191, 61],
[2251, 508, 181, 61],
[2031, 606, 90, 40],
[2341, 606, 90, 40],
[2760, 600, 41, 41],
[2830, 600, 41, 41],
[2970, 600, 41, 41],
[2900, 600, 41, 41],
[3524, 396, 131, 22],
[3524, 436, 131, 22],
[3344, 536, 121, 61],
[3464, 536, 121, 61],
[3604, 536, 101, 61],
[3344, 606, 90, 40],
[3614, 606, 90, 40],
[2222, 786, 113, 20],
[2222, 826, 113, 20],
[2042, 896, 121, 61],
[2182, 906, 101, 41],
[2312, 896, 121, 61],
[2042, 966, 90, 40],
[2332, 966, 90, 40],
[2660, 785, 61, 51],
[2610, 835, 51, 61],
[2720, 835, 50, 60],
[2660, 895, 61, 50],
[2670, 850, 41, 31],
[2530, 965, 90, 40],
[2770, 965, 90, 40],
[2938, 812, 140, 40],
[3108, 812, 151, 40],
[2938, 852, 161, 40],
[3108, 852, 140, 40],
[2938, 892, 151, 40],
[3108, 892, 151, 41],
[2928, 972, 90, 40],
[3158, 972, 90, 40],
[3490, 810, 121, 20],
[3490, 850, 121, 20],
[3490, 890, 120, 20],
[3490, 930, 120, 20],
[3340, 970, 90, 40],
[3620, 970, 90, 40]
]


class Area(object):

    def __init__(self):
        # 0-> round 1->rectangle
        self.shape=0
        self.center=np.array([0.,0.])
        # if is round -> r=width=length
        self.width=0
        self.length=0
        self.areaNum=''
        self.TFCorner=np.array([0.,0.])
        self.LRCorner=np.array([0.,0.])

    @staticmethod
    def initFromTetrad(tetrad):
        area=Area()
        area.center = np.array([int(tetrad[0] + tetrad[2] * 0.5), int(tetrad[1] + tetrad[3] * 0.5)])
        area.width = tetrad[2]
        area.length = tetrad[3]
        return area

    def initPara(self,shape,center,width,length,areaNum=None):
        self.shape=shape
        self.width=width
        self.length=length
        self.areaNum=areaNum
        self.center=center
        if shape==1:
            x,xx=center[0]-width*0.5,center[0]+width*0.5
            y,yy=center[1]-length*0.5,center[1]+width*0.5
            self.TFCorner=np.array([x,y])
            self.TFCorner=np.array([xx,yy])

    def initParaFromTetrad(self,tetrad):
        self.center=np.array([int(tetrad[0]+tetrad[2]*0.5),int(tetrad[1]+tetrad[3]*0.5)])
        self.width=tetrad[2]
        self.length=tetrad[3]


    def judge(self,point):
        if self.shape==0:
            square=np.pi*self.width*self.width
        else:
            square=self.width*self.length
        if ( self.TFCorner[0]<=point[0]<= self.LRCorner[0])and (self.TFCorner[1]<=point[1]<=self.LRCorner[1]):
            return True,square
        return False,square


class MainWindow(object):

    def __init__(self):
        self.areas=[]

    def initFromTetrad(self,tetrads):
        for tetrad in tetrads:
            self.areas.append(Area.initFromTetrad(tetrad))

    def judge(self,point):
        minSqaure=0.
        x,y=-1.,-1.
        for area in self.areas:
            flag,sqaure=area.judge(point)
            if flag:
                if sqaure < minSqaure:
                    minSqaure=sqaure
                    x,y=area.center[0],area.center[1]
        return x,y

class AdsorptionSite(object):
    def __init__(self,tetrads):
        self.mw=MainWindow()
        self.tetrads=tetrads
        self.mw.initFromTetrad(tetrads)

    def judge(self,point):
        point[0]*=1.25
        x,y=self.mw.judge(point)
        if x==-1 and y==-1:
            x,y=point[0],point[1]
        point=np.array([x/1.25,y/1.25])
        return point

class EyeFilter(object):

    def __init__(self, r=10):
        self.xFormer = collections.deque(maxlen=r)
        self.yFormer = collections.deque(maxlen=r)
        self.xLatter = collections.deque(maxlen=r)
        self.yLatter = collections.deque(maxlen=r)
        self.TFast = 0.1
        self.TSlow = 10
        self.threshold = 50
        self.filteredPoint = np.array([0., 0.])
        self.leftFull = 0
        self.maxlen = r
        self.lastFilter = np.array([0., 0.])

    # 0-r-1 before r- 2*r-1 after
    def add(self, point):
        x, y = point[0], point[1]
        if self.leftFull >= self.maxlen:
            self.xLatter.append(self.xFormer[0])
            self.yLatter.append(self.yFormer[0])
            self.xFormer.append(x)
            self.yFormer.append(y)
        else:
            self.xFormer.append(x)
            self.yFormer.append(y)
            self.leftFull += 1

    def filter(self):
        if self.leftFull < self.maxlen and self.leftFull != 0:
            x = sum(self.xFormer) / self.leftFull
            y = sum(self.yFormer) / self.leftFull
            return np.array([x, y])
        xFormerMean = np.mean(self.xFormer)
        xLatterMean = np.mean(self.xLatter)
        yFormerMean = np.mean(self.yFormer)
        yLatterMean = np.mean(self.yLatter)
        if np.abs(xLatterMean - xFormerMean) > self.threshold:

            x = (self.xFormer[-1] + self.TFast * self.filteredPoint[0]) / (1 + self.TFast)
        else:
            x = (self.xFormer[-1] + self.TSlow * self.filteredPoint[0]) / (1 + self.TSlow)
        if np.abs(yLatterMean - yFormerMean) > self.threshold:
            y = (self.yFormer[-1] + self.TFast * self.filteredPoint[1]) / (1 + self.TFast)
        else:
            y = (self.yFormer[-1] + self.TSlow * self.filteredPoint[1]) / (1 + self.TSlow)
        self.lastFilter = dp(self.filteredPoint)
        self.filteredPoint = np.array([x, y])
        return self.filteredPoint

    def clear(self):
        self.xFormer.clear()
        self.yFormer.clear()
        self.xLatter.clear()
        self.yLatter.clear()
        self.TFast = 0.1
        self.TSlow = 10
        self.threshold = 50
        self.filteredPoint = np.array([0., 0.])
        self.leftFull = 0
        self.lastFilter = np.array([0., 0.])



EYEAREAS = [[0, 0, 1300, 1080], [1300, 0, 1920, 390], [1300, 390, 1920, 1080],
            [0 + 1920, 0, 560 + 1920, 260], [0 + 1920, 260, 560 + 1920, 670], [0 + 1920, 670, 560 + 1920, 1080],  # left
            [560 + 1920, 0, 1360 + 1920, 680], [560 + 1920, 680, 975 + 1920, 1080],
            [975 + 1920, 680, 1380 + 1920, 1080],  # mid
            [1380 + 1920, 0, 1920 + 1920, 260], [1380 + 1920, 260, 1920 + 1920, 680],
            [1380 + 1920, 680, 1920 + 1920, 1080]]  # right

CLICKAREAS = [[0, 0, 1300, 1080], [1300, 0, 1920, 390], [1300, 390, 1920, 1080],
              [0 + 1920, 0, 560 + 1920, 260], [0 + 1920, 260, 560 + 1920, 670], [0 + 1920, 670, 560 + 1920, 1080],
              # left
              [560 + 1920, 0, 1360 + 1920, 680], [560 + 1920, 680, 975 + 1920, 1080],
              [975 + 1920, 680, 1380 + 1920, 1080],  # mid
              [1380 + 1920, 0, 1920 + 1920, 260], [1380 + 1920, 260, 1920 + 1920, 680],
              [1380 + 1920, 680, 1920 + 1920, 1080]]  # right


class Region(object):

    def __init__(self, name):
        # 0-> round 1->rectangle
        self.regions = []
        self.regionNum = []
        self.regionHeart = []
        self.counting = []
        self.name = name

    def setRegions(self, regions):
        number =int( 0)
        for region in regions:
            self.regions.append(region)
            self.regionNum.append(number)
            number += 1
            self.regionHeart.append([0.5 * (region[0] + region[2]), 0.5 * (region[1] + region[3])])
        # self.counting = [0 for i in range(number + 4)]

    def judge(self, regionNum:int):
        # print(f'this is regionNum {regionNum}')
        for heart, num in zip(self.regionHeart, self.regionNum):
            # print(f'this is num {num}')
            if regionNum==num:
                return heart


    def judgeNow(self, x, y):
        # x = x * 1.25
        # y = y * 1.25
        for region, num in zip(self.regions, self.regionNum):
            if region[0] <= x <= region[2] and region[1] <= y <= region[3]:
                # self.counting[num] += 1
                return num
        #  1
        # 2 4
        #  3
        flagX = 0
        flagY = 0
        if x < 0:
            flagX = 2
            x = int(np.abs(x))
        elif x > 3840:
            flagX = 4
            x -= 3840
        if y < 0:
            flagY = 1
            y = int(np.abs(y))
        elif y > 1080:
            flagY = 3
            y -= 1080
        if flagX != 0 and flagY != 0:
            if x >= y:
                flagY = 0
            else:
                flagX = 0
        # print(flagX,flagY)
        # self.counting[len(self.regionNum) + flagX + flagY - 1] += 1
        return len(self.regionNum) + flagX + flagY - 1


class EyeHandPolicy(object):

    def __init__(self,listLen=100,threshold=30):
        self.handList=np.zeros((listLen,2))
        self.maxSize=listLen
        self.cursor=0
        self.threshold=threshold


    def add(self,x,y):
        # print(self.cursor,self.maxSize)
        self.handList[self.cursor][0],self.handList[self.cursor][1]=x,y
        self.cursor=(1+self.cursor)%self.maxSize
        # print(self.cursor,self.maxSize)
        ave=self.handList.mean(axis=0)
        scores=np.sqrt(np.mean(np.sum(np.square(self.handList-ave),axis=-1)))
        # print(scores)
        if scores>self.threshold:
            return False
        return True


class DataCreater(object):

    def __init__(self):
        self.eyeRegion = Region('eye')
        self.clickRegion = Region('click')
        self.eyeRegion.setRegions(EYEAREAS)
        self.clickRegion.setRegions(CLICKAREAS)
        self.clickList=[]
        self.eyeMoveList=[]
        self.lastClick=-1
        self.lastEyeRegion=-1
        self.lastClickMode = False
        self.beginFlag=True


    def getEyeList(self):
        moveList=[]
        if len(self.eyeMoveList)<10:
            moveList=dp(self.eyeMoveList)
        else:
            step=len(self.eyeMoveList)//10
            for i in range(9):
                moveList.append(self.eyeMoveList[i*step])
            moveList.append(self.eyeMoveList[-1])
        length=len(moveList)
        if length<10:
            moveList.append([0 for i in range(10-len(moveList))])
        return moveList,length

    # refresh when the transaction ends
    def refresh(self):
        self.eyeMoveList=[]
        self.clickList=[]
        self.lastClick=-1
        self.lastEyeRegion=-1
        self.lastClickMode = False

    # refresh when tra ends
    def refreshTra(self):
        self.eyeMoveList=[]

    def addClick(self,mouseX,mouseY):
        clickRegion=self.clickRegion.judgeNow(mouseX,mouseY)
        if len(self.clickList)==3:
            moveList=dp(self.clickList)
            self.clickList[0]=self.clickList[1]
            self.clickList[1]=self.clickList[2]
            self.clickList[2]=clickRegion
            return (True,moveList,clickRegion)
        else:
            self.clickList.append(clickRegion)
            return (False,-1)

    def addEye(self,eyeX,eyeY):
        eyeRegion=self.eyeRegion.judgeNow(eyeX,eyeY)
        if eyeRegion!=self.lastEyeRegion:
            self.eyeMoveList.append(eyeRegion)
            self.lastEyeRegion=eyeRegion
            moveList,length = self.getEyeList()
            return (True, moveList, length)
        else:
            return (False,-1)


    def predict(self,*data):
        eyeX,eyeY,mouseX,mouseY,mouseS=data
        eyeList=self.addEye(eyeX,eyeY)
        clickList=self.addClick(mouseX,mouseY)
        if clickList[0]==False or eyeList[0]==False:
            return (False,-1)
        #eyeList length clickList clickRegion,clickMouse
        return (True,eyeList[1],eyeList[2],clickList[1],clickList[2],mouseS)

    def judge(self,*data):
        eyeX, eyeY, mouseX, mouseY, mouseS = data
        eyeRegion = self.eyeRegion.judgeNow(eyeX, eyeY)
        clickRegion=self.clickRegion.judgeNow(mouseX,mouseY)

        if eyeRegion != self.lastEyeRegion:
            # self.eyeMoveList.append(eyeRegion)
            self.lastEyeRegion = eyeRegion
            # moveList, length = self.getEyeList()

            return True,clickRegion,eyeRegion,mouseS
        else:
            return False,clickRegion,eyeRegion,mouseS


if __name__ == '__main__':
    # global areas
    # ads=AdsorptionSite(AREAS)
    # print(len(AREAS))
    print(CLICKAREAS)
