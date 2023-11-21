import os.path
import socket
import requests
import json
from copy import deepcopy as dp
import numpy as np
from filter import *
import zmq
from config import *

import pandas as pd
import time


class RequestGetter(object):
    def __init__(self):
        pass

    #0->predict 1->order
    def getRequestJson(self,flag,data):
        requestDict = {}
        #clickRegion,eyeRegion,mouseS
        if flag==0:
            clickRegion,eyeRegion,mouseS=data

            requestDict['clickRegion'] = clickRegion
            requestDict['eyeRegion'] = eyeRegion
            requestDict['mouseS']=mouseS
        elif flag==1:
            orderType=data
            requestDict['order'] = orderType
        requestJson=json.dumps(requestDict)
        return requestJson


def getTimeStamp():

    import calendar
    now=time.gmtime()
    now=calendar.timegm(now)

    from  datetime import datetime
    timeFormat=datetime.fromtimestamp(now)
    ans=timeFormat.strftime('%Y%m%d%H%M%S')
    return ans



# # 1.创建socket
# tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#
# # 2. 链接服务器
# #10.170.40.186
# server_addr = ("10.170.40.186", 7788)
# tcp_socket.connect(server_addr)
#
# # 3. 发送数据
# send_data = '123'
# tcp_socket.send(send_data.encode("gbk"))
#
# # 4. 关闭套接字
# tcp_socket.close()
# url='http://5301.ETVP.TECH:7788/getPredict'
#
# post_json=json.dumps({'type':0,'data':'{"4":[1,2,3]}'})
#
# r=requests.post(url,data=post_json,headers={'content-type':'application/json'})
# print(type(json.loads(r.text).get('data')))

'''
send formation json
type: 0-> predict 1->put into training
type 0- > len=1
len:
0:...
1:...
2:...
3:...
....
len:...

'''

class ResposeF(object):

    def __init__(self,code=200):
        self.code=code
        self.data='False'
        self.message='error'
        self.flag=False


    def __str__(self):
        return 'code: '+str(self.code)+' flag'+str(self.flag)+' data: '+str(self.data)+' message: '+self.message

class FlaskSender(object):
    def __init__(self):
        pass

    def getPredct(self,data):
        response=requests.post(os.path.join(ConfigEnum.BASEPATH.value,ConfigEnum.PREDICT.value),data=data,headers={'content-type':'application/json'})
        # data.clear()
        rf=ResposeF(response.status_code)
        if response.status_code!=200:
            return rf
        # print(f'this is response {response.text}')
        rf.data=json.loads(response.text).get('data')
        rf.message=json.loads(response.text).get('message')
        rf.flag=bool(json.loads(response.text).get('flag'))
        # updata data -> rewards and
        return rf

    def excuteOrder(self,data):
        messageJson=json.dumps({"order": data})
        response = requests.post(os.path.join(ConfigEnum.BASEPATH.value, ConfigEnum.ORDER.value), data=messageJson,
                                 headers={'content-type': 'application/json'})

        rf = ResposeF(response.status_code)
        if response.status_code != 200:
            return rf
        # print(f'this is response {response.text}')
        rf.data = json.loads(response.text).get('data')
        rf.message = json.loads(response.text).get('message')
        rf.flag = bool(json.loads(response.text).get('flag'))
        # updata data -> rewards and
        return rf

    def sendFiles(self,filePath,fileName):
       file=open(filePath,'rb')
       data=file.read()
       # peoplenum,filenum,scenenum
       # response=requests.post(os.path.join(ConfigEnum.BASEPATH.value,ConfigEnum.UPLOAD.value),files={'file':data},data=jsonDict,headers={'content-type':'application/binary'})
       response=requests.post(os.path.join(ConfigEnum.BASEPATH.value,ConfigEnum.UPLOAD.value),files={'file':data,'fileName':fileName})

    def sendData(self,fileName,peopleNum,sceneNum,dataList):
        messageJson=json.dumps({"dataList": dataList,"fileName":fileName,"peopleNum":peopleNum,"sceneNum":sceneNum})
        response = requests.post(os.path.join(ConfigEnum.BASEPATH.value, ConfigEnum.DATA.value), data=messageJson,
                                 headers={'content-type': 'application/json'})
        rf = ResposeF(response.status_code)
        if response.status_code != 200:
            return rf
        # print(f'this is response {response.text}')
        rf.data = json.loads(response.text).get('data')
        rf.message = json.loads(response.text).get('message')
        rf.flag = bool(json.loads(response.text).get('flag'))
        # updata data -> rewards and
        return rf


class OnlineDataRecorder(object):
    def __init__(self, peopleNum='01', sceneNum='1', storePath='', fileNum=0) -> None:
        self.peopleNum = peopleNum
        self.sceneNum = sceneNum
        self.dataList = []
        self.temp = []
        self.beginTimeStamps = ''
        self.endTimesStamps = ''
        self.fileNum = fileNum
        self.fileDate = getTimeStamp()[4:-6]
        self.recordFlag = False
        self.beginFlag=False
        self.updated = 0
        self.fs=FlaskSender()


        # peopleNum Scene
        self.baseStorePath = os.path.join(storePath, self.peopleNum, self.sceneNum)
        if not os.path.exists(self.baseStorePath):
            os.makedirs(self.baseStorePath)
        else:
            self.fileNum = len(os.listdir(self.baseStorePath))

    def beginRecord(self):
        self.recordFlag = True
        self.beginTimeStamps = getTimeStamp()[-6:]

    def endRecord(self):
        self.recordFlag = False
        self.endTimesStamps = getTimeStamp()[-6:]
        self.write()

    def dataTimeStamp(self):
        timeStamp = int(time.time())
        return timeStamp

    # eyeX,eyeY,mouseX,mouseY,mouseS
    def add(self, *data):
        eyeX, eyeY, mouseX, mouseY, mouseS = data
        timeStamp = self.dataTimeStamp()
        self.temp.append([timeStamp, eyeX, eyeY, mouseX, mouseY, mouseS])
        if mouseS == 0:
            self.dataList.extend(self.temp)
            self.temp = []

    # filename-> S scene L fileNum P peopleNum
    def write(self):
        if self.fileNum < 10:
            fileNum = '0' + str(self.fileNum)
        else:
            fileNum = str(self.fileNum)
        self.fileNum += 1

        fileName = 'S' + self.sceneNum + 'L' + fileNum + 'P' + self.peopleNum + '-' + self.fileDate + '-' + self.beginTimeStamps + '-' + self.endTimesStamps + '.csv'
        savePath = os.path.join(self.baseStorePath,fileName)
        if len(self.dataList) == 0:
            return
        # print(savePath)
        # print(self.dataList)
        df = pd.DataFrame(self.dataList)
        df.to_csv(savePath, header=None, index=False)
        response=self.fs.sendData(fileName,self.peopleNum,self.sceneNum,self.dataList)
        print(response.message)
        self.clear()
        self.updated += 1

    def clear(self):
        self.dataList = []
        self.temp = []
        self.fileNum += 1


# create when client click
class JsonCreater(object):
    def __init__(self):
        self.data={"type":1,"len":0}

    def creater(self,datas):
        for data in datas:
            self.data[data['8']]=data.createJson()
        d=json.dumps(self.data)
        self.clear()
        return d

    def clear(self):
        self.data.clear()
        self.data['type']=1
        self.data['len']=0

# state,near_state,last_goal_list_,lastaction,action,isend_,reward,ep
#state,near_state,last_goal_list,last_action,action,isend,rewards,goal,ep


    #state,near_state,last_goal_list,last_action,action,isend,rewards,goal,i


# create data could be predict

if __name__=='__main__':
    fc=FlaskSender()
    fc.sendFiles('D:\\online_run\\test2.csv',[1,2,3])

