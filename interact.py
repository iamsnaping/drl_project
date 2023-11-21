import random
import time

import pynput.mouse as pm
import pynput.keyboard as pk
import threading

import zmq
from pynput.keyboard import Key
from filter import *
from flaskclient import *
import flaskclient as fc
import filter as FILTER
from flaskclient import *
import json
import pyautogui
FLAG = False
PRESS_FLAG = False
# 2 -> end 1 -> begin to record·
BEGIN_FLAG = False
X = 0
Y = 0
# push esc to exit

lock = threading.Lock()
condition = threading.Condition(lock)
START_TRAIN=False
CLICK_TIMES=0
LAST_CLICK_TIMES=0
MOVE_FLAG=False
fs=FlaskSender()


'''
ctrl_l -> begin the record eye coord and mouse operation
/ -> end the recording and operation
alt_l -> begin to train
\ end the train

'''
def onPress(key):
    global START_TRAIN,BEGIN_FLAG,CLICK_TIMES,MOVE_FLAG,condition,fs
    try:
        if key.char=='[':
            print('begin')
            response = fs.excuteOrder(1)
            print(response.message)
            BEGIN_FLAG=True
            with condition:
                print('awake')
                condition.notify_all()
        elif key.char==';':
            print('begin to train')
            response=fs.excuteOrder(1)
            print(response.message)
        elif key.char =='\'':
            print('end train')
            response=fs.excuteOrder(0)
            print(response.message)
        elif key.char ==']':
            print('end')
            response=fs.excuteOrder(3)
            print(response.message)
            BEGIN_FLAG=False
        elif key.char=='.':
            print('move')
            MOVE_FLAG=True
        elif key.char=='/':
            MOVE_FLAG=False
    except:
        pass
        # print(f'this is key {key}')
        # print('ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR')

#
# def onClick(x, y, button, pressed):
#     global FLAG, X, Y, PRESS_FLAG
#     if pressed:
#         PRESS_FLAG = True
#         with condition:
#             # do predict and send whole data call runFinc
#             runFunc(x, y, 1)
#             pass
#     if FLAG:
#         return False
#
#
# def ls():
#     with pm.Listener(on_click=onClick) as listener:
#         listener.join()
#
#
def kls():
    # listener=pk.Listener(on_press=onPress)
    # listener.start()
    print(4)
    with pk.Listener(on_press=onPress) as listener:
        print(5)
        listener.join()
        print(6)

class Client(object):

    def __init__(self):
        self.sender=FlaskSender()

    # x->mouse_x y->mouse_y
    def predict(self,data):
 #clickRegion,eyeRegion,mouseS
        response = self.sender.getPredct(data)
        if response.code==200:
            ans = response.data
            message=response.message
            return True,response.flag,ans,message
        else:
            message=response.message
            return False,response.flag,response.data,message


    def sendMessage(self,t,data):
        # 2 -> stop train
        messageJson=json.dumps({"flag": t, "order": data})
        return self.sender.sender(messageJson)




global GLOBAL_GAZE_DATA_1, GLOBAL_GAZE_DATA_2


# 0-> 指令
# 1-> 数据

def getEye():
    global BEGIN_FLAG,START_TRAIN,CLICK_TIMES,LAST_CLICK_TIMES,MOVE_FLAG,condition
    interactClient=Client()
    policy=EyeHandPolicy()

    context = zmq.Context()
    skt = context.socket(zmq.SUB)
    skt.connect('tcp://localhost:5566')
    skt.setsockopt_string(zmq.SUBSCRIBE,'')


    clickRegion = Region('click')
    clickRegion.setRegions(CLICKAREAS)
    lastRegion=-1
    odr=OnlineDataRecorder(storePath='D:\\online_run\\onlineRecorde')
    dc=DataCreater()
    rg=RequestGetter()
    while True:
        if not BEGIN_FLAG:
            with condition:
                if odr.beginFlag==True:
                    odr.beginFlag=False
                    odr.endRecord()
                condition.wait()
        if not odr.beginFlag:
            odr.beginFlag=True
            odr.beginRecord()

        message = skt.recv()
        jsonData=json.loads(message.decode())
        eyeX,eyeY=float(jsonData.get('actualX')),float(jsonData.get('actualY'))
        mouseX,mouseY,mouseS,id=float(jsonData.get('x')),float(jsonData.get('y')),int(jsonData.get('pd')),int(jsonData.get('id'))

        # id=1
        # eyeX,eyeY,mouseX,mouseY=random.randint(0,3840),random.randint(0,1080),random.randint(0,3840),random.randint(0,1080)
        # pro=random.randint(0,9)
        # if pro<2:
        #     mouseS=0
        # else:
        #     mouseS=3

        if id==2:
            eyeX+=1920
        odr.add(eyeX, eyeY, mouseX, mouseY, mouseS)
        ans=dc.judge(eyeX,eyeY,mouseX,mouseY,mouseS)
        moveFlag=policy.add(mouseX,mouseY)
        if ans[0]==False:
            continue
        requestJson=rg.getRequestJson(0,ans[1:])
        ans=interactClient.predict(requestJson)
        if ans[0]==False:
            continue
        if ans[1]==False:
            print(ans)
        print(f'predict {ans}')
        regionNum=int(ans[1])
        print(regionNum)
        if regionNum in [12,-1]:
            continue
        if lastRegion==regionNum:
            continue
        xy=clickRegion.judge(regionNum)
        # print(f'this is xy {xy} {type(xy)}')
        if not MOVE_FLAG:
            continue
        if moveFlag:
            continue
        pyautogui.moveTo(x=int(xy[0]),y=int(xy[1]),duration=0.1)
        # print(f'this is ans {ans} flag {ans[0]} data {ans[1]} message {ans[2]}')





if __name__ == '__main__':
    t=threading.Thread(target=getEye)
    print(1)
    t.start()
    print(2)
    kls()
    print(3)
