from ddpg_base_net import *
import PIL
# import pynput
from PIL import ImageGrab
import numpy as np
# import pynput.keyboard
from ddpg_env import *
import pygame
import os

import threading
# from pynput import keyboard
from copy import deepcopy as dp



class Shewer:
    def __init__(self,name='datashewer'):
        self.RED=(0, 0, 255)
        self.BLUE = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BGC=(255, 255, 255)
        self.screen=None
        self.name=name
        self.nowGoal=[0.,0.]
        self.nowBegin=[0.,0.]
        self.picNum=0
        self.picPath='/home/wu_tian_ci/drl_project/mymodel/ddpg/pretrain_data/pic_path'
        self.picOrder='0'
        # if not os.path.exists(self.picPath):
        #     os.makedirs(self.picPath)
    
    def getScreen(self,name='datashewer'):
        pygame.init()
        self.name=name
        screen = pygame.display.set_mode((1920, 1080))
        pygame.display.set_caption(name)
        self.screen=screen
        return screen
    
    def startScreen(self):
        pygame.init()
        screen = pygame.display.set_mode((1920, 1080))
        pygame.display.set_caption(self.name)
        self.screen=screen
        self.picPath=os.path.join(self.picPath,self.picOrder)
        if not os.path.exists(self.picPath):
            os.makedirs(self.picPath)
    
    def showPoints(self,x,y,flag=False,nowGoal=None,nowBegin=None):
        if nowGoal is not None:
            self.nowGoal=nowGoal
        if nowBegin is not None:
            self.nowBegin=nowBegin
        self.screen.fill(self.BGC)
        pygame.draw.circle(self.screen, self.RED, (x,y), 10)
        pygame.draw.circle(self.screen, self.RED, (x,y), 10)
        if flag:
            pygame.draw.circle(self.screen, self.GREEN, (x,y), 10)
        else:
            pygame.draw.circle(self.screen, self.BLUE, (x,y), 10)
        pygame.display.update()
        pygame.image.save(self.screen, os.path.join(self.picPath,str(self.picNum)+'.png'))
        self.picNum+=1

def load_model(modelpath):
    modelpath=os.path.join('/home/wu_tian_ci/drl_project/mymodel/ddpg/pretrain_data/ddpg_train',modelpath)
    model=PolicyBaseNet(6,20,10,2,7,2)
    model.load_state_dict(torch.load(modelpath))
    return model

# def on_press(key):
#     if isinstance(key, pynput.keyboard.KeyCode):
#         if key.char == 'q':
#             global EXIT_FLAG
#             EXIT_FLAG = True
#             return False

def main():
    # def run_key():
    #     with keyboard.Listener(on_press=on_press) as lsn:
    #         lsn.join()
    # t = threading.Thread(target=run_key)
    # t.start()
    shewer=Shewer()
    shewer.picOrder='2/0'
    model=load_model('1last_move5/ActorNet.pt')
    shewer.startScreen()
    env=DDPGEnv('/home/wu_tian_ci/eyedata/seperate/02/1')
    env.state_mode=1
    env.load_dict()
    env.get_file()
    last_action=np.array([0.,0.],dtype=np.float32)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    while True:
        ans=env.act()
        if isinstance(ans,bool):
            break
        else:
            if begin_flag==True:
                # print(ans[3],ans[4])
                last_action[0],last_action[1]=ans[3][0],ans[3][1]
                shewer.nowBegin=dp(ans[3])
                shewer.nowGoal=dp(ans[4])
                begin_flag=False
            state=torch.tensor([ans[0]],dtype=torch.float32).unsqueeze(0).to(device)
            near_state=torch.tensor([ans[1]],dtype=torch.float32).unsqueeze(0).to(device)
            last_goal_list=torch.tensor([ans[2]],dtype=torch.float32).unsqueeze(0).to(device)
            action=model(last_goal_list,state,near_state,torch.as_tensor([last_action[0],\
                last_action[1]],dtype=torch.float32).reshape(1,1,2).to(device))
            action=action.cpu().detach().numpy()
            # print(ans[4])
            if ans[5]==1:
                begin_flag=True
            last_action[0],last_action[1]=action[0],action[1]
        shewer.showPoints(action[0],action[1])



if __name__=='__main__':
    main()