import torch
from torch import nn
import os
import sys
from tqdm import tqdm
sys.path.append('/home/wu_tian_ci/drl_project')
from torch.utils.data import Dataset
import numpy as np
from mymodel.ddpg.ddpg_base_net import *
import matplotlib.pyplot as plt
import matplotlib
from mymodel.ddpg.ddpg_dataloader import *
from torch.nn import functional as F
import argparse
# import warnings
# warnings.filterwarnings('error') 


os.chdir(sys.path[0])

random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)


class Aminator(object):
    def __init__(self, name):
        self.max_loss = []
        self.min_loss = []
        self.ave_loss = []
        self.epoch = []
        self.lr=[]
        self.name = name

    def add_data(self, maxloss, minloss, aveloss,lr):
        self.max_loss.append(maxloss.cpu().detach().numpy())
        self.min_loss.append(minloss.cpu().detach().numpy())
        self.ave_loss.append(aveloss.cpu().detach().numpy())
        self.lr.append(lr)
        self.epoch.append(len(self.ave_loss))

    def draw(self):
        matplotlib.rcParams['font.family'] = ['Heiti TC']
        plt.plot(self.epoch[2:-1], self.max_loss[2:-1], label=self.name + 'max loss', marker='x', mfc='orange', ms=5,
                 mec='c', lw=1.0, ls="--", c='Red')
        plt.plot(self.epoch[2:-1], self.min_loss[2:-1], label=self.name + 'min loss', marker='x', mfc='orange', ms=5,
                 mec='c',
                 lw=1.0, ls="--", c='Green')
        plt.plot(self.epoch[2:-1], self.ave_loss[2:-1], label=self.name + 'ave loss', marker='x', mfc='orange', ms=5,
                 mec='c',
                 lw=1.0, ls="--", c='Blue')
        plt.plot(self.epoch[2:-1], self.lr[2:-1], label=self.name + 'lr', marker='x', mfc='black', ms=5,
                 mec='c',
                 lw=1.0, ls="--", c='Black')
        plt.xlabel('epochs')
        # y轴标签
        plt.ylabel('loss')

        # 图表标题
        plt.title('loss pic')

        # 显示图例
        plt.legend()
        # 显示图形
        plt.show()


def weight_init(m):
    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def test_acc(net, test_iter,device):
    max_loss = 0.
    min_loss = np.inf
    loss = nn.MSELoss(reduction='mean')
    t = 0
    ll = 0.
    net.eval()
    with torch.no_grad():
        for x, y, z in test_iter:
            x=x.to(device)
            y=y.to(device)
            z=z.to(device)
            l = loss(net(x, y), z)
            max_loss = max(max_loss, l.cpu().detach().numpy())
            min_loss = min(min_loss, l.cpu().detach().numpy())
            ll += l
            t += 1
    return max_loss, min_loss, ll / t


def get_gaussian(x):
    return 1/(np.sqrt(2)*np.pi)*np.exp(-((x)**10)/2)


def get_tanh(x):
    cnt=torch.e*torch.ones_like(x,dtype=torch.float32).to(x.get_device())
    # cnt2=cnt_*torch.ones_like(x,dtype=torch.float32).to(x.get_device())
    loge1=torch.log(cnt+x)
    loge2=-torch.log(cnt+x)
    # print(loge1,loge2)
    return (loge1-loge2)/(loge2+loge1)

def get_scores(goal_p,goal_n,last_action,action_,is_end,move_epochs):
    with torch.no_grad():
        loss=torch.nn.MSELoss(reduction='none')
        l1=torch.norm(last_action-goal_n,dim=-1,keepdim=True)
        l2=torch.norm(action_-goal_n,dim=-1,keepdim=True)
        ll=(l1-l2)*0.04
        # bias=torch.min((l1-l2)*0.5,torch.ones_like(l2).to(device)*50)*0.04
    return (1-is_end)*0.5*(torch.tanh(150-ll))+(torch.tanh((100-l2)*0.1))*is_end*move_epochs


def train_epoch(train_iter, test_iter, net,action_net, lr, epochs, device,load_path=None,store_path=None):
    net.to(device)
    action_net.to(device)
    trainer = torch.optim.Adam(lr=lr, params=net.parameters())
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(trainer, T_max=epochs)
    loss1 = nn.MSELoss()
    loss2=nn.MSELoss()
    best_loss = np.inf
    test_path=os.path.join(store_path,'test.txt')
    train_path=os.path.join(store_path,'train.txt')
    m_store_path=os.path.join(store_path,'CriticNet.pt')
    f=open(train_path,'w')
    f.close()
    if load_path is not None:
        m_load_path=os.path.join(load_path,'CriticNet.pt')
        net.load_state_dict(torch.load(m_load_path))
    for i in tqdm(range(epochs)):
        net.train()
        ll, t = 0, 0
        for last_goal,state,near_state,last_action,n_state,n_near_state,n_last_action,begin_,goal,isend,move_epochs in train_iter:
            last_goal=last_goal.to(device)
            state=state.to(device)
            near_state=near_state.to(device)
            last_action=last_action.to(device)
            n_state=n_state.to(device)
            n_near_state=n_near_state.to(device)
            n_last_action=n_last_action.to(device)
            begin_=begin_.to(begin_)
            goal=goal.to(device)
            isend=isend.to(device)
            trainer.zero_grad()
            action1=action_net(last_goal,state,near_state,last_action)
            # action,last_goal,state_n+n_state_n,mouse_n
            value1=net(action1,last_goal,torch.cat([state,near_state],dim=-1),last_action)
            action2=action_net(last_goal,n_state,n_near_state,action1)
            value2=net(action2,last_goal,torch.cat([n_state,n_near_state],dim=-1),action1)
            reward1=get_scores(begin_,goal,last_action,action1,isend, torch.as_tensor(move_epochs).to(device))
            l=loss2(value1,(1-isend)*value2*0.9+reward1)
            if l < best_loss:
                best_loss = l
                torch.save(net.state_dict(), m_store_path)
            ll += l
            t += 1
            l.backward()
            trainer.step()
        schedule.step()
        with open(train_path,'a') as f:
            f.write('epochs :'+str(i)+' ')
            f.write(str(((ll/t).cpu().detach().numpy()))+' '+str(trainer.param_groups[0]['lr'])+'\n')
            f.close()

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='ddpg_pretrain_critic parser')
    parser.add_argument('-epochs_1',type=int,default=450)
    parser.add_argument('-epochs_2',type=int,default=400)
    parser.add_argument('-lr_1',type=float,default=0.07)
    parser.add_argument('-lr_2',type=float,default=0.03)
    parser.add_argument('-load_path',type=str,default=None)
    parser.add_argument('-train_file_path',type=str,default='5_critic_1last_move5')
    parser.add_argument('-nums',type=int,default=1)
    parser.add_argument('-net',type=str,default='net2')
    parser.add_argument('-cuda',type=str,default='cuda:3')
    parser.add_argument('-load',type=str,default='5_policy_1last_move5')
    args=parser.parse_args()
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    print(device)
    base_path='/home/wu_tian_ci/eyedata/mixed/'
    data_path=os.path.join(base_path,args.train_file_path)
    data_path=os.path.join(data_path,'s1')
    train_path=os.path.join(data_path,'train1.json')
    test_path=os.path.join(data_path,'test1.json')
    store_path=os.path.join('/home/wu_tian_ci/drl_project/mymodel/ddpg/pretrain_data','critic')
    store_path=os.path.join(store_path,args.train_file_path)
    store_path=os.path.join(store_path,str(args.nums))
    store_path1=os.path.join(store_path,str(args.epochs_1)+str(args.lr_1).replace('.','_'))
    store_path2=os.path.join(store_path,str(args.epochs_2)+str(args.lr_2).replace('.','_'))
    if not os.path.exists(store_path1):
        os.makedirs(store_path1)
    if not os.path.exists(store_path2):
        os.makedirs(store_path2)
        #action,last_goal,state_n+n_state_n,mouse_n,num_layer,output_n
    net=PolicyBaseNet(2,30,6,2,5,1)
    train_dataset=dataset_loader_critic(train_path)
    test_dataset=dataset_loader_critic(test_path)
    # print(train_path,test_path)
    train_iter=torch.utils.data.DataLoader(dataset=train_dataset,shuffle=True,batch_size=10240,num_workers=18)
    test_iter=torch.utils.data.DataLoader(dataset=test_dataset,shuffle=True,batch_size= 512,num_workers=4)
    if args.net=='net1':
        action_net=BaseNet(38,2,10,2)
    else:
        action_net=PolicyBaseNet(6,20,10,2,7,2)
    action_load_path=os.path.join('/home/wu_tian_ci/drl_project/mymodel/ddpg/pretrain_data/policy',args.load)
    action_load_path=os.path.join(action_load_path,'1/5000_03/PolicyNet.pt')
    action_net.load_state_dict(torch.load(action_load_path))
    train_epoch(train_iter, test_iter, net,action_net, args.lr_1, args.epochs_1, device,load_path=None,store_path=store_path1)
    train_epoch(train_iter, test_iter, net, action_net,args.lr_2, args.epochs_2, device,load_path=store_path1,store_path=store_path2)
