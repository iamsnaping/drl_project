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
import argparse

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
    net.eval()
    max_loss = 0.
    min_loss = np.inf
    loss = nn.MSELoss(reduction='mean')
    t = 0
    ll = 0.
    # net.eval()
    with torch.no_grad():
       for last_goal,state,near_state,last_action,goal in train_iter:
            state=state.to(device)
            last_goal=last_goal.to(device)
            last_action=last_action.to(device)
            goal=goal.to(device)
            near_state=near_state.to(device)
            # print(x.shape,y.shape,z.shape)
            # print(state.shape,last_goal.shape,last_mouse.shape)
            l = loss(net(last_goal,state,near_state,last_action), goal)
            max_loss = max(max_loss, l.cpu().detach().numpy())
            min_loss = min(min_loss, l.cpu().detach().numpy())
            ll += l
            t += 1
    return max_loss, min_loss, ll / t


def train_epoch(train_iter, test_iter, net, lr, epochs, device,load_path=None,store_path=None):
    net.to(device)
    trainer = torch.optim.Adam(lr=lr, params=net.parameters())
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(trainer, T_max=epochs)
    loss = nn.MSELoss()
    # loss2=nn.CosineSimilarity(dim=-1)
    best_loss = np.inf
    test_path=os.path.join(store_path,'test.txt')
    train_path=os.path.join(store_path,'train.txt')
    m_store_path=os.path.join(store_path,'PolicyNet.pt')
    train_list=[]
    test_list=[]
    f=open(test_path,'w')
    f.close()
    f=open(train_path,'w')
    f.close()
    if load_path is not None:
        m_load_path=os.path.join(load_path,'PolicyNet.pt')
        net.load_state_dict(torch.load(m_load_path))
    p_lr=0.
    learning_rate=[]
    for i in tqdm(range(epochs)):
        net.train()
        max_loss = 0.
        min_loss = np.inf
        ll, t = 0, 0
        loss_flag=1
        for last_goal,state,near_state,last_action,goal in train_iter:
            state=state.to(device)
            last_goal=last_goal.to(device)
            last_action=last_action.to(device)
            goal=goal.to(device)
            near_state=near_state.to(device)
            trainer.zero_grad()
            # print(x.shape,y.shape,z.shape)
            # print(state.shape,last_goal.shape,last_mouse.shape)\
            # print(net(last_goal,state,near_state,last_action))
            # print(goal)
            ans=net(last_goal,state,near_state,last_action)
            # ans=torch.clamp(ans,max=100,min=-100)
            # print(loss(ans, goal))
            # print(last_goal,state,near_state,last_action,ans)
            # print(ans)
            l = loss(ans, goal)
            # l2= torch.sum(loss2(ans,goal).squeeze(),dim=-1)
            if ((loss_flag&1)==1) and l < best_loss :
                best_loss = l
                torch.save(net.state_dict(), m_store_path)
            loss_flag+=1
            max_loss = max(max_loss, l.cpu().detach().numpy())
            min_loss = min(min_loss, l.cpu().detach().numpy())
            ll += l
            t += 1
            l.backward()
            trainer.step()
        p_lr=trainer.param_groups[0]['lr']
        schedule.step()
        t_max, t_min, t_ave = test_acc(net, test_iter,device)
        # train_list.append([max_loss,min_loss,(ll/t).cpu().detach().numpy(),trainer.param_groups[0]['lr']])
        # test_list.append([t_max,t_min,t_ave.cpu().detach().numpy(),trainer.param_groups[0]['lr']])
        with open(train_path,'a') as f:
            f.write('epochs :'+str(i)+' ')
            f.write(str(max_loss)+' '+str(min_loss)+' '+str(((ll/t).cpu().detach().numpy()))+' '+str(trainer.param_groups[0]['lr'])+'\n')
            f.close()
        with open(test_path,'a') as f:
            f.write('epochs :'+str(i)+' ')
            f.write(str(t_max)+' '+str(t_min)+' '+str(t_ave.cpu().detach().numpy())+' '+str(trainer.param_groups[0]['lr'])+' '+'\n')
            f.close()
    net.load_state_dict(torch.load(m_store_path))
    # print('test acc ')
    # print(test_acc(net, test_iter,device),end=' ')
    # t=1
    # with open(test_path,'w') as f:
    #     for data in test_list:
    #         write_str='eposides: '+str(t)
    #         t+=1
    #         for i in data:
    #             write_str+=' '+str(i)
    #         write_str+='\n'
    #         f.write(write_str)
    #     f.close()
    # t=1
    # with open(train_path,'w') as f:
    #     for data in train_list:
    #         write_str='eposides: '+str(t)
    #         t+=1
    #         for i in data:
    #             write_str+=' '+str(i)
    #         write_str+='\n'
    #         f.write(write_str)
    #     f.close()
    t_max, t_min, t_ave =test_acc(net, test_iter,device)
    with open(test_path,'a') as f:
        f.write('last test acc: ')
        f.write(str(t_max)+' '+str(t_min)+' '+str(t_ave.cpu().detach().numpy()))
        f.close()
    with open(train_path,'a') as f:
        f.write('last test acc: ')
        f.write(str(t_max)+' '+str(t_min)+' '+str(t_ave.cpu().detach().numpy()))
        f.close()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='ddpg_pretrain_policy parser')
    parser.add_argument('-epochs_1',type=int,default=700)
    parser.add_argument('-epochs_2',type=int,default=500)
    parser.add_argument('-lr_1',type=float,default=0.07)
    parser.add_argument('-lr_2',type=float,default=0.03)
    parser.add_argument('-load_path',type=str,default=None)
    parser.add_argument('-train_file_path',type=str,default='5_policy_1last_move5')
    parser.add_argument('-nums',type=int,default='1')
    parser.add_argument('-net',type=str,default='net1')
    parser.add_argument('-cuda',type=str,default='cuda:3')
    args=parser.parse_args()
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    print(device)
    base_path='/home/wu_tian_ci/eyedata/mixed/'
    data_path=os.path.join(base_path,args.train_file_path)
    data_path=os.path.join(data_path,'s1')
    train_path=os.path.join(data_path,'train1.json')
    test_path=os.path.join(data_path,'test1.json')
    store_path=os.path.join('/home/wu_tian_ci/drl_project/mymodel/ddpg/pretrain_data','policy')
    store_path=os.path.join(store_path,args.train_file_path)
    store_path=os.path.join(store_path,str(args.nums))
    store_path1=os.path.join(store_path,str(args.epochs_1)+str(args.lr_1).replace('.','_'))
    store_path2=os.path.join(store_path,str(args.epochs_2)+str(args.lr_2).replace('.','_'))
    if not os.path.exists(store_path1):
        os.makedirs(store_path1)
    if not os.path.exists(store_path2):
        os.makedirs(store_path2)
    if args.net=='net1':
        ##self,action_n,state_n,mouse_n ,num_layer,output_n 6,10,2,7,2
        net=BaseNet(38,2,10,2)
    else:
        net=PolicyBaseNet(6,20,10,2,7,2)
    train_dataset=dataset_loader_policy(train_path)
    test_dataset=dataset_loader_policy(test_path)
    train_iter=torch.utils.data.DataLoader(dataset=train_dataset,shuffle=True,batch_size=10240,num_workers=12)
    test_iter=torch.utils.data.DataLoader(dataset=test_dataset,shuffle=True,batch_size= 5120,num_workers=6)
    train_epoch(train_iter, test_iter, net, args.lr_1, args.epochs_1, device,load_path=None,store_path=store_path1)
    # print(store_path1)
    train_epoch(train_iter, test_iter, net, args.lr_2, args.epochs_2, device,load_path=store_path1,store_path=store_path2)

# /home/wu_tian_ci/anaconda3/envs/drl_env/bin/python /home/wu_tian_ci/drl_project/mymodel/ddpg/ddpg_pretrain_policy.py -train_file_path 5_policy_1last_move_not_origin5
# /home/wu_tian_ci/anaconda3/envs/drl_env/bin/python /home/wu_tian_ci/drl_project/mymodel/ddpg/ddpg_pretrain_critic.py -train_file_path 5_critic_1last_move_not_origin5 -load 5_policy_1last_move_not_origin5
    
