import torch
from torch import nn
import os
import sys
from tqdm import tqdm

sys.path.append('/home/wu_tian_ci/drl_project')
from torch.utils.data import Dataset
from mymodel.envdataset.mydataset import *
import numpy as np
from mymodel.pretrain_model.mlpdrl import *
import matplotlib.pyplot as plt
import matplotlib


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


def train_epoch(train_iter, test_iter, net, lr, epochs, device,is_load,load_path=None,store_path=None):
    net.to(device)
    trainer = torch.optim.Adam(lr=lr, params=net.parameters())
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(trainer, T_max=epochs)
    loss = nn.MSELoss()
    best_loss = np.inf
    test_path=os.path.join(store_path,'test.txt')
    train_path=os.path.join(store_path,'train.txt')
    m_load_path=os.path.join(load_path,'MLPDRL.pt')
    m_store_path=os.path.join(store_path,'MLPDRL.pt')
    f=open(test_path,'w')
    f.close()
    f=open(train_path,'w')
    f.close()
    if is_load:
        net.load_state_dict(torch.load(m_load_path))
    p_lr=0.
    learning_rate=[]
    for i in tqdm(range(epochs)):
        net.train()
        max_loss = 0.
        min_loss = np.inf
        ll, t = 0, 0
        for x, y, z in train_iter:
            x=x.to(device)
            y=y.to(device)
            z=z.to(device)
            trainer.zero_grad()
            l = loss(net(x, y), z)
            if l < best_loss:
                best_loss = l
                torch.save(net.state_dict(), m_store_path)
            max_loss = max(max_loss, l.cpu().detach().numpy())
            min_loss = min(min_loss, l.cpu().detach().numpy())
            ll += l
            t += 1
            l.backward()
            trainer.step()
        p_lr=trainer.param_groups[0]['lr']
        schedule.step()
        t_max, t_min, t_ave = test_acc(net, test_iter,device)
        with open(train_path,'a') as f:
            f.write('epochs :'+str(i)+' ')
            f.write(str(max_loss)+' '+str(min_loss)+' '+str(((ll/t).cpu().detach().numpy()))+' '+str(trainer.param_groups[0]['lr'])+'\n')
            f.close()
        with open(test_path,'a') as f:
            f.write('epochs :'+str(i)+' ')
            f.write(str(t_max)+' '+str(t_min)+' '+str(t_ave.cpu().detach().numpy())+' '+str(trainer.param_groups[0]['lr'])+' '+'\n')
            f.close()
    net.load_state_dict(torch.load(m_store_path))
    print('test acc ')
    print(test_acc(net, test_iter,device),end=' ')


if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_od3 = dataset_loader3('/home/wu_tian_ci/eyedata/mixed/pretrain14/s1/train')
    test_od3 = dataset_loader3('/home/wu_tian_ci/eyedata/mixed/pretrain14/s1/test')
    train_iter3 = torch.utils.data.DataLoader(dataset=train_od3, shuffle=False, batch_size=1024,num_workers=12)
    test_iter3 = torch.utils.data.DataLoader(dataset=test_od3, shuffle=False, batch_size=256,num_workers=12)
    # for x,y,z in train_od3:
    #     print(x,y,z)
    net = MLPDRL2(13,15,8,10,3)
    net.apply(weight_init)
    # train_epoch(train_iter, test_iter, net, 0.01, 100,
    # device)
    base_path='/home/wu_tian_ci/drl_project/mymodel/pretrain_data'
    load_path=os.path.join(base_path,'1')
    store_path=os.path.join(base_path,'2')
    if not os.path.exists(load_path):
        os.makedirs(load_path)
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    train_epoch(train_iter3, test_iter3, net, 0.07, 450, device,False,load_path,load_path)
    train_epoch(train_iter3, test_iter3, net, 0.03, 400, device,True,load_path,store_path)
