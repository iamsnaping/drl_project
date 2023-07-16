import torch
from torch import nn
import random
from tide import *
from tide_dataset import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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
        plt.savefig(self.name+'.png')
        # 显示图形
        # plt.show()


def weight_init(m):
    random.seed(1114)
    np.random.seed(1114)
    torch.manual_seed(1114)
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
        for w,x, y, z in test_iter:
            w=w.to(device)
            y=y.to(device)
            z=z.to(device)
            x=x.to(device)
            l = loss(net(w,x, y), z)
            max_loss = max(max_loss, l)
            min_loss = min(min_loss, l)
            ll += l
            t += 1
    return max_loss, min_loss, ll / t


def train_epoch(train_iter, test_iter, net, lr, epochs, device,is_load):
    net.to(device)
    trainer = torch.optim.Adam(lr=lr, params=net.parameters())
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(trainer, T_max=epochs)
    loss = nn.MSELoss()
    best_loss = np.inf
    animator1 = Aminator('train2')
    animator2 = Aminator('test2')
    if is_load:
        net.load_state_dict(torch.load('TiDE2.pt'))
        animator1 = Aminator('train load2')
        animator2 = Aminator('test load2')
    p_lr=0.
    learning_rate=[]
    for i in range(epochs):
        net.train()
        max_loss = 0.
        min_loss = np.inf
        ll, t = 0, 0
        for w,x, y, z in train_iter:
            w=w.to(device)
            x=x.to(device)
            y=y.to(device)
            z=z.to(device)
            trainer.zero_grad()
            l = loss(net(w,x,y), z)
            if l < best_loss:
                best_loss = l
                torch.save(net.state_dict(), 'TiDE2.pt')
            max_loss = max(max_loss, l)
            min_loss = min(min_loss, l)
            ll += l
            t += 1
            l.backward()
            trainer.step()
        p_lr=trainer.param_groups[0]['lr']
        schedule.step()
        t_max, t_min, t_ave = test_acc(net, test_iter,device)
        print(f'train epoch {i} max loss {max_loss} min loss {min_loss} ave loss {ll / t} lr {p_lr}')
        print(f'test epoch {i} max loss {t_max} min loss {min_loss} ave loss {t_ave} lr {p_lr}')
        animator1.add_data(max_loss, min_loss,ll / t,lr=p_lr*100)
        animator2.add_data(t_max, t_min, t_ave,p_lr*100)
    animator1.draw()
    animator2.draw()
    net.load_state_dict(torch.load('TiDE2.pt'))
    print('test acc ')
    print(test_acc(net, test_iter,device),end=' ')


if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_od3 = tdataset_loader('/home/wu_tian_ci/eyedata/mixed/pretrain5/train')
    test_od3 = tdataset_loader('/home/wu_tian_ci/eyedata/mixed/pretrain5/test')
    train_iter3 = torch.utils.data.DataLoader(dataset=train_od3, shuffle=False, batch_size=102400)
    test_iter3 = torch.utils.data.DataLoader(dataset=test_od3, shuffle=False, batch_size=512)
   
    net = TiDE2(45,1,30,20,8,72,3)
    net.apply(weight_init)
    # train_epoch(train_iter3, test_iter3, net, 0.07, 200, device,False)
    train_epoch(train_iter3,test_iter3,net,0.03,200,device,True)
