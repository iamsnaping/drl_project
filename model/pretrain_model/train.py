import torch
from torch import nn
from utils.my_dataset import *
from torch.utils.data import Dataset
import numpy as np
from model.pretrain_model.mlpdrl import *
import matplotlib.pyplot as plt
import matplotlib

random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)

class Aminator(object):
    def __init__(self,name):
        self.max_loss=[]
        self.min_loss=[]
        self.ave_loss=[]
        self.epoch=[]
        self.name=name

    def add_data(self,maxloss,minloss,aveloss):
        self.max_loss.append(maxloss.detach().numpy())
        self.min_loss.append(minloss.detach().numpy())
        self.ave_loss.append(aveloss.detach().numpy())
        self.epoch.append(len(self.ave_loss))



    def draw(self):
        matplotlib.rcParams['font.family'] = ['Heiti TC']
        plt.plot(self.epoch[2:-1],self.max_loss[2:-1],label=self.name+'max loss',marker='x',mfc='orange',ms=5,mec='c',lw=1.0,ls="--",c='Red')
        plt.plot(self.epoch[2:-1], self.min_loss[2:-1], label=self.name + 'min loss', marker='x', mfc='orange', ms=5, mec='c',
                 lw=1.0, ls="--", c='Green')
        plt.plot(self.epoch[2:-1], self.ave_loss[2:-1], label=self.name + 'ave loss', marker='x', mfc='orange', ms=5, mec='c',
                 lw=1.0, ls="--", c='Blue')
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

def test_acc(net,test_iter):
    max_loss = 0.
    min_loss = np.inf
    loss = nn.MSELoss(reduction='mean')
    t = 0
    ll = 0.
    net.eval()
    with torch.no_grad():
        for X, y,z in test_iter:
            l = loss(net(X,y),z)
            max_loss = max(max_loss, l)
            min_loss = min(min_loss, l)
            ll += l
            t += 1
    return max_loss, min_loss, ll / t

def train_epoch(train_iter,test_iter,net,lr,epochs,device):
    net.to(device)
    trainer=torch.optim.Adam(lr=lr,params=net.parameters())
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(trainer,T_max=epochs)
    loss=nn.MSELoss()
    best_loss=np.inf
    animator1=Aminator('train')
    animator2=Aminator('test')
    net.load_state_dict(torch.load('MLPDRL.pt'))
    for i in range(epochs):
        net.train()
        max_loss=0.
        min_loss=np.inf
        ll,t=0,0
        for x,y,z in train_iter:
            x.to(device)
            y.to(device)
            z.to(device)
            trainer.zero_grad()
            l=loss(net(x,y),z)
            if l<best_loss:
                best_loss=l
                torch.save(net.state_dict(),'MLPDRL.pt')
            max_loss = max(max_loss, l)
            min_loss = min(min_loss, l)
            ll+=l
            t+=1
            l.backward()
            # print(l.grad)
            trainer.step()
        # tensor([-0.5303, 0.5303], requires_grad=True)
        # Parameter
        # containing:
        # tensor([0.4697, -0.0519], requires_grad=True)
        # Parameter
        # containing:
        # tensor([-0.5303, -1.0520], requires_grad=True)
        # Parameter
        # containing:
        # tensor([[-0.4892, -0.7022],
        #         [0.3382, -0.0555]], requires_grad=True)
        # Parameter
        # containing:
        # tensor([0., 0.], requires_grad=True)
        # for params in net.parameters():
        #     print(params)
        #     for name,param in net.named_parameters():
        #         print('name: ',name,'   grad:',param.grad,'  loss:',l)
        schedule.step()
        t_max, t_min, t_ave = test_acc(net, test_iter)
        print(f'train epoch {i} max loss {max_loss} min loss {min_loss} ave loss {ll / t}')
        print(f'test epoch {i} max loss {t_max} min loss {min_loss} ave loss {t_ave}')
        animator1.add_data(max_loss,min_loss,aveloss=ll/t)
        animator2.add_data(t_max,t_min,t_ave)
    animator1.draw()
    animator2.draw()
    net.load_state_dict(torch.load('MLPDRL.pt'))
    print('test acc ')
    print(test_acc(net,test_iter))

if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_od=dataset_loader('D:\\eyedata\\mixed\\pretrain\\s1\\train')
    test_od=dataset_loader('D:\\eyedata\\mixed\\pretrain\\s1\\test')
    train_iter=torch.utils.data.DataLoader(dataset=train_od, shuffle=False, batch_size=256)
    test_iter=torch.utils.data.DataLoader(dataset=test_od,shuffle=False,batch_size=256)
    net=MLPDRL(60,2)
    net.apply(weight_init)
    train_epoch(train_iter,test_iter,net,0.01,100,device)
