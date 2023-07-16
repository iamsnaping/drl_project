import torch
from torch import nn
import random
from tide import *
from tide_dataset import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
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
    max_loss_1 = 0
    min_loss_1 = np.inf
    max_loss_2 = 0.
    min_loss_2 = np.inf
    loss = nn.MSELoss(reduction='mean')
    t = 0
    ll1 = 0.
    ll2=0.
    net.eval()
    max_loss=nn.MSELoss(reduction='none')
    with torch.no_grad():
        for v,w,x, y, z in test_iter:
            v=v.to(device)
            w=w.to(device)
            y=y.to(device)
            z=z.to(device)
            x=x.to(device)
            res=net(v,w,x)
            click_s=res[0].to(device)
            click_=res[1].to(device)
            click_[:,:,0:15]=res[1][:,:,0:15]*16
            click_[:,:,15:]=res[1][:,:,15:]*9
            click_s[:,:,0:]=res[0][:,:,0:]*5
            click_.to(device)
            click_s.to(device)
            l1=loss(click_,y)
            l2=loss(click_s,z)
            max_loss_1 = max(max_loss_1, torch.max(max_loss(res[0],y)).cpu().detach().numpy())
            min_loss_1 = min(min_loss_1, torch.min(max_loss(res[0],y)).cpu().detach().numpy())
            max_loss_2 = max(max_loss_2, torch.max(max_loss(res[1],z)).cpu().detach().numpy())
            min_loss_2 = min(min_loss_2, torch.min(max_loss(res[1],z)).cpu().detach().numpy())
            ll1 += l1
            ll2+=l2
            t += 1
    # print(max_loss_1,min_loss_1,min_loss_2,max_loss_2)
    return max_loss_1, min_loss_1,max_loss_2, min_loss_2,(ll1/t).cpu().detach().numpy(),(ll2/t).cpu().detach().numpy()


def train_epoch(train_iter, test_iter, net, lr, epochs, device,store_name,is_load=False):
    net.to(device)
    trainer = torch.optim.Adam(lr=lr, params=net.parameters())
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(trainer, T_max=epochs)
    loss = nn.MSELoss()
    max_loss=nn.MSELoss(reduction='none')
    best_loss = np.inf
    # animator1 = Aminator('train')
    # animator2 = Aminator('test')
    test_list=[[] for i in range(8)]
    train_list=[[] for i in range(8)]
    if is_load:
        net.load_state_dict(torch.load('/home/wu_tian_ci/drl_project/txtans/'+'2TiDE.pt'))
        # animator1 = Aminator('train load')
        # animator2 = Aminator('test load')
    p_lr=0.
    test_path='/home/wu_tian_ci/drl_project/txtans/'+store_name+'test.txt'
    train_path='/home/wu_tian_ci/drl_project/txtans/'+store_name+'train.txt'
    # if not os.path.exists(test_path):
    f=open(test_path,'w')
    f.close()
    # if not os.path.exists(train_path):
    f=open(train_path,'w')
    f.close()

    learning_rate=[]
    for i in range(epochs):
        net.train()
        max_loss_1 = 0.
        min_loss_1 = np.inf
        max_loss_2 = 0.
        min_loss_2 = np.inf
        ll1,ll2, t = 0,0,0
        for v,w,x,y,z in tqdm(train_iter):
            v=v.to(device)
            w=w.to(device)
            x=x.to(device)
            y=y.to(device)
            z=z.to(device)
            trainer.zero_grad()
            res=net(v,w,x)
            click_s=res[0].to(device)
            click_=res[1].to(device)
            click_[:,:,0:15]=res[1][:,:,0:15]*16
            click_[:,:,15:]=res[1][:,:,15:]*9
            click_s[:,:,0:]=res[0][:,:,0:]*5
            click_.to(device)
            click_s.to(device)
            l1=loss(click_s,y)
            l2=loss(click_,z)
            l = 0.1*l1+l2*2
            if l < best_loss:
                best_loss = l
                torch.save(net.state_dict(), '/home/wu_tian_ci/drl_project/txtans/'+'2TiDE.pt')
            max_loss_1 = max(max_loss_1, torch.max(max_loss(res[0],y)).cpu().detach().numpy())
            min_loss_1 = min(min_loss_1, torch.min(max_loss(res[0],y)).cpu().detach().numpy())
            max_loss_2 = max(max_loss_2, torch.max(max_loss(res[1],z)).cpu().detach().numpy())
            min_loss_2 = min(min_loss_2, torch.min(max_loss(res[1],z)).cpu().detach().numpy())
            # ll += l
            ll1+=l1
            ll2+=l2
            t += 1
            l.backward()
            trainer.step()
        p_lr=trainer.param_groups[0]['lr']
        schedule.step()
        t_max_loss_1, t_min_loss_1,t_max_loss_2, t_min_loss_2,t_ll1,t_ll2= test_acc(net, test_iter,device)
        print(f'train epochs {i}')
        # print('max l1 loss\tmin l1 loss\tave l1 loss\tmax l2 loss\tmin l2 loss\tave l2 loss\tlr')
        with open(train_path,'a') as f:
            # print(type(ll2/t),type(max_loss_1),type(min_loss_1),type(ll1/t),type(max_loss_2),type(min_loss_2))
            f.write('epochs :'+str(i)+' ')
            f.write(str(max_loss_1)+' '+str(min_loss_1)+
            ' '+str((ll1/t).cpu().detach().numpy())+' '+str(max_loss_2)+' '+str(min_loss_2)+
            ' '+str((ll2/t).cpu().detach().numpy())+' '+str(p_lr)+'\n')
            f.close()
        with open(test_path,'a') as f:
            f.write('epochs :'+str(i)+' ')
            f.write(str(t_max_loss_1)+' '+str(t_min_loss_1)+
            ' '+str(t_ll1)+' '+str(t_max_loss_2)+' '+str(t_min_loss_2)+
            ' '+str(t_ll2)+' '+str(p_lr)+'\n')
            f.close()
        # print(f'{max_loss_1}\t{min_loss_1}\t{ll1/t}\t{max_loss_2}\t{min_loss_2}\t{ll2/t}\t{p_lr}')
        # print(f'{t_max_loss_1}\t{t_min_loss_1}\t{t_ll1}\t{t_max_loss_2}\t{t_min_loss_2}\t{t_ll2}\t{p_lr}')
        # train_list[0].append(i),train_list[1].append(max_loss_1),train_list[2].append(min_loss_1)
        # train_list[3].append(ll1),train_list[4].append(max_loss_2),train_list[5].append(min_loss_2),train_list[6].append(ll2),train_list[7].append(p_lr)
        # test_list[0].append(i),test_list[1].append(t_max_loss_1),test_list[2].append(t_min_loss_1)
        # test_list[3].append(t_ll1),test_list[4].append(t_max_loss_2),test_list[5].append(t_min_loss_2),test_list[6].append(t_ll2),test_list[7].append(p_lr)
        # animator1.add_data(max_loss, min_loss,ll / t,lr=p_lr*100)
        # animator2.add_data(t_max, t_min, t_ave,p_lr*100)
    # animator1.draw()
    # animator2.draw()
    # with open('/home/wu_tian_ci/drl_project/tide/record_train.txt','w') as f:
    #     for info in train_list:
    #         contents=''
    #         for i in info:
    #             contents+=str(i)
    #         contents+='\n'
    #         f.write(contents)
    # with open('/home/wu_tian_ci/drl_project/tide/record_train.txt','w') as f:
    #     for info in train_list:
    #         contents=''
    #         for i in info:
    #             contents+=str(i)
    #         contents+='\n'
    #         f.write(contents)

    net.load_state_dict(torch.load('/home/wu_tian_ci/drl_project/txtans/'+'2TiDE.pt'))
    print('test acc ')
    print(test_acc(net, test_iter,device),end=' ')


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    # train_od = dataset_loader('D:\\eyedata\\mixed\\pretrain\\s1\\train')
    # test_od = dataset_loader('D:\\eyedata\\mixed\\pretrain\\s1\\test')
    # train_iter = torch.utils.data.DataLoader(dataset=train_od, shuffle=False, batch_size=256)
    # test_iter = torch.utils.data.DataLoader(dataset=test_od, shuffle=False, batch_size=256)
    # train_od2=dataset_loader2('D:\\eyedata\\mixed\\pretrain2\\s1\\train')
    # test_od2 = dataset_loader2('D:\\eyedata\\mixed\\pretrain2\\s1\\test')
    # train_iter2 = torch.utils.data.DataLoader(dataset=train_od2, shuffle=False, batch_size=256)
    # test_iter2 = torch.utils.data.DataLoader(dataset=test_od2, shuffle=False, batch_size=256)
    # net = MLPDRL(60, 45)

    train_od3=tdataset_loader('/home/wu_tian_ci/eyedata/mixed/pretrain10/train')
    train_iter3=torch.utils.data.DataLoader(dataset=train_od3,shuffle=True,batch_size=10240,num_workers=24)
    test_od3=tdataset_loader('/home/wu_tian_ci/eyedata/mixed/pretrain10/test')
    test_iter3=torch.utils.data.DataLoader(dataset=test_od3,shuffle=True,batch_size=512,num_workers=24)
    # for x,y,z in train_od3:
    #     print(x,y,z)
    tide=TiDE(360,120,120,90,45,30,15,100,7)
    tide.apply(weight_init)
    # train_epoch(train_iter, test_iter, net, 0.01, 100,
    # device)
    # train_epoch(train_iter3, test_iter3, net, 0.07, 200, device,False)
    train_epoch(train_iter3,test_iter3,tide,0.1,1000,device,'3')
    train_epoch(train_iter3,test_iter3,tide,0.01,1000,device,'4',True)
