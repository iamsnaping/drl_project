import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
from tide import *


def trans_num(nums):
    number=0.
    flag=1
    # print(nums)
    nums=nums.replace('\n','')
    for i in nums:
        if i=='':
            continue
        if i=='-':
            flag=-1
        else:
            number*=10
            number+=int(i)
    return flag*number


def trans_nums(nums):
    n = []
    for num in nums:
        if num=='':
            continue
        n.append(int(num))
        # n.append(trans_num(num))
    return n


class tdataset_loader(Dataset):
    def __init__(self, data_path):
        super(tdataset_loader, self).__init__()
        d_path=os.path.join(data_path,'*.txt')
        self.data_path = glob.glob(d_path)

    def __getitem__(self, item):
        txt_path = self.data_path[item]
        nums = []
        with open(txt_path) as f:
            lines = f.readlines()
            for line in lines:
                line=line.replace('\n','')
                nums.append(trans_nums(line.split(' ')))
        eye_list=torch.tensor(nums[0],dtype=torch.float32).reshape(1,-1)
        click_past=nums[1]
        click_past_s=nums[2]
        click_=torch.tensor(nums[3],dtype=torch.float32).reshape(1,-1)
        click_s=torch.tensor(nums[4],dtype=torch.float32).reshape(1,-1)
        # print(scheme)
        scheme=torch.tensor(nums[5],dtype=torch.float32).reshape(1,-1)
        click_p=torch.concat([torch.tensor(click_past,dtype=torch.float32).reshape(1,-1),torch.tensor(click_past_s,dtype=torch.float32).reshape(1,-1)],dim=-1)
        return click_p,scheme,eye_list,click_s,click_


    def __len__(self):
        return len(self.data_path)




if __name__=='__main__':
    # od=dataset_loader('D:\\eyedata\\mixed\\pretrain\\s1\\train')
    # dataloader = torch.utils.data.DataLoader(dataset=od, shuffle=False, batch_size=1)
    tdl=tdataset_loader('/home/wu_tian_ci/eyedata/mixed/pretrain7/train')
    dl=torch.utils.data.DataLoader(dataset=tdl,shuffle=True,batch_size=1)
    tide=TiDE(180,60,60,90,45,30,15,100,5)
    for v,w,x,y,z in dl:
        print(v.shape)
        # print(f'v {v} len {len(v[0][0])}')
        # print(f'w {w} len {len(w[0][0])}')
        # print(f'x {x} len {len(x[0][0])}')
        # print(f'y {y} len {len(y[0][0])}')
        # print(f'z {z} len {len(z[0][0])}')
        print(tide(v,w,x))
    # for x,y,z in dataloader:
    #     # pass
    #     print(z)
        # print(x,y,z)
        # break
