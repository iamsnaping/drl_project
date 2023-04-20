import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random


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
        n.append(trans_num(num))
    return n


class dataset_loader(Dataset):
    def __init__(self, data_path):
        super(dataset_loader, self).__init__()
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
        s = nums[0]
        begin_goal =nums[1][0:2]
        goal=nums[1][2:4]
        # s_=nums[2]
        return torch.tensor(s, dtype=torch.float32).reshape(1, -1), torch.tensor(begin_goal, dtype=torch.float32).reshape(1, -1),\
               torch.tensor(goal,dtype=torch.float32).reshape(1,-1)
            # ,torch.tensor(s_,dtype=torch.float32).reshape(1,-1)

    def __len__(self):
        return len(self.data_path)



if __name__=='__main__':
    od=dataset_loader('D:\\eyedata\\mixed\\pretrain\\s1\\train')
    dataloader = torch.utils.data.DataLoader(dataset=od, shuffle=False, batch_size=1)
    for x,y,z in dataloader:
        # pass
        print(len(x[0][0]))
        # print(x,y,z)
        # break
