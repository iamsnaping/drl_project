import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):

    def __init__(self,i_put,c_a,c_b):
        super(ResBlock, self).__init__()
        self.s1=nn.Sequential(nn.Linear(i_put,i_put*2),nn.ReLU(),nn.Linear(i_put*2,i_put))
        self.norm=nn.Sequential(nn.LayerNorm(i_put+c_a+c_b))
    def forward(self,x1,a,b):
        x2=self.s1(x1)
        return self.norm(torch.concat([x1+x2,a,b],dim=-1))


class MyModule2(nn.Module):
    def __init__(self,input_num,n_s,n_a,num_layer):
        super(MyModule2, self).__init__()
        self.net=nn.ModuleList()
        i_n=input_num
        for i in range(num_layer):
            self.net.append(ResBlock(i_n,n_s,n_a))
            i_n=i_n+n_s+n_a

    def forward(self,x,state,action):
        for layer in self.net:
            x=layer(x,state,action)
        return x

class ResBlock2(nn.Module):

    def __init__(self,i_put,output):
        super(ResBlock2, self).__init__()
        self.s1=nn.Sequential(nn.Linear(i_put,i_put*2),nn.ReLU(),nn.Linear(i_put*2,output))
        self.s2=nn.Linear(i_put,output)
        self.norm=nn.Sequential(nn.LayerNorm(output))
    def forward(self,x):
        return self.norm(self.s1(x)+self.s2(x))
        



class DQNNET(nn.Module):
    def __init__(self,input_num ,output_num,num_layer,n_s, n_a,n_1,n_2,n_3):
        super(DQNNET, self).__init__()
        self.s1=nn.Sequential(nn.Linear(input_num,output_num),nn.LayerNorm(output_num),nn.ReLU())
        self.net=MyModule2(output_num,n_s,n_a,num_layer)
        out_c=output_num+num_layer*(n_s+n_a)
        self.fp1=ResBlock2(out_c,n_1)
        self.fp2=ResBlock2(out_c,n_2)
        self.fp3=ResBlock2(out_c,n_3)
        self.meta1=nn.Linear(n_a,n_1)
        self.meta2=nn.Linear(n_a,n_2)
        self.meta3=nn.Linear(n_a,n_3)
        

    def forward(self,state,action):
        # print(action.shape,state.shape)
        s1=self.s1(torch.concat([state,action],dim=-1))
        n=self.net(s1,state,action)
        x=self.fp1(n)+self.meta1(action)
        y=self.fp2(n)+self.meta2(action)
        z=self.fp3(n)+self.meta3(action)
        return x,y,z
    
    def act(self,state,action):
        state_tensor=torch.as_tensor(state,dtype=torch.float32)
        action_tensor=torch.as_tensor(action,dtype=torch.float32)
        q_value=self(state_tensor,action_tensor)
        max_q_idx=torch.argmax(q_value)
        action=max_q_idx.detach().item()
        return action
        



