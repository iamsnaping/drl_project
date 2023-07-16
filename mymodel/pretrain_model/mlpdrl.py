import torch
from torch import nn
from torch.nn import functional as F


# class MLPDRL(nn.Module):
#     def __init__(self,n_s,n_a):
#         super(MLPDRL, self).__init__()
#         self.s1 = nn.Sequential(nn.Linear(n_s+n_a, 80), nn.LayerNorm(80), nn.ReLU())
#         self.s2 = nn.Sequential(nn.Linear(92, 96), nn.LayerNorm(96), nn.ReLU(), nn.Linear(96, 192), nn.LayerNorm(192),
#                                 nn.ReLU(), nn.Linear(192, 384), nn.LayerNorm(384), nn.ReLU(), nn.Linear(384, 80),
#                                 nn.LayerNorm(80), nn.ReLU())
#         self.s3 = nn.Sequential(nn.Linear(92, 60), nn.LayerNorm(60), nn.ReLU())
#         self.s4 = nn.Sequential(nn.Linear(72, 8), nn.LayerNorm(8),nn.ReLU())
#         # self.s4_2=nn.Sequential(nn.Linear(2,8),nn.LayerNorm(8),nn.ReLU())
#         self.s5=nn.Sequential(nn.Linear(20,n_a))
#         self.a_meta=nn.Sequential(nn.Linear(n_a,n_a),nn.LayerNorm(2),nn.ReLU())
#         self.as_meta=nn.Sequential(nn.Linear(n_s,10),nn.LayerNorm(10),nn.ReLU())
#
#     def forward(self,state,action):
#         a_meta=self.a_meta(action)
#         as_meta=self.as_meta(state)
#         s1 = self.s1(torch.concat([state,action],dim=-1))
#         s2 = self.s2(torch.concat([s1,action,as_meta],dim=-1))
#         s4=self.s3(torch.concat([s2,action,as_meta],dim=-1))
#         s6 = self.s4(torch.concat([state+s4,action,as_meta],dim=-1))
#         return self.s5(torch.concat([s6,a_meta,as_meta],dim=-1))+a_meta


class MyModule(nn.Module):
    def __init__(self, in_c, out_c, num_layers):
        super(MyModule, self).__init__()
        self.net = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_c, out_c), nn.LayerNorm(out_c), nn.ReLU()) for i in range(num_layers)])
        self.nums_layers = num_layers

    def forward(self, x, a_meta, as_meta):
        i = 0
        for layer in self.net:
            x = layer(x)
            if i != self.nums_layers - 1:
                x=as_meta+x+a_meta
                # x = torch.concat([as_meta, x, a_meta], dim=-1)
            i += 1
        return x

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

class MLPDRL2(nn.Module):
    def __init__(self,input_num ,output_num,num_layer,n_s, n_a):
        super(MLPDRL2, self).__init__()
        self.s1=nn.Sequential(nn.Linear(input_num,output_num),nn.LayerNorm(output_num),nn.ReLU())
        self.net=MyModule2(output_num,n_s,n_a,num_layer)
        out_c=output_num+num_layer*(n_s+n_a)
        self.s2=nn.Sequential(nn.Linear(out_c,n_a))
        self.a_meta=nn.Sequential(nn.Linear(n_a,n_a),nn.LayerNorm(n_a),nn.ReLU())

    def forward(self,state,action):
        s1=self.s1(torch.concat([state,action],dim=-1))
        n=self.net(s1,state,action)
        return self.s2(n)+self.a_meta(action)



class MLPDRL(nn.Module):
    def __init__(self, n_s, n_a):
        super(MLPDRL, self).__init__()
        self.a_meta = nn.Sequential(nn.Linear(n_a, 3), nn.LayerNorm(3), nn.ReLU())
        self.as_meta = nn.Sequential(nn.Linear(n_s + n_a, 45), nn.LayerNorm(45), nn.ReLU())
        self.mul = MyModule(135, 45, 5)
        self.s1 = nn.Sequential(nn.Linear(48, 3))
        self.trans_input=nn.Sequential(nn.Linear(150,135),nn.LayerNorm(135),nn.ReLU())

    def forward(self, state, action):
        a_meta = self.a_meta(action)
        as_meta = self.as_meta(torch.concat([state, action], dim=-1))
        # as_meta=self.as_meta(state)
        s1 = self.mul(self.trans_input(torch.concat([state, action,as_meta], dim=-1)), action, as_meta)
        return self.s1(torch.concat([s1 + as_meta, a_meta], dim=-1)) + a_meta
