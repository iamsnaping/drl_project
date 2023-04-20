import torch
from torch import nn
from torch.nn import functional as F


class MLPDRL(nn.Module):
    def __init__(self,n_s,n_a):
        super(MLPDRL, self).__init__()
        self.s1 = nn.Sequential(nn.Linear(n_s, 80), nn.LayerNorm(80), nn.ReLU())
        self.s2 = nn.Sequential(nn.Linear(80, 96), nn.LayerNorm(96), nn.ReLU(), nn.Linear(96, 192), nn.LayerNorm(192),
                                nn.ReLU(), nn.Linear(192, 384), nn.LayerNorm(384), nn.ReLU(), nn.Linear(384, 80),
                                nn.LayerNorm(80), nn.ReLU())
        self.s3 = nn.Sequential(nn.Linear(80, 60), nn.LayerNorm(60), nn.ReLU())
        self.s4 = nn.Sequential(nn.Linear(60, 8), nn.LayerNorm(8),nn.ReLU())
        self.s4_2=nn.Sequential(nn.Linear(2,8),nn.LayerNorm(8),nn.ReLU())
        self.s5=nn.Sequential(nn.Linear(8,n_a),nn.ReLU())

    def forward(self,state,action):
        s1 = self.s1(state)  # 16
        s2 = self.s2(s1)  # 60
        # s3 = torch.concat([s1, s2], dim=-1)
        # s4 = self.s3(s3)  # 16
        s4=self.s3(s2)
        # s5 = torch.concat([s1, s4], dim=-1)
        s6 = self.s4(state+s4) #6
        # s6=self.s4(torch.concat([state,s4],dim=-1))
        s7=self.s4_2(action)
        return self.s5(s7+s6)
        # return self.s5(torch.concat([s6,s7],dim=-1))
        # s7=torch.concat([action,s6],dim=-1)
        # return self.s5(s7)
