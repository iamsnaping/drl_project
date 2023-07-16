import torch
from torch import nn
import numpy as np
import random

random.seed(1114)
np.random.seed(1114)
torch.manual_seed(1114)


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

class ReplayBuffer:

    def __init__(self, n_s, n_a, batch_size=256, memory_size=int(1e6)):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memo_s = np.empty(shape=(memory_size, n_s), dtype=np.float32)
        self.memo_s_ = np.empty(shape=(memory_size, n_s), dtype=np.float32)
        self.memo_a = np.empty(shape=(memory_size, n_a), dtype=np.float32)
        self.memo_r = np.empty(memory_size)
        self.memo_d = np.empty(memory_size, dtype=np.uint8)
        self.t_max = 0
        self.t_memo = 0

    def add(self, s, s_, d, a, r):
        self.memo_s[self.t_memo] = s
        self.memo_s_[self.t_memo] = s_
        self.memo_r[self.t_memo] = r
        self.memo_a[self.t_memo] = a
        self.memo_d[self.t_memo] = d
        self.t_max = max(self.t_memo + 1, self.t_max)
        self.t_memo = (1 + self.t_memo) % self.memory_size

    def sample(self):
        if self.t_max > self.batch_size:
            idxes = random.sample(range(self.t_max), self.batch_size)
        else:
            idxes = range(0, self.t_max)
        batch_s = []
        batch_s_ = []
        batch_r = []
        batch_a = []
        batch_d = []
        for idx in idxes:
            batch_s.append(self.memo_s[idx])
            batch_s_.append(self.memo_s_[idx])
            batch_r.append(self.memo_r[idx])
            batch_d.append(self.memo_d[idx])
            batch_a.append(self.memo_a[idx])
        tensor_batch_s = torch.as_tensor(np.asarray(batch_s), dtype=torch.float32)
        tensor_batch_s_ = torch.as_tensor(np.asarray(batch_s_), dtype=torch.float32)
        tensor_batch_d = torch.as_tensor(np.asarray(batch_d), dtype=torch.int64).unsqueeze(-1)
        tensor_batch_r = torch.as_tensor(np.asarray(batch_r), dtype=torch.float32).unsqueeze(-1)
        tensor_batch_a = torch.as_tensor(np.asarray(batch_a), dtype=torch.float32)
        return tensor_batch_a, tensor_batch_s, tensor_batch_s_, tensor_batch_d, tensor_batch_r


class ActorNet(nn.Module):
    def __init__(self, n_s, n_a):
        super(ActorNet, self).__init__()
        self.s1 = nn.Sequential(nn.Linear(n_s,16),nn.LayerNorm(16),nn.ReLU())
        self.s2=nn.Sequential(nn.Linear(16,96),nn.LayerNorm(96),nn.ReLU(),nn.Linear(96,192),nn.LayerNorm(192),nn.ReLU()
                              ,nn.Linear(192,384),nn.LayerNorm(384),nn.ReLU(),nn.Linear(384,80),nn.LayerNorm(80),nn.ReLU())

        self.s3=nn.Sequential(nn.Linear(96,16),nn.LayerNorm(16),nn.ReLU())
        self.s4=nn.Sequential(nn.Linear(32,n_a),nn.ReLU())
        self.is_noise = True
        # self.apply(weight_init)

    def forward(self, state):
        s1=self.s1(state)  #16
        s2=self.s2(s1)     #60
        s3=torch.concat([s1,s2],dim=-1)
        s4=self.s3(s3)  #16
        s5=torch.concat([s1,s4],dim=-1)
        return self.s4(s5)

    def act(self, state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        action = self(state_tensor)
        return action.detach().numpy().ravel()


class CriticNet(nn.Module):
    def __init__(self, n_s, n_a):
        super(CriticNet, self).__init__()
        self.s1_s = nn.Sequential(nn.Linear(n_s, 96), nn.LayerNorm(96), nn.ReLU())
        self.s1_a = nn.Sequential(nn.Linear(n_a, 96), nn.LayerNorm(96), nn.ReLU())
        self.s2 = nn.Sequential(nn.Linear(192, 384), nn.LayerNorm(384), nn.ReLU())
        self.s3 = nn.Sequential(nn.Linear(384, 1))
        # self.apply(weight_init)

    def forward(self, state, action):
        s1 = self.s1_s(state)
        a1 = self.s1_a(action)
        as1 = torch.concat([a1, s1], dim=-1)
        as2 = self.s2(as1)
        return self.s3(as2)


class Critic_Net:
    def __init__(self, n_s, n_a, lr, tao):
        self.target = CriticNet(n_s, n_a)
        self.online = CriticNet(n_s, n_a)
        self.target.apply(weight_init)
        self.online.apply(weight_init)
        self.trainer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.tao = tao

    def update_params(self):
        for param_target, param in zip(self.target.parameters(), self.online.parameters()):
            param_target.data.copy_(param_target * (1 - self.tao) + param * self.tao)

    def print_params(self):
        for param_target, param in zip(self.target.parameters(), self.online.parameters()):
            if len(param.data) < 5:
                print(f'actor param {param.data}')
                print(f'actor target {param_target.data}')


class Actor_Net:
    def __init__(self, n_s, n_a, lr, tao, GAMMA):
        self.target = ActorNet(n_s, n_a)
        self.online = ActorNet(n_s, n_a)
        self.target.apply(weight_init)
        self.online.apply(weight_init)
        self.online.is_noise = True
        self.target.is_noise = False
        self.trainer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.tao = tao
        self.GAMMA = GAMMA

    def update_params(self):
        for param_target, param in zip(self.target.parameters(), self.online.parameters()):
            param_target.data.copy_(param_target * (1 - self.tao) + param * self.tao)

    def print_params(self):
        for param_target, param in zip(self.target.parameters(), self.online.parameters()):
            if len(param.data) <5:
                print(f'actor param {param.data}')
                print(f'actor target {param_target.data}')
            # param_target.data.copy_(param_target * (1 - self.tao) + param * self.tao)

class Agent:
    def __init__(self, n_s, n_a, tao, lr=0.03, GAMMA=0.9):
        self.memo = ReplayBuffer(n_s, n_a)
        self.actor_net = Actor_Net(n_s, n_a, tao, lr, GAMMA)
        self.critic_net = Critic_Net(n_s, n_a, lr, tao)
