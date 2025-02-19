
import os
import sys
import warnings
# warnings.filterwarnings('error')

sys.path.append('/home/wu_tian_ci/drl_project')
sys.path.append('/home/wu_tian_ci/drl_project/mymodel/pretrain_model')
sys.path.append('/home/wu_tian_ci/drl_project/mymodel/pretrain_model/mlpdrl.py')
# print(sys.path)
# os.chdir(sys.path[0])
import gym
import torch
import random
import torch.nn as nn
import collections
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Normal
from mymodel.pretrain_model.mlpdrl import*
from sac_environment import *


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

class ReplayBeffer():
    def __init__(self, buffer_maxlen):
        self.buffer = collections.deque(maxlen=buffer_maxlen)

    def push(self, data):
        self.buffer.append(data)

    def sample(self, batch_size):
        state_e = []
        state_c = []
        action_list = []
        reward_list = []
        next_state_e = []
        next_state_c = []
        done_list = []

        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            se, sc, al,rl,nse,nsc,dl = experience
            # state, action, reward, next_state, done
            state_e.append(se)
            state_c.append(sc)
            action_list.append(al)
            reward_list.append(rl)
            next_state_e.append(nse)
            next_state_c.append(nsc)
            done_list.append(dl)

        return torch.tensor(np.array(state_e),dtype=torch.float32).to(device), \
               torch.tensor(np.array(state_c),dtype=torch.float32).to(device), \
               torch.tensor(np.array(action_list),dtype=torch.float32).to(device),\
               torch.tensor(np.array(reward_list),dtype=torch.float32).to(device), \
               torch.tensor(np.array(next_state_e),dtype=torch.float32).to(device), \
               torch.tensor(np.array(next_state_c),dtype=torch.float32).to(device), \
               torch.tensor(np.array(done_list),dtype=torch.float32).to(device)

    def buffer_len(self):
        return len(self.buffer)


# Value Net
class ValueNet(nn.Module):
    def __init__(self, state_dim,action_dim, edge=3e-3):
        super(ValueNet, self).__init__()
        self.net=nn.Sequential(nn.Linear(state_dim,256),nn.LayerNorm(256),nn.ReLU(),
        nn.Linear(256,256),nn.LayerNorm(256),nn.ReLU(),
        nn.Linear(256,action_dim))
        self.s_meta = nn.Linear(state_dim, action_dim)

    def forward(self, eye_list,click_):
        state=torch.concat([eye_list,click_],dim=-1)
        return self.net(state)+self.s_meta(state)


# Soft Q Net
class SoftQNet(nn.Module):
    def __init__(self, state_dim, action_dim, edge=3e-3):
        super(SoftQNet, self).__init__()
        self.net=nn.Sequential(nn.Linear(state_dim+action_dim,256),nn.LayerNorm(256),nn.ReLU(),
        nn.Linear(256,256),nn.LayerNorm(256),nn.ReLU(),
        nn.Linear(256,action_dim))
        self.s_meta = nn.Linear(state_dim+action_dim, action_dim)

    def forward(self, eye_list,click_, action):
        state = torch.cat([eye_list,click_, action], -1)
        return self.net(state)+self.s_meta(state)


# Policy Net
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-2, log_std_max=2, edge=3e-3):
        super(PolicyNet, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.mean_net=MLPDRL2(13,15,8,10,3)
        self.log_std_net=MLPDRL2(13,15,5,10,3)

        # self.linear1 = nn.Linear(state_dim, 256)
        # self.linear2 = nn.Linear(256, 256)

        # self.mean_linear = nn.Linear(256, action_dim)
        # self.mean_linear.weight.data.uniform_(-edge, edge)
        # self.mean_linear.bias.data.uniform_(-edge, edge)

        # self.log_std_linear = nn.Linear(256, action_dim)
        # self.log_std_linear.weight.data.uniform_(-edge, edge)
        # self.log_std_linear.bias.data.uniform_(-edge, edge)

    def forward(self, eye_list,click_):
        mean = self.mean_net(eye_list,click_)
        log_std = self.log_std_net(eye_list,click_)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def action(self, eye_list,click_):
        mean, log_std = self.forward(eye_list,click_)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.sample()
        # action = torch.tanh(z).detach().cpu().numpy()
        action=z.cpu().detach().numpy()
        return action

    # Use re-parameterization tick
    def evaluate(self, eye_list,click_, epsilon=1e-6):
        mean, log_std = self.forward(eye_list,click_)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = Normal(0, 1)

        z = noise.sample()
        action = torch.tanh(mean + std * z.to(device))
        log_prob = normal.log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)

        return action, log_prob


class SAC:
    def __init__(self, env, gamma, tau, buffer_maxlen, value_lr, q_lr, policy_lr):

        self.env = env
        self.state_dim = 13
        self.action_dim = 3

        # hyperparameters
        self.gamma = gamma
        self.tau = tau

        # initialize networks
        self.value_net = ValueNet(self.state_dim,self.action_dim).to(device)
        self.target_value_net = ValueNet(self.state_dim,self.action_dim).to(device)
        self.q1_net = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.q2_net = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.policy_net = PolicyNet(self.state_dim, self.action_dim).to(device)
        # Load the target value network parameters
        # for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
        #     target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            # Initialize the optimizer
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        # Initialize thebuffer
        self.buffer = ReplayBeffer(buffer_maxlen)

    def get_action(self, eye_list,click_):
        action = self.policy_net.action(eye_list,click_)
        return action

    def update(self, batch_size):
        state_e,state_c, action, reward, next_state_e,next_state_c, done = self.buffer.sample(batch_size)
        new_action, log_prob = self.policy_net.evaluate(state_e,state_c)

        # V value loss
        value = self.value_net(state_e,state_c)
        new_q1_value = self.q1_net(state_e,state_c, new_action)
        new_q2_value = self.q2_net(state_e,state_c, new_action)
        next_value = torch.min(new_q1_value, new_q2_value) - log_prob
        # print(value.shape,new_q1_value.shape,new_q2_value.shape,log_prob.shape)
        value_loss = F.mse_loss(value, next_value.detach())

        # Soft q  loss
        q1_value = self.q1_net(state_e,state_c, action)
        q2_value = self.q2_net(state_e,state_c, action)
        target_value = self.target_value_net(next_state_e,next_state_c)
        target_q_value = reward + done * self.gamma * target_value
        # try:
        q1_value_loss = F.mse_loss(q1_value, target_q_value.detach())
        q2_value_loss = F.mse_loss(q2_value, target_q_value.detach())
        # if q2_value.shape!=target_q_value.shape:
        #     print(target_value.shape,target_value.shape,target_q_value.shape,reward.shape,reward.shape)
        # except warnings as e:
            # print(target_value.shape,target_value.shape)

        # Policy loss
        policy_loss = (log_prob - torch.min(new_q1_value, new_q2_value)).mean()

        # Update Policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update v
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update Soft q
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q1_value_loss.backward()
        q2_value_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)



def main(agent, Episode, batch_size):
    Return = []
    env=EyeEnvironment('/home/wu_tian_ci/eyedata/seperate/01/1','1')
    f=open('/home/wu_tian_ci/drl_project/mymodel/sac/sac_result/result.txt','w')
    for episode in range(Episode):
        env.load_dict()
        env.get_file()
        flag,last_state=env.reset()
        scores=0.
        kflag=False
        last_goal=last_state[1].copy()
        t=0
        new_sores=[0.,0.,0.]
        while not kflag:
            eye_list=torch.tensor([last_state[0]],dtype=torch.float32).unsqueeze(0).to(device)
            click_=torch.tensor([last_goal],dtype=torch.float32).unsqueeze(0).to(device)
            my_action=agent.get_action(eye_list,click_)
            s_action=np.squeeze(my_action)
            kflag,next_state=env.act()
            if len(next_state[1])<3:
                break
            reward1=5-(s_action[0]-last_state[1][0])**2
            reward2=5-(s_action[1]-last_state[1][1])**2
            reward3=1-(s_action[2]-last_state[1][2])**2
            reward=env.get_reward(s_action)
            new_sores[0]+=(reward1)
            new_sores[1]+=(reward2)
            new_sores[2]+=(reward3)
            isdone=0 if kflag else 1
            agent.buffer.push(([last_state[0]],[last_goal],[s_action],
            [[reward,reward,reward]],[next_state[0]],[next_state[1]],[[isdone,isdone,isdone]]))
            last_state=next_state
            scores+=reward1+reward2+reward3
            t+=1
            if last_state[1][2]==0:
                last_goal=last_goal.copy()
                env.get_goal()
            if kflag:
                break
            if agent.buffer.buffer_len()>512:
                agent.update(batch_size)
        flag,last_state=env.reset()
        if flag:
            env.refresh()
        print(f'eposide{episode+1} ave reward {scores/t} t {t} new scores {new_sores[0]/t} {(new_sores[1]/t)} {new_sores[2]/t}')
        with open('/home/wu_tian_ci/drl_project/mymodel/sac/sac_result/result.txt','a') as f:
            f.write('eposide: '+str(episode+1)+' '+str(scores/t)+' '+str(t)+' '+str(new_sores[0]/t)+' '+str(new_sores[1]/t)+' '+str(new_sores[2]/t)+'\n')
        # print("episode:{}, Return:{}, buffer_capacity:{}".format(episode, scores/t, agent.buffer.buffer_len()))
        Return.append(scores)
        score = 0
    # env.close()
    plt.plot(Return)
    plt.ylabel('Return')
    plt.xlabel("Episode")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    env = gym.make("Pendulum-v1")
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # Params
    tau = 0.01
    gamma = 0.99
    q_lr = 0.07
    value_lr = 0.07
    policy_lr = 0.01
    buffer_maxlen = 50000

    Episode = 1000
    batch_size = 512

    agent = SAC(env, gamma, tau, buffer_maxlen, value_lr, q_lr, policy_lr)
    agent.policy_net.mean_net.load_state_dict(torch.load('/home/wu_tian_ci/drl_project/mymodel/pretrain_data/2/MLPDRL.pt'))
    agent.policy_net.log_std_net.apply(weight_init)
    agent.value_net.apply(weight_init)
    agent.target_value_net.apply(weight_init)
    agent.q1_net.apply(weight_init)
    agent.q2_net.apply(weight_init)
    main(agent, Episode, batch_size)