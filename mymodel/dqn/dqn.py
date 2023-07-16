import random

import gym
import numpy as np
import torch
from torch import nn
from agent import *
import matplotlib.pyplot as plt
import matplotlib

# env = gym.make('CartPole-v1',render_mode='human')
env = gym.make('CartPole-v1')
env2=gym.make('CartPole-v1',render_mode='human')
# env = gym.make('CartPole-v1')

s, _ = env.reset()
n_episode = 5000
n_time_step = 1000
TARGET_UPDATE_FREQUENCY = 10
EPSILON_DECAY = 100000
EPSILON_START = 1.0
EPSILON_END = 0.02
REWARD_BUFFER = np.empty(shape=n_episode)

n_state = len(s)
n_action = env.action_space.n
print(n_state,n_action)
agent = Agent(n_state, n_action)
loss = nn.MSELoss(reduction='mean')
lr=1e-3
trainer = torch.optim.RMSprop(agent.online_net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-8)
# trainer=torch.optim.Adam(agent.online_net.parameters(),lr=lr)
# scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(trainer,T_max=n_episode)
episodes=[]
scores=[]
for episode_i in range(n_episode):
    episode_reward = 0.
    for step_i in range(n_time_step):
        eposilon = np.interp(episode_i * n_time_step + step_i, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
        random_sample = random.random()
        if random_sample < eposilon:
            a = env.action_space.sample()
        else:
            print('act')
            a = agent.online_net.act(s)  # TODO
        s_, r, done, _, info = env.step(a)
        agent.memo.add_memo(s=s, r=r, a=a, done=done, s_=s_)
        s = s_
        episode_reward += r

        if done:
            s,_ = env.reset()
            REWARD_BUFFER[episode_i] = episode_reward
            break

        batch_s, batch_a, batch_r, batch_done, batch_s_ = agent.memo.sample()
        target_q_value = agent.target_net(batch_s_)
        max_target_q_value = target_q_value.max(dim=1, keepdim=True)[0]
        y = (1 - batch_done) * agent.GAMMA * max_target_q_value + batch_r

        q_values = agent.online_net(batch_s)
        a_q_values = torch.gather(input=q_values, dim=1, index=batch_a)

        # loss = nn.functional.smooth_l1_loss(y, a_q_values)

        # print(y,a_q_values)
        # print(y,a_q_values)
        l=loss(y,a_q_values)
        trainer.zero_grad()
        # agent.optimizer.zero_grad()
        l.backward()
        # loss.backward()
        trainer.step()
        # agent.optimizer.step()
    # scheduler.step()
    if episode_i % TARGET_UPDATE_FREQUENCY == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())

        print(f'Episode {episode_i}')
        print(f'Avg Reward {np.mean(REWARD_BUFFER[:episode_i])}')
    scores.append(np.mean(REWARD_BUFFER[:episode_i]))
    episodes.append(episode_i)
    # if np.mean(REWARD_BUFFER[:episode_i])>100:
    #     score = 0.
    #     _s, _ = env2.reset()
    #     while True:
    #         a = agent.online_net.act(_s)
    #         _s, _r, _done, __, _infor = env2.step(a)
    #         score += _r
    #         # print(_done)
    #         if _done:
    #             # print('end')
    #             scores.append(score)
    #             episodes.append(episode_i)
    #             env2.reset()
    #             break


"""
font:设置中文
"""
matplotlib.rcParams['font.family'] = ['Heiti TC']

# 绘图，设置(label)图例名字为'小龙虾价格'，显示图例plt.legend()
plt.plot(episode_i,scores,label='小龙虾价格')

# x轴标签
plt.xlabel('月份')
# y轴标签
plt.ylabel('价格')

# 图表标题
plt.title('小龙虾全年价格')

# 显示图例
plt.legend()
# 显示图形
plt.show()



