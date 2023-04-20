from torch import nn
import torch
import numpy as np
import gym
from agent import *
import matplotlib.pyplot as plt
import matplotlib
from enviroment import *

env = gym.make('LunarLander-v2', render_mode='human')
# env=gym.make('BipedalWalkerHardcore-v3')
loss = nn.MSELoss()


def scale_action(action, high, low):
    action = np.clip(action, -1, 1)
    weight = (high - low) / 2
    bias = (high + low) / 2
    action_ = action * weight + bias

    return action_


def show_data(episodes, nums):
    matplotlib.rcParams['font.family'] = ['FangSong']
    plt.plot(episodes, nums, label='ave_loss')
    # x轴标签
    plt.xlabel('episodes')
    # y轴标签
    plt.ylabel('nums')
    # 图表标题
    plt.title('loss')

    plt.legend()

    plt.savefig('rewards_1.png')


def train(n_eposides, n_steps, device):
    loss = nn.MSELoss()
    env = EyeEnvironment()
    ave_rewards = []
    eposides = []
    agent = Agent(env.state_space, env.action_space, tao=1., GAMMA=1., lr=0.0003)
    agent.actor_net.target.to(device)
    agent.actor_net.online.to(device)
    agent.critic_net.target.to(device)
    agent.critic_net.online.to(device)
    trainer_1 = torch.optim.Adam(agent.actor_net.online.parameters(), 0.0003)
    trainer_2 = torch.optim.Adam(agent.critic_net.online.parameters(), 0.0003)
    best_rewards = -100000
    for i in range(n_eposides):
        rewards = []
        env.set_mode('train')
        flag, _ = env.reset()
        s = np.asarray(_[0], dtype=np.float32)
        goal = np.asarray(_[1], dtype=np.float32)
        while True:
            s_tensor = torch.as_tensor(s).to(device)
            a = agent.actor_net.online.act(s_tensor)
            r = 1.5 - np.linalg.norm(goal - a)
            flag, _ = env.act()
            s_ = np.asarray(_[0], dtype=np.float32)
            goal = np.asarray(_[1], dtype=np.float32)
            if flag == True:
                is_done = 1
            else:
                is_done = 0
            agent.memo.add(s=s, s_=s_, d=is_done, r=r, a=a)
            s = s_
            if is_done == 1:
                flag, _ = env.reset()
                if flag == False:
                    break
                s = np.asarray(_[0], dtype=np.float32)
                goal = np.asarray(_[1], dtype=np.float32)
            batch_a, batch_s, batch_s_, batch_d, batch_r = agent.memo.sample()
            batch_a = batch_a.to(device)
            batch_s = batch_s.to(device)
            batch_s_ = batch_s_.to(device)
            batch_r = batch_r.to(device)
            batch_d = batch_d.to(device)
            t_actions = agent.actor_net.target(batch_s_)
            t_actions = t_actions.to(device)
            max_q_values = agent.critic_net.target(batch_s_, t_actions)
            max_q_values = max_q_values.to(device)
            y = batch_r + (1 - batch_d) * agent.actor_net.GAMMA * max_q_values
            q_values = agent.critic_net.online(batch_s, batch_a)
            q_values = q_values.to(device)
            trainer_2.zero_grad()
            l1 = loss(y, q_values)
            l1.backward()
            trainer_2.step()
            trainer_2.zero_grad()
            actions = agent.actor_net.online(batch_s)
            values = agent.critic_net.online(batch_s, actions)
            l2 = -torch.mean(values)
            l2.backward()
            trainer_1.step()
            agent.actor_net.update_params()
            agent.critic_net.update_params()
        # agent.actor_net.print_params()
        # agent.critic_net.print_params()
        env.set_mode('test')
        r_rewards = 0
        flag, _ = env.reset()
        s = np.asarray(_[0], dtype=np.float32)
        goal = np.asarray(_[1], dtype=np.float32)
        with torch.no_grad():
            while True:
                s_tensor = torch.as_tensor(s).to(device)
                a = agent.actor_net.online.act(s_tensor)
                r = 1.5 - np.linalg.norm(goal - a)
                r_rewards += r
                flag, _ = env.act()
                s_ = np.asarray(_[0], dtype=np.float32)
                goal = np.asarray(_[1], dtype=np.float32)
                if flag == True:
                    is_done = 1
                    # print(r_rewards)
                    rewards.append(r_rewards)
                    r_rewards = 0
                else:
                    is_done = 0
                s = s_
                if is_done == 1:
                    flag, _ = env.reset()
                    if flag == False:
                        break
                    s = np.asarray(_[0], dtype=np.float32)
                    goal = np.asarray(_[1], dtype=np.float32)
        if np.mean(rewards) > 0.5:
            agent.actor_net.tao = 0.7
        if np.mean(rewards) >1.0:
            agent.actor_net.tao=0.5
        if np.mean(rewards) > best_rewards:
            best_rewards = np.mean(rewards)
            torch.save(agent.actor_net.target.state_dict(), 'actor.pt')
            torch.save(agent.critic_net.target.state_dict(), 'critic.pt')
        eposides.append(i)
        print(f'episode {i + 1} reward {r_rewards} ave reward {np.mean(rewards)}')
        ave_rewards.append(np.mean(rewards))
    show_data(eposides, ave_rewards)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    train(100, 1000, device)
