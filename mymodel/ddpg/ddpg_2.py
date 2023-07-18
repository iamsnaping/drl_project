import torch
from torch import nn
import os
import sys
from tqdm import tqdm
from torch.nn import functional as F
sys.path.append('/home/wu_tian_ci/drl_project')
from torch.utils.data import Dataset
import numpy as np
from mymodel.ddpg.ddpg_base_net import *
from mymodel.ddpg.agent import *
import matplotlib.pyplot as plt
import matplotlib
from mymodel.ddpg.ddpg_dataloader import *
import argparse
from mymodel.ddpg.ddpg_env2 import*
os.chdir(sys.path[0])
from copy import deepcopy as dp


random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)
import time


# import warnings
# warnings.filterwarnings('error')  # achieves nothing


class DataRecorder(object):
    def __init__(self):
        self.object_list=[]
        self.seq_num={}
        self.max_len=0
        self.res_list=[]
        self.restore_path=''
    
    def add(self,last_goal_list,last_action,action,reward,goal,K):
        # print(f'last action {last_action} action {action} goal {goal}')
        if goal[0]==-1 and goal[1]==-1:
            return
        last_goal_list.extend(goal)
        # print(last_action_list)
        last_goal_list.append([K])
        hash_list=str(last_goal_list)
        # print(hash_list)
        if self.seq_num.get(hash_list) is None:
            self.seq_num[hash_list]=self.max_len
            self.res_list.append([[last_action,action,reward]])
            self.max_len+=1
        else:
            idx=self.seq_num.get(hash_list)
            self.res_list[idx].append([last_action.tolist(),action.tolist(),float(reward)])
    
    def print_res(self):
        for item in self.seq_num.items():
            print(item[0],self.res_list[item[1]])


    def restore_data(self):
        f=open(self.restore_path,'w')
        json_dict={}
        for item in self.seq_num.items():
            # print(item[0])
            json_dict[item[0]]=str(self.res_list[item[1]])
        json.dump(json_dict,f,indent=2)
    
    def __del__(self):
        self.restore_data()

class ValueRecoder:
    def __init__(self):
        self.nums=0
        self.data_dic={}
        self.restore_path=''


    def add(self,value):
        self.data_dic[str(self.nums)]=value
        self.nums+=1

    def restore_data(self):
        f=open(self.restore_path,'w')
        json.dump(self.data_dic,f,indent=2)
    
    def __del__(self):
        self.restore_data()
        



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


def test_acc(net, test_iter,device):
    max_loss = 0.
    min_loss = np.inf
    loss = nn.MSELoss(reduction='mean')
    t = 0
    ll = 0.
    net.eval()
    with torch.no_grad():
        for x, y, z in test_iter:
            x=x.to(device)
            y=y.to(device)
            z=z.to(device)
            l = loss(net(x, y), z)
            max_loss = max(max_loss, l.cpu().detach().numpy())
            min_loss = min(min_loss, l.cpu().detach().numpy())
            ll += l
            t += 1
    return max_loss, min_loss, ll / t




# def get_scores(x,y,z,ans):
#     x=(x.cpu().detach().numpy()-ans[:,:,0].reshape(-1,1,1).cpu().detach().numpy())**2
#     y=(y.cpu().detach().numpy()-ans[:,:,1].reshape(-1,1,1).cpu().detach().numpy())**2
#     z=(z.cpu().detach().numpy()-ans[:,:,2].reshape(-1,1,1).cpu().detach().numpy())**2
#     x_scores=get_gaussian(x)
#     y_scores=get_gaussian(y)
#     z_scores=get_gaussian(z)
#     # print(f'sum {x_scores.sum(),y_scores.sum(),z_scores.sum()} ans {ans[0][0]}')
#     return 1.5-x,1.5-y,1-z


'''
def get_scores(goal_p,goal_n,s,s_,is_end):
    with torch.no_grad():
        loss=torch.nn.MSELoss(reduction='none')
        cnt1=100*torch.ones_like(goal_n,dtype=torch.float32).to(goal_n.get_device())
        cnt2=50*torch.ones_like(goal_n,dtype=torch.float32).to(goal_n.get_device())
        l1=torch.mean(loss(goal_n,s),dim=-1,keepdim=True)/100
        l2=torch.mean(loss(goal_n,s_),dim=-1,keepdim=True)/100
        ll=l2-l1
        l3=torch.mean(loss(goal_n,s_),dim=-1,keepdim=True)/50
    return (1-is_end)*0.5*(get_tanh(ll))+(1-get_tanh(l3))*is_end
'''

# goal_last goal_present last_action action is_end for action


# Monte Carlo
def get_scores(goal_n,last_action,action,is_end,epochs):
    l1=np.linalg.norm(last_action-goal_n)
    # print(action,goal_n)
    l2=np.linalg.norm(action-goal_n)
    ll=l1-l2
    flag=False
    if l2<75:
        l=10
        flag=True
    elif l1 <75 and l2>=75:
        l=-20
    else:
        l=np.tanh(ll*0.015)*0.1
        if l<0:
            l*=1.5
    return l,flag

'''
undo



# state,near_state,last_goal_list,last_action,action,isend,rewards,goal,i for action
done
'''

def get_scores_trajectory(trajectory):
    new_tra=[]
    #new ans-> state,near_state,last_goal_list,last_goal,goal,isend
    tra_len=len(trajectory)
    stable_scores=0
    stable_flag=False
    rewards=0
    with torch.no_grad():
        for i in range(tra_len-1,-1,-1):
            state,near_state,last_goal_list,goal,isend,action,last_action=trajectory[i]
            reward,flag=get_scores(goal,last_action,action,isend,tra_len)
            if not flag:
                rewards=rewards*0.9+reward
            else:
                if not stable_flag:
                    stable_flag=True
                    stable_scores=rewards
                else:
                    rewards=stable_scores
            if np.isnan(rewards):
                breakpoint()
            # print(len(last_goal_list_),len(last_goal_list),last_goal_list_,last_goal_list)
            new_tra.append([state,near_state,last_goal_list,last_action,action,isend,rewards,goal,i])
    return new_tra

def train_epoch(agent, lr, epochs, batch_size,device,mode,store_path=None,is_evaluate=False,json_path=None,value_path=None,critic_loss_path=None):
    agent.actor_net.online.to(device)
    agent.actor_net.target.to(device)
    agent.critic_net.online.to(device)
    agent.critic_net.target.to(device)
    if is_evaluate==False:
        trainer_1 = torch.optim.Adam(lr=0.0005, params=agent.actor_net.online.parameters())
        trainer_2=torch.optim.Adam(lr=lr,params=agent.critic_net.online.parameters())
        schedule1 = torch.optim.lr_scheduler.CosineAnnealingLR(trainer_1, T_max=epochs)
        schedule2=torch.optim.lr_scheduler.CosineAnnealingLR(trainer_2,T_max=epochs)
        loss=nn.MSELoss()
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        m_store_path_c=os.path.join(store_path,'CriticNet.pt')
        m_store_path_a=os.path.join(store_path,'ActorNet.pt')
        reward_path=os.path.join(store_path,'reward.txt')
        f=open(reward_path,'w')
    t_reward=[]
    t_reward_len=-1
    best_scores=-np.inf
    is_training=False
    dis_=[]
    # json_path='/home/wu_tian_ci/drl_project/mymodel/ddpg/pretrain_data/json_path'
    for K in tqdm(range(epochs)):
        jdr=DataRecorder()
        jdr.restore_path=os.path.join(json_path,str(K)+'.json')
        vdr=ValueRecoder()
        vdr.restore_path=os.path.join(value_path,str(K)+'.json')
        clp=ValueRecoder()
        clp.restore_path=os.path.join(critic_loss_path,str(K)+'.json')
        # env=DDPGEnv('/home/wu_tian_ci/eyedata/feather/seperate/01/1')
        env=DDPGEnv('/home/wu_tian_ci/eyedata/seperate/02/1')
        env.state_mode=mode
        ll, t = 0, 0
        trajectory=[]
        # env.refresh()
        env.load_dict()
        env.get_file()
        t=0
        rewards=[]
        last_action=np.array([0.,0.],dtype=np.float32)
        #ans -> state last_goal_state last_goal goal is_end
        evalue_scores=[]
        evalua_scores_end=[]
        evalue_reward=[]
        evalua_reward_end=[]
        begin_flag=True
        update_flag=0
        actor_update_flags=0
        while True:
            update_flag+=1
            ans=env.act()
            if isinstance(ans,bool):
                break
            else:
                if begin_flag==True:
                    # print(ans[3],ans[4])
                    last_action[0],last_action[1]=ans[3][0],ans[3][1]
                    begin_flag=False
                state=torch.tensor([ans[0]],dtype=torch.float32).unsqueeze(0).to(device)
                near_state=torch.tensor([ans[1]],dtype=torch.float32).unsqueeze(0).to(device)
                last_goal_list=torch.tensor([ans[2]],dtype=torch.float32).unsqueeze(0).to(device)
                action=agent.actor_net.online.act(last_goal_list,state,near_state,torch.as_tensor([last_action[0],\
                    last_action[1]],dtype=torch.float32).reshape(1,1,2).to(device),K)
                # print(ans[4])
                if ans[5]==1:
                    begin_flag=True
                    # print(action,ans[4],np.linalg.norm(action-np.array(ans[4])),np.mean(evalua_scores_end))
                    # evalua_scores_end.append(np.linalg.norm(action-np.array(ans[4])))
                new_ans=[dp(ans[0]),dp(ans[1]),dp(ans[2]),dp(ans[4]),dp(ans[5]),dp(action),dp(last_action)]
                last_action[0],last_action[1]=action[0],action[1]
                trajectory.append(new_ans)
                if ans[5]==1:
                    # print(trajectory[-1])
                    # breakpoint()
                    scores=0
                    new_trajectory=get_scores_trajectory(trajectory)
                    for tra in new_trajectory:
                        # state,near_state,last_goal_list,last_action,action,isend,rewards,goal,i
                        #0:state,
                        # 1:near_state,
                        # 2:last_goal_list_,
                        # 3:last_action,
                        # 4:action,
                        # 5:isend_,
                        # 6:expectation
                        # 7:goal
                        # print(tra[4])
                        # breakpoint()
                        a,b,c,d,e=dp(tra[2]),dp(tra[3]),dp(tra[4]),dp(tra[6]),dp(tra[7])
                        if tra[5]==1:
                            evalua_scores_end.append(np.linalg.norm(tra[7]-tra[4]))
                            evalua_reward_end.append(dp(tra[6]))
                        elif tra[5]==0:
                            evalue_scores.append(np.linalg.norm(tra[4]-tra[7])-np.linalg.norm(tra[3]-tra[7]))
                            evalue_reward.append(dp(tra[6]))
                        # print(a,b,c,d)
                        jdr.add(a,b,c,d,e,K)
                        # print(tra)
                        agent.memo.push(tra)
                        scores+=tra[8]
                        # print(scores)
                        t+=1
                    trajectory=[]
                    rewards.append(scores)
                    scores=0
            if is_evaluate==False:
                if len(agent.memo)>batch_size and (update_flag%2)==0:
                    # t1=time.process_time()
                    actor_update_flags+=1
                    if not is_training:
                        with open(reward_path,'a') as f:
                            f.write('begin to train\n')
                    state,near_state,last_goal_list_,lastaction,action,isend_,reward,ep = agent.memo.sample(batch_size)
                    # t2=time.process_time()
                    state=state.to(device)
                    near_state=near_state.to(device)
                    last_goal_list_=last_goal_list_.to(device)
                    lastaction=lastaction.to(device)
                    action=action.to(device)
                    done=isend_.to(device)
                    reward=reward.to(device)  
                    ep=ep.to(device)
                    q_values = agent.critic_net.online(lastaction,last_goal_list_,torch.concat([state,near_state],dim=-1),torch.cat([action,ep],dim=-1))
                    l1 = loss(reward, q_values)
                    trainer_2.zero_grad()
                    l1.backward()
                    trainer_2.step()
                    clp.add(l1.cpu().detach().numpy().tolist())
                    if K>0 and actor_update_flags==2:
                        actions = agent.actor_net.online(last_goal_list_,state,near_state,lastaction)
                        values = agent.critic_net.online(lastaction,last_goal_list_,torch.concat([state,near_state],dim=-1),torch.cat([actions,ep],dim=-1))
                        vdr.add(torch.mean(values).cpu().detach().numpy().tolist())
                        l2 = -torch.mean(values)
                        trainer_1.zero_grad()
                        l2.backward()
                        trainer_1.step()
                    is_training=True
                    actor_update_flags=actor_update_flags%2
                    # if update_flag%10==0:
                    agent.actor_net.update_params()
                    agent.critic_net.update_params()
        if is_evaluate==False:
            t_reward_len+=1
            if t_reward_len>=1000:
                t_reward_len%=1000
                t_reward[t_reward_len]=np.mean(rewards)
            else:
                t_reward.append(np.mean(rewards))
            if len(t_reward)>0 and t_reward[-1]>best_scores and is_training:
                torch.save(agent.actor_net.target.state_dict(), m_store_path_a)
                torch.save(agent.critic_net.target.state_dict(),m_store_path_c)
                best_scores=t_reward[-1]
            # print(tr,np.array(tr)/trt)
            with open(reward_path,'a',encoding='UTF-8') as f:
                f.write('eposides:'+str(K+1)+' ave_eposides_rewards:'+str(t_reward[-1])+' ave_total_rewards:'+str(np.mean(t_reward))\
                    +'   mean dis:'+str(np.mean(evalue_scores))+' end_dis:'+str(np.mean(evalua_scores_end))+'\n'+
                    'ave reward '+str(np.mean(evalue_reward)) +' end_reward ' +str(np.mean(evalua_reward_end))+'\n')
        # schedule1.step()
        # schedule2.step()



if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-load_policy',type=str,default='5_policy_1last_move_5')
    parser.add_argument('-store',type=str,default='ddpg_2')
    parser.add_argument('-load_critic',type=str,default='5_policy_1last_move_5')
    parser.add_argument('-cuda',type=str,default='cuda:3')
    parser.add_argument('-mode',type=int,default=1)
    parser.add_argument('-eval',type=bool,default=False)
    parser.add_argument('-net',type=int,default=1)
    parser.add_argument('-sup',type=str,default='50')
    # net1- >base net
    args=parser.parse_args()
    agent=Agent(buffer_maxlen=50000,mode=args.net)
    print(args.sup)
    if args.eval==False:
        actor_load=os.path.join('/home/wu_tian_ci/drl_project/mymodel/ddpg/pretrain_data/policy/',args.load_policy)
        actor_load=os.path.join(actor_load,'1/5000_03/PolicyNet.pt')
        agent.actor_net.online.load_state_dict(torch.load(actor_load))
        agent.actor_net.target.load_state_dict(torch.load(actor_load))
        # agent.critic_net.online.load_state_dict(torch.load(critic_load))
        # agent.critic_net.target.load_state_dict(torch.load(critic_load))
        agent.critic_net.online.apply(weight_init)
        agent.critic_net.target.apply(weight_init)
    else:
        actor_load='/home/wu_tian_ci/drl_project/mymodel/ddpg/pretrain_data/ddpg_train/1last_move5/ActorNet.pt'
        agent.actor_net.online.load_state_dict(torch.load(actor_load))
        agent.actor_net.target.load_state_dict(torch.load(actor_load))
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    store_path=os.path.join('/home/wu_tian_ci/drl_project/mymodel/ddpg/pretrain_data/ddpg_train/',args.store)
    json_path=os.path.join('/home/wu_tian_ci/drl_project/mymodel/ddpg/pretrain_data/json_path/',args.store)
    value_path=os.path.join('/home/wu_tian_ci/drl_project/mymodel/ddpg/pretrain_data/value_path/',args.store)
    critic_loss_path=os.path.join('/home/wu_tian_ci/drl_project/mymodel/ddpg/pretrain_data/critic_loss/',args.store)
    if not os.path.exists(json_path):
        os.makedirs(json_path)
    if not os.path.exists(value_path):
        os.makedirs(value_path)
    if not os.path.exists(critic_loss_path):
        os.makedirs(critic_loss_path)
    train_epoch(agent, 0.03, 1000, 1024,device,args.mode,store_path,is_evaluate=args.eval,json_path=json_path,value_path=value_path,critic_loss_path=critic_loss_path)