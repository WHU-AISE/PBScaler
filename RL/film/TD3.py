'''
    the code is written based on {film}
    @inproceedings {258914,
        author = {Haoran Qiu and Subho S. Banerjee and Saurabh Jha and Zbigniew T. Kalbarczyk and Ravishankar K. Iyer},
        title = {{FIRM}: An Intelligent Fine-grained Resource Management Framework for {SLO-Oriented} Microservices},
        booktitle = {14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20)},
        year = {2020},
        isbn = {978-1-939133-19-9},
        pages = {805--825},
        url = {https://www.usenix.org/conference/osdi20/presentation/qiu},
        publisher = {USENIX Association},
        month = nov,
    }
    https://gitlab.engr.illinois.edu/DEPEND/firm
'''

# PyTorch
from time import sleep
from RL.Environment import Environment
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

# pyg
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

# Lib
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from RL.Simulation import Simulation
from RL.film.actorcritic import Actor, Critic

# Files
from RL.grScaler.replaybuffer import Buffer
from RL.common.MPNN import MPNN
from RL.common.GAT import GCNCell, GATCell

from RL.common.StateModel import STATE_MODEL
from RL.grScaler.GraphData import GraphData
from RL.grScaler.noise import OrnsteinUhlenbeckActionNoise as OUNoise

PLOT_FIG = True

# Hyperparameters
ACTOR_LR = 0.03
CRITIC_LR = 0.03
MINIBATCH_SIZE = 256
NUM_EPISODES = 10000
MU = 0
SIGMA = 0.2
CHECKPOINT_DIR = './RL/film/checkpoints/TD3/v3/'
STATE_MODEL_PATH = './RL/state.pt'
BUFFER_SIZE = 2**20
DISCOUNT = 0.99
TAU = 0.001
DELAY = 100
# noise
NOISE_CLIP = 0.1
POLICY_NOISE = 0.2
EPSILON = 1.0
EPSILON_DECAY = 1e-6
CUDA = True

# device
# set device to cpu or cuda
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


HIDDEN_LAYER = 128

class TD3:
    def __init__(self, env, transfer=False):
        self.env = env
        # 5 metrics
        self.state_dim = len(self.env.svcs) * 5
        # svc count: [1~8]
        self.max_action = 7
        self.action_dim = len(self.env.svcs) * 8
        self.buffer = Buffer(BUFFER_SIZE)
        self.critic_loss = nn.MSELoss()
        self.rewardgraph = []
        self.start = 0
        self.end = NUM_EPISODES
        self.epsilon = EPSILON
        self.noise = OUNoise(mu=np.zeros(self.action_dim), sigma=SIGMA)

        self.build_net()

         
    def build_net(self):
        # 6 networks
        self.actor = Actor(self.state_dim, self.action_dim).to(device)
        self.target_actor = Actor(self.state_dim, self.action_dim).to(device)
        self.critic_1 = Critic(self.state_dim, self.action_dim).to(device)
        self.target_critic_1 = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_2 = Critic(self.state_dim, self.action_dim).to(device)
        self.target_critic_2 = Critic(self.state_dim, self.action_dim).to(device)
        # optimizer
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optim_1 = optim.Adam(self.critic_1.parameters(), lr=CRITIC_LR)
        self.critic_optim_2 = optim.Adam(self.critic_2.parameters(), lr=CRITIC_LR)


    # def begin(self):
    #     self.edge_index = torch.tensor(self.env.edge_index, dtype=torch.long)
    #     while True:
    #         state, _ = self.env.get_state()
    #         action = torch.Tensor.cpu(self.choose_action(self.actor, state, self.edge_index))
    #         self.env.new_step(action)
    #         sleep(60)

    def update_targets(self, target, original):
        for targetParam, orgParam in zip(target.parameters(), original.parameters()):
            targetParam.data.copy_((1 - TAU)*targetParam.data + TAU*orgParam.data)

    # def choose_action(self, agent, state, edge_index):
    #     if np.random.random() < EPSILON:
    #         action_prob = self.get_action_prob(agent, state, edge_index)
    #         action = (torch.argmax(action_prob, dim=-1, keepdim=False) + 1).view(-1,1)
    #     else:
    #         action = torch.randint(1, self.action_dim + 1,(state.shape[0], 1))
    #     return action, action_prob.detach()

    # def get_action_prob(self, agent, state, edge_index):
    #     state = state.to(device)
    #     edge_index = edge_index.to(device)
    #     action_prob = agent(state, edge_index)
    #     return action_prob

    def getMaxAction(self, curState):
        if CUDA:
            noise = (self.epsilon * Variable(torch.FloatTensor(self.noise()))).cuda()
        else:
            noise = (self.epsilon * Variable(torch.FloatTensor(self.noise())))
        action = self.actor(curState.view(1,-1).to(device))
        actionNoise = action + noise
        
        # get the max
        # action_list = actionNoise.tolist()[0]
        # max_action = max(action_list)
        # max_index = action_list.index(max_action)
        actionProb = nn.Softmax(dim=-1)(actionNoise.view(len(self.env.svcs), 8))
        final_action = torch.argmax(actionProb, dim=-1, keepdim=False) + 1
        return torch.Tensor.cpu(final_action), actionNoise.detach()

    def train(self):
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
            
        print('Training started...')

        self.simulation_env = Simulation(train=False)
        # simulation: qps
        gradually_decrease_df = pd.read_csv('./traffic/simulation/gradually_decrease.csv')
        gradually_increase_df = pd.read_csv('./traffic/simulation/gradually_increase.csv')
        sudden_decrease_df = pd.read_csv('./traffic/simulation/sudden_decrease.csv')
        sudden_increase_df = pd.read_csv('./traffic/simulation/sudden_increase.csv')
        single_peak_df = pd.read_csv('./traffic/simulation/single_peak.csv')
        qps_dfs = [single_peak_df, sudden_increase_df, sudden_decrease_df, gradually_increase_df, gradually_decrease_df]
        svcs = self.simulation_env.svcs

        # for each episode 
        for episode in range(self.start, self.end):
            state = self.simulation_env.get_state()
            ep_reward = 0
            qps_df = pd.concat(qps_dfs)[svcs].fillna(0).astype('float')
            qps_df = qps_df[::10]
            state = self.simulation_env.get_state()

            step = 0

            for _, qps_row in qps_df.iterrows():
                # get maximizing action
                self.actor.eval()
                action, action_prob = self.getMaxAction(state)
                action = (action).view(-1, 1)
                self.actor.train()
                # step episode
                qps_input = torch.from_numpy(qps_row[svcs].values).reshape(-1,1)
                next_state, reward = self.simulation_env.new_step(state, qps_input, torch.Tensor.cpu(action))
                # store transition
                self.buffer.append((state.view(1,-1), torch.Tensor.cpu(action_prob), next_state.view(1,-1), torch.FloatTensor([reward]).view(-1,1)))
                state = next_state
                ep_reward += reward
                step += 1

                # if step % 500 == 0:
                #     print('Actor Loss: {}'.format(actor_loss), 'critic1 Loss: {}'.format(critic_loss_1), 'critic2 Loss: {}'.format(critic_loss_2))

                    
                # Training loop
            cur_state_batch, action_prob_batch, next_state_batch, reward_batch = self.buffer.sample_batch(MINIBATCH_SIZE)
            
            cur_state_batch = torch.cat(cur_state_batch).to(device)
            action_prob_batch = torch.cat(action_prob_batch).to(device)
            reward_batch = torch.cat(reward_batch).to(device)
            next_state_batch = torch.cat(next_state_batch).to(device)

            # Compute target Q-value:
            target_Q1 = self.target_critic_1(next_state_batch, action_prob_batch)
            target_Q2 = self.target_critic_2(next_state_batch, action_prob_batch)
            q_next = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch + (DISCOUNT * q_next.data)

            # Optimize Critic 1:
            current_Q1 = self.critic_1(cur_state_batch, action_prob_batch)
            critic_loss_1 = self.critic_loss(current_Q1, target_Q)
            self.critic_optim_1.zero_grad()
            critic_loss_1.backward()
            self.critic_optim_1.step()

            # Optimize Critic 2:
            current_Q1 = self.critic_2(cur_state_batch, action_prob_batch)
            critic_loss_2 = self.critic_loss(current_Q1, target_Q)
            self.critic_optim_2.zero_grad()
            critic_loss_2.backward()
            self.critic_optim_2.step()

            _cur_state_batch = cur_state_batch.detach()
            a = self.actor(_cur_state_batch)
            q = self.critic_1(_cur_state_batch, a)
            actor_loss = -torch.mean(q)

            # Actor update
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # Update Targets                        
            # self.update_targets(self.target_actor, self.actor)
            # self.update_targets(self.target_critic_1, self.critic_1)
            # self.update_targets(self.target_critic_2, self.critic_2)
            if episode % 5 == 0:
                self.target_actor.load_state_dict(self.actor.state_dict())
                self.target_critic_1.load_state_dict(self.critic_1.state_dict())
                self.target_critic_2.load_state_dict(self.critic_2.state_dict())
            self.epsilon -= EPSILON_DECAY

                
            # save to checkpoints
            if episode % 100 == 0:
                self.save_checkpoint(episode)

            # test
            if episode % 10 == 0:
                ep_reward = self.test()
                print("EP -", episode, "| Total Reward -", ep_reward.detach().numpy())
                self.rewardgraph.append(ep_reward)

            if PLOT_FIG:
                if not os.path.exists(CHECKPOINT_DIR + 'img/'):
                    os.makedirs(CHECKPOINT_DIR + 'img/')
                if episode % 100 ==0 and episode != 0:
                    plt.plot(self.rewardgraph, color='darkorange')  # total rewards in an iteration or episode
                    # plt.plot(avg_rewards, color='b')  # (moving avg) rewards
                    plt.xlabel('Episodes')
                    plt.savefig(CHECKPOINT_DIR + 'img/ep'+str(episode)+'.png')
        if PLOT_FIG:
            plt.plot(self.rewardgraph, color='darkorange')  # total rewards in an iteration or episode
            # plt.plot(avg_rewards, color='b')  # (moving avg) rewards
            plt.xlabel('Episodes')
            plt.savefig('final.png')

    # remove noise
    def test(self):
        ep_reward = 0
        # simulation: qps
        gradually_decrease_df = pd.read_csv('./traffic/simulation/gradually_decrease.csv')
        gradually_increase_df = pd.read_csv('./traffic/simulation/gradually_increase.csv')
        sudden_decrease_df = pd.read_csv('./traffic/simulation/sudden_decrease.csv')
        sudden_increase_df = pd.read_csv('./traffic/simulation/sudden_increase.csv')
        single_peak_df = pd.read_csv('./traffic/simulation/single_peak.csv')
        qps_dfs = [single_peak_df, sudden_increase_df, sudden_decrease_df, gradually_increase_df, gradually_decrease_df]
        svcs = self.simulation_env.svcs
        state = self.simulation_env.get_state()
        qps_df = pd.concat(qps_dfs)[svcs].dropna(axis=0,how='any').astype('float')
        qps_df = qps_df[::10]
        for _, qps_row in qps_df.iterrows():
            # get maximizing action
            self.actor.eval()
            _state = state.view(1,-1).to(device)
            action_prob = self.actor(_state).view(-1,1)
            action_prob = nn.Softmax(dim=-1)(action_prob.view(len(self.env.svcs), 8))
            action = torch.argmax(action_prob, dim=-1, keepdim=False) + 1
            action = action.view(-1,1)
            # step episode
            qps_input = torch.from_numpy(qps_row[svcs].values).reshape(-1,1)
            next_state, reward = self.simulation_env.new_step(state, qps_input, torch.Tensor.cpu(action))
            state = next_state
            ep_reward += reward
        return ep_reward


    def save_checkpoint(self, episode_num):
        checkpointName = CHECKPOINT_DIR + 'ep{}.pth.tar'.format(episode_num)
        checkpoint = {
            'episode': episode_num,
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'actor_target': self.target_actor.state_dict(),
            'critic_target_1': self.target_critic_1.state_dict(),
            'critic_target_2': self.target_critic_2.state_dict(),
            'actor_opt': self.actor_optim.state_dict(),
            'critic_opt_1': self.critic_optim_1.state_dict(),
            'critic_opt_2': self.critic_optim_2.state_dict(),
            'buffer': self.buffer,
            'rewardgraph': self.rewardgraph            
        } 
        torch.save(checkpoint, checkpointName)
    
    def loadCheckpoint(self, checkpointName):
        if os.path.isfile(checkpointName):
            print("Loading checkpoint...")
            checkpoint = torch.load(checkpointName)
            self.start = checkpoint['episode'] + 1
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic_1.load_state_dict(checkpoint['critic_1'])
            self.critic_2.load_state_dict(checkpoint['critic_2'])
            self.target_actor.load_state_dict(checkpoint['actor_target'])
            self.target_critic_1.load_state_dict(checkpoint['critic_target_1'])
            self.target_critic_2.load_state_dict(checkpoint['critic_target_2'])
            self.actor_optim.load_state_dict(checkpoint['actor_opt'])
            self.critic_optim_1.load_state_dict(checkpoint['critic_opt_1'])
            self.critic_optim_2.load_state_dict(checkpoint['critic_opt_2'])
            self.buffer = checkpoint['buffer']
            self.rewardgraph = checkpoint['rewardgraph']
            print('Checkpoint loaded')
        else:
            raise OSError('Checkpoint not found')
