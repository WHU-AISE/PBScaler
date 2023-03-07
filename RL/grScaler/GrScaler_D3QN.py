# https://github.com/mengwanglalala/RL-algorithms.git

import pandas as pd
import torch
from torch import optim
import torch.nn as nn
from torch_geometric.nn import global_mean_pool


import numpy as np
import matplotlib.pyplot as plt
import os
from more_itertools import chunked

from RL.Environment import Environment
from RL.Simulation import Simulation
from RL.common.MPNN import MPNN
from RL.common.GAT import GCNCell, GATCell
from RL.grScaler.GraphData import GraphData
from RL.grScaler.replaybuffer import Buffer

LR = 0.003
DELAY = 10
CHECKPOINT_DIR = './RL/grScaler/checkpoints/d3qn/v4/'
HIDDEN_LAYER = 128
BUFFER_SIZE = int(1e7)
BATCH_SIZE = 256
EPSILON = 0.1
EPSILON_DECAY = 8e-9
EPSILON_MIN = 0.05
EPSOIDE = 10000
DISCOUNT = 0.99
TAU = 0.005
PLOT_FIG = True


# device
# set device to cpu or cuda
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class DuelingDeepQNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDeepQNet, self).__init__()

        self.mpnn = GCNCell(state_dim, HIDDEN_LAYER)
        self.bn0 = nn.BatchNorm1d(HIDDEN_LAYER)
        self.fc1 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.fc1.weight.data = nn.init.uniform_(self.fc1.weight.data, a=0, b=1)
        self.fc2 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.fc1.weight.data = nn.init.uniform_(self.fc2.weight.data, a=0, b=1)
        self.V = nn.Linear(HIDDEN_LAYER, 1)
        self.A = nn.Linear(HIDDEN_LAYER, action_dim)

        self.Relu = nn.ReLU()

        self.optim = optim.Adam(self.parameters(), lr=LR)
        self.crit = nn.MSELoss()


    def forward(self, state, edge_index, batch_size):
        x = self.Relu(self.mpnn(state, edge_index))
        x_norm = self.bn0(x)
        x = self.Relu(self.fc1(x_norm))
        x = self.Relu(self.fc2(x))
        A = self.A(x)
        A_BATCH = global_mean_pool(A, batch_size)
        x = global_mean_pool(x, batch_size)  # [batch_size, hidden_channels]
        V = self.V(x)        
        Q = V + (A_BATCH - torch.mean(A_BATCH, dim=1, keepdim=True))
        return Q

    def advantage(self, state, edge_index):
        x = self.Relu(self.mpnn(state, edge_index))
        x = self.Relu(self.fc1(x))
        x = self.Relu(self.fc2(x))
        return self.A(x)


class Agent:
    def __init__(self, env: Environment, transfer=False):
        self.env = env
        # 5 metrics
        self.state_dim = 5
        # svc count: [1~8]
        self.action_dim = 8
        self.epsilon = EPSILON
        self.learn_step_counter = 0
        self.start = 0
        self.end = EPSOIDE
        self.rewardgraph = []

        self.buffer = Buffer(BUFFER_SIZE)
        self.q_eval = DuelingDeepQNet(state_dim=self.state_dim, action_dim=self.action_dim).to(device)
        self.q_target = DuelingDeepQNet(state_dim=self.state_dim, action_dim=self.action_dim).to(device)

    def choose_action(self, state, edge_index):
        state = state.to(device)
        edge_index = edge_index.to(device)
        if np.random.random() > self.epsilon:
            advantage = self.q_eval.advantage(state, edge_index)
            action = (torch.argmax(advantage, dim=-1, keepdim=False) + 1).view(-1, 1)
        else:
            action = torch.randint(1, self.action_dim + 1,(state.shape[0], 1))
        return action


    def train(self):
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
            
        print('Training started...')

        simulation_env = Simulation(train=False)
        self.edge_index = torch.tensor(simulation_env.edge_index, dtype=torch.long).to(device)
        # simulation: qps
        gradually_decrease_df = pd.read_csv('./traffic/simulation/gradually_decrease.csv')
        gradually_increase_df = pd.read_csv('./traffic/simulation/gradually_increase.csv')
        sudden_decrease_df = pd.read_csv('./traffic/simulation/sudden_decrease.csv')
        sudden_increase_df = pd.read_csv('./traffic/simulation/sudden_increase.csv')
        single_peak_df = pd.read_csv('./traffic/simulation/single_peak.csv')
        qps_dfs = [single_peak_df, sudden_increase_df, sudden_decrease_df, gradually_increase_df, gradually_decrease_df]
        svcs = simulation_env.svcs
        for episode in range(self.start, self.end):
            state = simulation_env.get_state()
            qps_df = pd.concat(qps_dfs)[svcs].dropna(axis=0,how='any').astype('float')
            qps_df = qps_df[::10]
            for _, qps_row in qps_df.iterrows():
                # get maximizing action
                self.q_eval.eval()
                action = self.choose_action(state, self.edge_index)
                self.q_eval.train()
                # step episode
                qps_input = torch.from_numpy(qps_row[svcs].values).reshape(-1,1)
                next_state, reward = simulation_env.new_step(state, qps_input, torch.Tensor.cpu(action))
                # store transition
                self.buffer.append((state, torch.Tensor.cpu(action), next_state, torch.FloatTensor([reward]).view(-1,1)))
                state = next_state

                if episode > 1 and self.learn_step_counter % DELAY == 0:           

                    cur_state_batch, _, next_state_batch, reward_batch = self.buffer.sample_batch(BATCH_SIZE)
                    cur_graph_datas = GraphData(cur_state_batch, self.edge_index).get_batch_data(BATCH_SIZE).to(device)
                    next_graph_datas = GraphData(next_state_batch, self.edge_index).get_batch_data(BATCH_SIZE).to(device)
                    reward_batch = torch.cat(reward_batch).to(device)

                    q_pred_batch = self.q_eval(cur_graph_datas.x, cur_graph_datas.edge_index, cur_graph_datas.batch)
                    q_next = self.q_target(next_graph_datas.x, next_graph_datas.edge_index, next_graph_datas.batch)
                    q_target_batch = reward_batch + (DISCOUNT * q_next.data)                
                    critic_loss = self.q_eval.crit(q_pred_batch, q_target_batch)

                    self.q_eval.optim.zero_grad()
                    critic_loss.backward()
                    self.q_eval.optim.step()

                    # update target network
                    if self.learn_step_counter % 100 == 0:
                        self.q_target.load_state_dict(self.q_eval.state_dict())


                    self.epsilon = self.epsilon - EPSILON_DECAY if self.epsilon > EPSILON_MIN else EPSILON_MIN
                self.learn_step_counter += 1  
            # save to checkpoints
            if episode % 100 == 0:
                self.save_checkpoint(episode)
                print('the epsilon: ', self.epsilon)

            # test
            if episode % 10 == 0:
                ep_reward = self.test()
                print("EP -", episode, "| Total Reward -", ep_reward.detach().numpy())
                self.rewardgraph.append(ep_reward)

            if PLOT_FIG:
                if not os.path.exists(CHECKPOINT_DIR + 'img/'):
                    os.makedirs(CHECKPOINT_DIR + 'img/')
                if episode % 100 ==0 and episode != 0:
                    reward_mean = self.rewardgraph
                    # reward_mean= [sum(x) / len(x) for x in chunked(self.rewardgraph, 12)]
                    plt.plot(reward_mean, color='darkorange')  # total rewards in an iteration or episode
                    # plt.plot(avg_rewards, color='b')  # (moving avg) rewards
                    plt.xlabel('Episodes')
                    plt.savefig(CHECKPOINT_DIR + 'img/ep'+str(episode)+'.png')
        if PLOT_FIG:
            reward_mean = self.rewardgraph
            # reward_mean= [sum(x) / len(x) for x in chunked(self.rewardgraph, 12)]
            plt.plot(reward_mean, color='darkorange')  # total rewards in an iteration or episode
            # plt.plot(avg_rewards, color='b')  # (moving avg) rewards
            plt.xlabel('Episodes')
            plt.savefig('final.png')

    # remove noise
    def test(self):
        ep_reward = 0
        simulation_env = Simulation(train=False)
        self.edge_index = torch.tensor(simulation_env.edge_index, dtype=torch.long).to(device)
        # simulation: qps
        gradually_decrease_df = pd.read_csv('./traffic/simulation/gradually_decrease.csv')
        gradually_increase_df = pd.read_csv('./traffic/simulation/gradually_increase.csv')
        sudden_decrease_df = pd.read_csv('./traffic/simulation/sudden_decrease.csv')
        sudden_increase_df = pd.read_csv('./traffic/simulation/sudden_increase.csv')
        single_peak_df = pd.read_csv('./traffic/simulation/single_peak.csv')
        qps_dfs = [single_peak_df, sudden_increase_df, sudden_decrease_df, gradually_increase_df, gradually_decrease_df]
        svcs = simulation_env.svcs
        state = simulation_env.get_state()
        qps_df = pd.concat(qps_dfs)[svcs].dropna(axis=0,how='any').astype('float')
        qps_df = qps_df[::10]
        for _, qps_row in qps_df.iterrows():
            # get maximizing action
            self.q_eval.eval()
            _state = state.to(device)
            edge_index = self.edge_index.to(device)
            advantage = self.q_eval.advantage(_state, edge_index)
            action = (torch.argmax(advantage, dim=-1, keepdim=False) + 1).view(-1, 1)
            # step episode
            qps_input = torch.from_numpy(qps_row[svcs].values).reshape(-1,1)
            next_state, reward = simulation_env.new_step(state, qps_input, torch.Tensor.cpu(action))
            state = next_state
            ep_reward += reward
        return ep_reward

    def save_checkpoint(self, episode_num):
        checkpointName = CHECKPOINT_DIR + 'ep{}.pth.tar'.format(episode_num)
        checkpoint = {
            'episode': episode_num,
            'q_eval': self.q_eval.state_dict(),
            'q_target': self.q_target.state_dict(),
            'buffer': self.buffer,
            'rewardgraph': self.rewardgraph,
            'learn_step_counter': self.learn_step_counter       
        } 
        torch.save(checkpoint, checkpointName)
    
    def loadCheckpoint(self, checkpointName):
        if os.path.isfile(checkpointName):
            print("Loading checkpoint...")
            checkpoint = torch.load(checkpointName)
            self.start = checkpoint['episode'] + 1
            self.q_eval.load_state_dict(checkpoint['q_eval'])
            self.q_target.load_state_dict(checkpoint['q_target'])
            self.buffer = checkpoint['buffer']
            self.rewardgraph = checkpoint['rewardgraph']
            self.learn_step_counter = checkpoint['learn_step_counter']
            print('Checkpoint loaded')
        else:
            raise OSError('Checkpoint not found')
