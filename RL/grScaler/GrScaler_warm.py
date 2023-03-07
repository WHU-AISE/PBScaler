# https://github.com/mengwanglalala/RL-algorithms.git

import time
import pandas as pd
import torch
from torch import optim
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


import numpy as np
import matplotlib.pyplot as plt
import os

from RL.Environment import Environment
from RL.Simulation import Simulation
from RL.common.MPNN import MPNN
from RL.grScaler.GraphData import GraphData
from RL.grScaler.replaybuffer import Buffer


WARM_BATCH = 512
WARM_EPOCH = 200

LR = 0.003
REPLACE = 10
CHECKPOINT_DIR = './RL/grScaler/checkpoints/warm/v1/'
HIDDEN_LAYER = 128
BUFFER_SIZE = 2**17
BATCH_SIZE = 64
EPSILON = 1.0
EPSILON_DECAY = 1e-8
EPSOIDE = 100
EPSOIDE_TIME = 960
DISCOUNT = 0.99
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

        self.mpnn = MPNN(state_dim, HIDDEN_LAYER)
        self.fc1 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.fc2 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.V = nn.Linear(HIDDEN_LAYER, 1)
        self.A = nn.Linear(HIDDEN_LAYER, action_dim)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.optim = optim.Adam(self.parameters(), lr=LR)
        self.crit = nn.MSELoss()


    def forward(self, state, edge_index, batch_size):
        x = self.relu1(self.mpnn(state, edge_index))
        x = self.relu2(self.fc1(x))
        x = self.relu3(self.fc2(x))
        A = self.A(x)
        A_BATCH = global_mean_pool(A, batch_size)
        x = global_mean_pool(x, batch_size)  # [batch_size, hidden_channels]
        V = self.V(x)        
        Q = V + (A_BATCH - torch.mean(A_BATCH, dim=1, keepdim=True))
        return Q

    def advantage(self, state, edge_index):
        x = self.relu1(self.mpnn(state, edge_index))
        x = self.relu2(self.fc1(x))
        x = self.relu3(self.fc2(x))
        return self.A(x)


class GrScaler:
    def __init__(self, env: Environment):
        self.env = env
        # 5 metrics
        self.state_dim = 5
        # svc count: [1~8]
        self.action_dim = 8
        self.epsilon = EPSILON
        self.eps_min = 0.01
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
        if np.random.random() < self.epsilon:
            advantage = self.q_eval.advantage(state, edge_index)
            action = (torch.argmax(advantage, dim=-1, keepdim=False) + 1).view(-1, 1)
        else:
            action = torch.randint(1, self.action_dim + 1,(state.shape[0], 1))
        return action

    def build_graph_data(self, state, edge_index, action=None):
        if action is not None:
            return Data(x=state, y=action, edge_index=edge_index)
        else:
            return Data(x=state, edge_index=edge_index)
    
    def warm(self):
        print('warm the model...')
        # build state from true trace
        train_df = pd.read_csv('./RL/grScaler/KHPA_trace.csv')
        train_df = train_df.dropna(axis=0,how='any')

        self.edge_index = torch.from_numpy(self.env.edge_index)

        # build the train_data
        train_datas = []
        for index, row in train_df.iterrows():
            if index < len(train_df)-1:
                next_row = train_df.iloc[index+1]
                state = []
                action = []
                for svc in self.env.svcs:
                    svc_state = []
                    svc_state.append(row[svc+'&cpu_usage']/row[svc+'&cpu_limit'])
                    svc_state.append(row[svc+'&mem_usage']/row[svc+'&mem_limit'])
                    svc_state.append(row[svc + '&p90'])
                    svc_state.append(row[svc + '&qps'])
                    svc_state.append(row[svc + '&count'])
                    state.append(svc_state)
                    action.append([next_row[svc+'&count']])
                state = torch.FloatTensor(state)
                action = torch.LongTensor(action)

                graph = self.build_graph_data(state, self.edge_index, action)
                train_datas.append(graph)
        train_loader = DataLoader(dataset=train_datas, batch_size=WARM_BATCH,shuffle=True)
        
        loss_func = nn.CrossEntropyLoss()
        for epoch in range(WARM_EPOCH):
            correct = 0
            total = 0
            for step, data in enumerate(train_loader):
                self.q_eval.train()
                data.to(device)
                actionWeight = self.q_eval.advantage(data.x, data.edge_index)
                loss = loss_func(actionWeight, (data.y - 1).squeeze())

                # action = torch.multinomial(actionWeight.exp(), 1, replacement=False) + 1
                action = torch.argmax(actionWeight, dim=-1, keepdim=False) + 1
                correct += int((action.squeeze() == data.y.squeeze()).sum())
                total += len(action)
        
                # Backpropagation
                self.q_eval.optim.zero_grad()
                loss.backward()
                self.q_eval.optim.step()
            # scheduler.step()
            train_loss = loss.item()
            print(f'Epoch: {epoch:03d}, Train_loss: {train_loss:10.8f}, Train_acc: {correct/total * 100:10.4f}%')

        # save model
        torch.save(self.net, CHECKPOINT_DIR + 'warmModel.pt')


    def train(self):
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
            
        print('Training started...')
        self.edge_index = torch.from_numpy(self.env.edge_index)

        state, state_df = self.env.get_state()
        state_df.to_csv('state_trans.csv')

        for episode in range(self.start, self.end):
            ep_reward = 0
            state, state_df = self.env.get_state()
            state_df.to_csv('state_trans.csv', mode='a', header=False)
            time_start = time.time()
            
            while True:
                time_now = time.time()
                if time_now - time_start > EPSOIDE_TIME:
                    break
                if self.learn_step_counter % REPLACE == 0:
                    self.q_target.load_state_dict(self.q_eval.state_dict())
                # get maximizing action
                self.q_eval.eval()
                action = self.choose_action(state, self.edge_index)
                self.q_eval.train()
                # step episode
                next_state, next_state_df, reward = self.env.new_step(torch.Tensor.cpu(action))
                next_state_df.to_csv('state_trans.csv', mode='a', header=False)
                # store transition
                self.buffer.append((state, torch.Tensor.cpu(action), next_state, torch.FloatTensor([reward]).view(-1,1)))
                state = next_state
                ep_reward += reward

                if episode > 1:
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

                self.epsilon = self.epsilon - EPSILON_DECAY if self.epsilon > self.eps_min else self.eps_min
                self.learn_step_counter += 1
            print("EP -", episode, "| Total Reward -", ep_reward.detach().numpy())
  
            # save to checkpoints
            if episode % 10 == 0:
                self.save_checkpoint(episode)
            self.rewardgraph.append(ep_reward)

            if PLOT_FIG:
                if not os.path.exists(CHECKPOINT_DIR + 'img/'):
                    os.makedirs(CHECKPOINT_DIR + 'img/')
                if episode % 10 ==0 and episode != 0:
                    plt.plot(self.rewardgraph, color='darkorange')  # total rewards in an iteration or episode
                    # plt.plot(avg_rewards, color='b')  # (moving avg) rewards
                    plt.xlabel('Episodes')
                    plt.savefig(CHECKPOINT_DIR + 'img/ep'+str(episode)+'.png')
        if PLOT_FIG:
            plt.plot(self.rewardgraph, color='darkorange')  # total rewards in an iteration or episode
            # plt.plot(avg_rewards, color='b')  # (moving avg) rewards
            plt.xlabel('Episodes')
            plt.savefig('final.png')


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
