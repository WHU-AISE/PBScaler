# PyTorch
from RL.Environment import Environment
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

# Lib
import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display
import os
import pandas as pd
from RL.Simulation import Simulation

# Files
from RL.film.noise import OrnsteinUhlenbeckActionNoise as OUNoise
from RL.film.replaybuffer import Buffer
from RL.film.actorcritic import Actor, Critic

PLOT_FIG = True

LR = 0.003
DELAY = 100
CHECKPOINT_DIR = './RL/film/checkpoints/d3qn/v1/'
HIDDEN_LAYER = 128
BUFFER_SIZE = int(1e7)
BATCH_SIZE = 256
EPSILON = 0.1
EPSILON_DECAY = 8e-7
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

def fanin_init(size, fanin=None):
    """Utility function for initializing actor and critic"""
    fanin = fanin or size[0]
    w = 1./ np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-w, w)

class DuelingDeepQNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDeepQNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, HIDDEN_LAYER)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.V = nn.Linear(HIDDEN_LAYER, 1)
        self.A = nn.Linear(HIDDEN_LAYER, action_dim)

        self.Relu = nn.ReLU()

        self.optim = optim.Adam(self.parameters(), lr=LR)
        self.crit = nn.MSELoss()


    def forward(self, state):
        x = self.Relu(self.fc1(state))
        x = self.Relu(self.fc2(x))
        A = self.A(x)
        V = self.V(x)        
        Q = V + (A - torch.mean(A, dim=1, keepdim=True))
        return Q

    def advantage(self, state):
        x = self.Relu(self.fc1(state))
        x = self.Relu(self.fc2(x))
        return self.A(x)


class D3QN:
    def __init__(self, env):
        self.env = env
        # 13 metrics
        self.state_dim = len(env.svcs) * 5
        # svc count: [1~8]
        self.action_dim = len(env.svcs) * 8
        self.q_eval = DuelingDeepQNet(state_dim=self.state_dim, action_dim=self.action_dim).to(device)
        self.q_target = DuelingDeepQNet(state_dim=self.state_dim, action_dim=self.action_dim).to(device)
        self.epsilon = EPSILON
        self.learn_step_counter = 0
        self.start = 0
        self.end = EPSOIDE
        self.rewardgraph = []
        self.buffer = Buffer(BUFFER_SIZE)
    

    def choose_action(self, state):
        state = state.view(1,-1).to(device)
        if np.random.random() > self.epsilon:
            advantage = self.q_eval.advantage(state)
            actionProb = nn.Softmax(dim=-1)(advantage.view(len(self.env.svcs), 8))
            action = (torch.argmax(actionProb, dim=-1, keepdim=False) + 1).view(-1, 1)
        else:
            action = torch.randint(1, 8,(len(self.env.svcs), 1))
        return action


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
        for episode in range(self.start, self.end):
            state = self.simulation_env.get_state()
            qps_df = pd.concat(qps_dfs)[svcs].dropna(axis=0,how='any').astype('float')
            qps_df = qps_df[::10]
            for _, qps_row in qps_df.iterrows():
                # get maximizing action
                self.q_eval.eval()
                action = self.choose_action(state)
                self.q_eval.train()
                # step episode
                qps_input = torch.from_numpy(qps_row[svcs].values).reshape(-1,1)
                next_state, reward = self.simulation_env.new_step(state, qps_input, torch.Tensor.cpu(action))
                # store transition
                self.buffer.append((state.view(1,-1), torch.Tensor.cpu(action), next_state.view(1,-1), torch.FloatTensor([reward]).view(-1,1)))
                state = next_state

                if episode > 1 and self.learn_step_counter % DELAY == 0:           

                    cur_state_batch, _, next_state_batch, reward_batch = self.buffer.sample_batch(BATCH_SIZE)
                    cur_state_batch = torch.cat(cur_state_batch).to(device)
                    reward_batch = torch.cat(reward_batch).to(device)
                    next_state_batch = torch.cat(next_state_batch).to(device)

                    q_pred_batch = self.q_eval(cur_state_batch)
                    q_next = self.q_target(next_state_batch)
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
        self.edge_index = torch.tensor(self.simulation_env.edge_index, dtype=torch.long).to(device)
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
            self.q_eval.eval()
            _state = state.view(1,-1).to(device)
            advantage = self.q_eval.advantage(_state)
            actionProb = nn.Softmax(dim=-1)(advantage.view(len(self.env.svcs), 8))
            action = (torch.argmax(actionProb, dim=-1, keepdim=False) + 1).view(-1, 1)
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
