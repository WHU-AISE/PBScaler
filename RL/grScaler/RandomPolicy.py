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
import time
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
import csv

# Files
from RL.Simulation import Simulation

from RL.grScaler.replaybuffer import Buffer
from RL.common.MPNN import MPNN
from RL.common.StateModel import STATE_MODEL
from RL.grScaler.GraphData import GraphData
from RL.grScaler.noise import OrnsteinUhlenbeckActionNoise as OUNoise

PLOT_FIG = True

# Hyperparameters
ACTOR_LR = 0.0003
CRITIC_LR = 0.0003
MINIBATCH_SIZE = 256
NUM_EPISODES = 10000
MU = 0
SIGMA = 0.2
CHECKPOINT_DIR = './RL/grScaler/checkpoints/random/'
STATE_MODEL_PATH = './RL/state.pt'
BUFFER_SIZE = 2**20
DISCOUNT = 0.9
TAU = 0.1
EPSILON = 1.0
EPSILON_DECAY = 1e-5

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

class GrScaler:
    def __init__(self, env: Environment, transfer=False):
        self.action_dim=7
        self.env = env
        self.rewardgraph = []
        self.begin = 0
        self.end = NUM_EPISODES

    def start(self, time_end):
        last_action = None
        print('随即策略')
        while True:
            now_time = time.time()
            if now_time > time_end:
                break
            try:
                state, state_df = self.env.get_state()
                action = torch.Tensor.cpu(self.choose_action(state, last_action))
#                 for i in range(len(self.env.svcs)):
#                     svc = self.env.svcs[i]
#                     state_df[svc+'_action'] = action.squeeze().numpy().tolist()[i]
                _, next_state_df, _ = self.env.new_step(action)

#                 for col in next_state_df.columns:
#                     state_df['next_'+col] = next_state_df[col]
#                 if not os.path.exists('state_trans.csv'):
#                     state_df.to_csv('state_trans.csv')
#                 else:
#                     state_df.to_csv('state_trans.csv', mode='a', header=False)
                last_action = action
                sleep(60)
            except:
                last_action = None
                continue


    def choose_action(self, state, last_action=None):
        if last_action is None:
            return torch.randint(1, self.action_dim + 1,(state.shape[0], 1))
        else:
            add_a = torch.randint(-2, 2,(state.shape[0], 1))
            a = last_action + add_a
            return torch.clamp(a, 1, 8)

    def train(self):
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
            
        print('Training started...')

        simulation_env = Simulation(train=False)
        self.edge_index = torch.tensor(simulation_env.edge_index, dtype=torch.long)
        # simulation: qps
        gradually_decrease_df = pd.read_csv('./traffic/simulation/gradually_decrease.csv')
        gradually_increase_df = pd.read_csv('./traffic/simulation/gradually_increase.csv')
        sudden_decrease_df = pd.read_csv('./traffic/simulation/sudden_decrease.csv')
        sudden_increase_df = pd.read_csv('./traffic/simulation/sudden_increase.csv')
        single_peak_df = pd.read_csv('./traffic/simulation/single_peak.csv')
        qps_dfs = [single_peak_df, sudden_increase_df, sudden_decrease_df, gradually_increase_df, gradually_decrease_df]
        svcs = simulation_env.svcs

        # for each episode 
        for episode in range(self.begin, self.end):
            state = simulation_env.get_state()
            ep_reward = 0
            qps_df = pd.concat(qps_dfs)[svcs].fillna(0).astype('float')
            qps_df = qps_df[::10]
            state = simulation_env.get_state()

            step = 0
            rewards = []

            for _, qps_row in qps_df.iterrows():
                # get maximizing action
                action = torch.randint(1, self.action_dim + 1,(state.shape[0], 1))
                # step episode
                qps_input = torch.from_numpy(qps_row[svcs].values).reshape(-1,1)
                next_state, reward = simulation_env.new_step(state, qps_input, torch.Tensor.cpu(action))
                # store transition
                state = next_state
                ep_reward += reward
                rewards.append(reward)
                step += 1

            print("EP -", episode, "| Total Reward -", ep_reward.detach().numpy())

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