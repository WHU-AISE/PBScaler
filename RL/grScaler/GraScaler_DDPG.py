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

# Files
from RL.grScaler.replaybuffer import Buffer
from RL.common.MPNN import MPNN
from RL.common.GAT import GCNCell, GATCell
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
CHECKPOINT_DIR = './RL/grScaler/checkpoints/ddpg/5/'
STATE_MODEL_PATH = './RL/state.pt'
BUFFER_SIZE = 2**20
DISCOUNT = 0.9
TAU = 0.1
EPSILON = 1.0
EPSILON_DECAY = 1e-5
DELAY = 10
# noise
NOISE_CLIP = 0.1
POLICY_NOISE = 0.2

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

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.mpnn = GATCell(state_dim, HIDDEN_LAYER)
        self.bn0 = nn.BatchNorm1d(HIDDEN_LAYER)
                                    
        self.fc1 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.fc1.weight.data = nn.init.normal_(self.fc1.weight.data, std=0.1)
                                    
        self.fc2 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.fc2.weight.data = nn.init.normal_(self.fc2.weight.data, std=0.1)                                    
        self.bn2 = nn.BatchNorm1d(HIDDEN_LAYER)
                                    
        self.fc3 = nn.Linear(HIDDEN_LAYER, action_dim)
        self.fc3.weight.data = nn.init.normal_(self.fc3.weight.data, std=0.1)                             
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
            
    def forward(self, x, edge_index):
        h0 = self.relu1(self.mpnn(x, edge_index))
        h0_norm = self.bn0(h0)
        h1 = self.relu2(self.fc1(h0_norm))
        h2 = self.relu3(self.fc2(h1))
        h2_norm = self.bn2(h2)
        action = self.fc3(h2_norm)
        return F.softmax(action,dim=1)
        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.mpnn = GCNCell(state_dim, HIDDEN_LAYER)
        self.bn0 = nn.BatchNorm1d(HIDDEN_LAYER)
        
        self.fc1 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.fc1.weight.data = nn.init.normal_(self.fc1.weight.data, std=0.1)
        self.bn1 = nn.BatchNorm1d(HIDDEN_LAYER)

        self.fc2 = nn.Linear(action_dim, HIDDEN_LAYER)
        self.fc2.weight.data = nn.init.normal_(self.fc2.weight.data, std=0.1)
        self.bn2 = nn.BatchNorm1d(HIDDEN_LAYER)
        
        self.fc3 = nn.Linear(HIDDEN_LAYER * 2, 1)
        self.fc3.weight.data = nn.init.normal_(self.fc3.weight.data, std=0.1)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        
    def forward(self, x, edge_index, batch_size, action):
        h0 = self.relu1(self.mpnn(x, edge_index))
        h0_norm = self.bn0(h0)
        h1 = self.relu2(self.fc1(h0_norm))
        h1_norm = self.bn1(h1)
        h2 = self.relu3(self.fc2(action))
        h2_norm = self.bn2(h2)
        x = global_mean_pool(torch.cat([h1_norm, h2_norm], dim=1), batch_size)  # [batch_size, hidden_channels]
        Qval = self.fc3(x)
        return Qval

class GrScaler:
    def __init__(self, env: Environment, transfer=False):
        self.env = env
        # 5 metrics
        self.state_dim = 5
        # svc count: [1~8]
        self.action_dim = 8
        self.buffer = Buffer(BUFFER_SIZE)
        self.critic_loss = nn.MSELoss()
        self.rewardgraph = []
        self.start = 0
        self.end = NUM_EPISODES
        self.epsilon = EPSILON

        self.build_net()

         
    def build_net(self):
        # 6 networks
        self.actor = Actor(self.state_dim, self.action_dim).to(device)
        self.targetActor = Actor(self.state_dim, self.action_dim).to(device)
        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.targetCritic = Critic(self.state_dim, self.action_dim).to(device)

        self.actorOptim = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.criticOptim = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

    def transfer(self):
        # transfer the mpnn model
        state_model = torch.load(STATE_MODEL_PATH)
        self.actor.mpnn = state_model.mpnn
        self.targetActor.mpnn = state_model.mpnn
        self.critic.mpnn = state_model.mpnn
        self.targetCritic.mpnn = state_model.mpnn


    def get_Q_target(self, next_graph_data, reward_batch): 
        next_action_prob = self.get_action_prob(self.targetActor, next_graph_data.x, next_graph_data.edge_index)
        qNext = self.targetCritic(next_graph_data.x, next_graph_data.edge_index, next_graph_data.batch, next_action_prob)
        targetBatch = reward_batch + (DISCOUNT * qNext.data)
        return Variable(targetBatch)

    def update_targets(self, target, original):
        for targetParam, orgParam in zip(target.parameters(), original.parameters()):
            targetParam.data.copy_((1 - TAU)*targetParam.data + TAU*orgParam.data)

    def build_graph_data(self, state, edge_index, action=None):
        if action is not None:
            return Data(x=state, y=action, edge_index=edge_index)
        else:
            return Data(x=state, edge_index=edge_index)

#     def choose_action(self, agent, state, edge_index):
#         action_prob = self.get_action_prob(agent, state, edge_index)
#         self.noise = OUNoise(mu=np.zeros(action_prob.shape), sigma=SIGMA)
#         noise = (self.epsilon * Variable(torch.FloatTensor(self.noise()))).to(device)
#         action_noise = action_prob + noise
#         action = (torch.argmax(action_noise, dim=-1, keepdim=False) + 1).view(-1,1)
#         return action, action_noise.detach()

#     def get_action_prob(self, agent, state, edge_index):
#         state = state.to(device)
#         edge_index = edge_index.to(device)
#         action_prob = agent(state, edge_index)
#         return action_prob

    def choose_action(self, agent, state, edge_index):
        action_prob = self.get_action_prob(agent, state, edge_index)
        action = (torch.argmax(action_prob, dim=-1, keepdim=False) + 1).view(-1,1)
        return action, action_prob.detach()

    def get_action_prob(self, agent, state, edge_index):
        state = state.to(device)
        edge_index = edge_index.to(device)
        action_prob = agent(state, edge_index)
        noise = torch.ones_like(action_prob).data.normal_(0, POLICY_NOISE).to(device)
        noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
        return action_prob + noise

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
        for episode in range(self.start, self.end):
            state = simulation_env.get_state()
            ep_reward = 0
            qps_df = pd.concat(qps_dfs)[svcs].fillna(0).astype('float')
            qps_df = qps_df[::10]
            state = simulation_env.get_state()

            step = 0

            for _, qps_row in qps_df.iterrows():
                # get maximizing action
                self.actor.eval()
                action, action_prob = self.choose_action(self.actor, state, self.edge_index)
                self.actor.train()
                # step episode
                qps_input = torch.from_numpy(qps_row[svcs].values).reshape(-1,1)
                next_state, reward = simulation_env.new_step(state, qps_input, torch.Tensor.cpu(action))
                # store transition
                self.buffer.append((state, torch.Tensor.cpu(action_prob), next_state, torch.FloatTensor([reward]).view(-1,1)))
                state = next_state
                ep_reward += reward
                step += 1
                    
                # Training loop
                if episode > 1 and step % DELAY == 0:
                    cur_state_batch, action_prob_batch, next_state_batch, reward_batch = self.buffer.sample_batch(MINIBATCH_SIZE)
                    cur_graph_datas = GraphData(cur_state_batch, self.edge_index).get_batch_data(MINIBATCH_SIZE).to(device)
                    next_graph_datas = GraphData(next_state_batch, self.edge_index).get_batch_data(MINIBATCH_SIZE).to(device)

                    action_prob_batch = torch.cat(action_prob_batch).to(device)
                    reward_batch = torch.cat(reward_batch).to(device)

                    q_pred_batch = self.critic(cur_graph_datas.x, cur_graph_datas.edge_index, cur_graph_datas.batch, action_prob_batch)
                    q_target_batch = self.get_Q_target(next_graph_datas, reward_batch)
                    critic_loss = self.critic_loss(q_pred_batch, q_target_batch)

                    # Critic update
                    self.criticOptim.zero_grad()
                    critic_loss.backward()
                    self.criticOptim.step()

                    _cur_state_batch = cur_graph_datas.x.detach()
                    a = self.get_action_prob(self.actor, _cur_state_batch, cur_graph_datas.edge_index)
                    q = self.critic(_cur_state_batch, cur_graph_datas.edge_index, cur_graph_datas.batch, a)
                    actor_loss = -torch.mean(q)

                    # Actor update
                    self.actorOptim.zero_grad()
                    actor_loss.backward()
                    self.actorOptim.step()

                    # Update Targets                        
                    self.update_targets(self.targetActor, self.actor)
                    self.update_targets(self.targetCritic, self.critic)

                    if step % 500 == 0:
                        print('Actor Loss: {}'.format(actor_loss), 'critic Loss: {}'.format(critic_loss))

            
            print("EP -", episode, "| Total Reward -", ep_reward.detach().numpy())
    
            # save to checkpoints
            if episode % 100 == 0:
                self.save_checkpoint(episode)
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


    def save_checkpoint(self, episode_num):
        checkpointName = CHECKPOINT_DIR + 'ep{}.pth.tar'.format(episode_num)
        checkpoint = {
            'episode': episode_num,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'targetActor': self.targetActor.state_dict(),
            'targetCritic': self.targetCritic.state_dict(),
            'actorOpt': self.actorOptim.state_dict(),
            'criticOpt': self.criticOptim.state_dict(),
            'buffer': self.buffer,
            'rewardgraph': self.rewardgraph,
            'epsilon': self.epsilon            
        } 
        torch.save(checkpoint, checkpointName)
    
    def loadCheckpoint(self, checkpointName):
        if os.path.isfile(checkpointName):
            print("Loading checkpoint...")
            checkpoint = torch.load(checkpointName)
            self.start = checkpoint['episode'] + 1
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.targetActor.load_state_dict(checkpoint['targetActor'])
            self.targetCritic.load_state_dict(checkpoint['targetCritic'])
            self.actorOptim.load_state_dict(checkpoint['actorOpt'])
            self.criticOptim.load_state_dict(checkpoint['criticOpt'])
            self.buffer = checkpoint['buffer']
            self.rewardgraph = checkpoint['rewardgraph']
            self.epsilon = checkpoint['epsilon']
            print('Checkpoint loaded')
        else:
            raise OSError('Checkpoint not found')
