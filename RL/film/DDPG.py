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

# Files
from RL.film.noise import OrnsteinUhlenbeckActionNoise as OUNoise
from RL.film.replaybuffer import Buffer
from RL.film.actorcritic import Actor, Critic

PLOT_FIG = True

# Hyperparameters
ACTOR_LR = 0.03
CRITIC_LR = 0.03
# ACTOR_LR = 0.01
# CRITIC_LR = 0.03
MINIBATCH_SIZE = 256
NUM_EPISODES = 10000
NUM_TIMESTEPS = 100
MU = 0
SIGMA = 0.2
CHECKPOINT_DIR = './RL/film/checkpoints/test/'
BUFFER_SIZE = 100000
DISCOUNT = 0.9
TAU = 0.001
WARMUP = 70
EPSILON = 1.0
EPSILON_DECAY = 1e-6
CUDA = True

def obs2state(state_list):
    return torch.FloatTensor(state_list).view(1, -1)

class DDPG:
    def __init__(self, env):
        self.env = env
        # 13 metrics
        self.stateDim = len(env.svcs) * 5
        # svc count: [1~8]
        self.actionDim = len(env.svcs) * 8
        self.replayBuffer = Buffer(BUFFER_SIZE)
        if CUDA:
            self.actor = Actor(self.stateDim, self.actionDim).cuda()
            self.critic = Critic(self.stateDim, self.actionDim).cuda()
            self.targetActor = deepcopy(Actor(self.stateDim, self.actionDim)).cuda()
            self.targetCritic = deepcopy(Critic(self.stateDim, self.actionDim)).cuda()
        else:
            self.actor = Actor(self.stateDim, self.actionDim)
            self.critic = Critic(self.stateDim, self.actionDim)
            self.targetActor = deepcopy(Actor(self.stateDim, self.actionDim))
            self.targetCritic = deepcopy(Critic(self.stateDim, self.actionDim))
        self.actorOptim = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.criticOptim = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.criticLoss = nn.MSELoss()
        self.noise = OUNoise(mu=np.zeros(self.actionDim), sigma=SIGMA)
        self.batchSize = MINIBATCH_SIZE
        self.checkpoint_dir = CHECKPOINT_DIR
        self.discount = DISCOUNT
        self.warmup = WARMUP
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.rewardgraph = []
        self.start = 0
        self.end = NUM_EPISODES

    # Inputs: Batch of next states, rewards and terminal flags of size self.batchSize
    # Target Q-value <- reward and bootstraped Q-value of next state via the target actor and target critic
    # Output: Batch of Q-value targets
    def getQTarget(self, nextStateBatch, rewardBatch): 
        nextActionBatch = self.targetActor(nextStateBatch)
        qNext = self.targetCritic(nextStateBatch, nextActionBatch)
        
        # nonFinalMask = self.discount * nonFinalMask.type(torch.FloatTensor)
        # targetBatch += nonFinalMask * qNext.squeeze().data
        targetBatch = rewardBatch + (self.discount * qNext.data)
        
        return Variable(targetBatch)

    # weighted average update of the target network and original network
    # Inputs: target actor(critic) and original actor(critic)
    def updateTargets(self, target, original):
        for targetParam, orgParam in zip(target.parameters(), original.parameters()):
            targetParam.data.copy_((1 - TAU)*targetParam.data + TAU*orgParam.data)

    # Inputs: Current state of the episode
    # Output: the action which maximizes the Q-value of the current state-action pair
    def getMaxAction(self, curState):
        if CUDA:
            noise = (self.epsilon * Variable(torch.FloatTensor(self.noise()))).cuda()
        else:
            noise = (self.epsilon * Variable(torch.FloatTensor(self.noise())))
        action = self.actor(curState)
        actionNoise = action + noise
        
        # get the max
        # action_list = actionNoise.tolist()[0]
        # max_action = max(action_list)
        # max_index = action_list.index(max_action)
        actionProb = nn.Softmax(dim=-1)(actionNoise.view(len(self.env.svcs), 8))
        final_action = torch.argmax(actionProb, dim=-1, keepdim=False) + 1
        return torch.Tensor.cpu(final_action), actionNoise.detach()

    # training of the original and target actor-critic networks
    def train(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        print('Training started...')

        # for each episode 
        for episode in range(self.start, self.end):
            state = self.env.get_state()
            
            ep_reward = 0

            # simulation: qps
            gradually_decrease_df = pd.read_csv('./traffic/simulation/gradually_decrease.csv')
            gradually_increase_df = pd.read_csv('./traffic/simulation/gradually_increase.csv')
            single_peak_df = pd.read_csv('./traffic/simulation/single_peak.csv')
            # true: qps
            wiki_df = pd.read_csv('./traffic/true/wiki.csv')
            qps_dfs = [single_peak_df, gradually_increase_df, gradually_decrease_df, wiki_df]
            svcs = self.env.svcs
            for episode in range(self.start, self.end):
                # randomize qps pattern
                # random.shuffle(qps_dfs)
                qps_df = pd.concat(qps_dfs)[svcs].fillna(0).astype('float')
                qps_df = qps_df[::10]
                state = self.env.get_state()
                ep_reward = 0
                step = 0
                for _, qps_row in qps_df.iterrows():
                    step+=1
                    # get maximizing action
                    if CUDA:
                        currStateTensor = Variable(obs2state(state)).cuda()
                    else:
                        currStateTensor = Variable(obs2state(state))
                    self.actor.eval()     
                    action, actionToBuffer = self.getMaxAction(currStateTensor)
                    action = (action).view(-1, 1)
                    self.actor.train() 
                    # step episode
                    qps_input = torch.from_numpy(qps_row[svcs].values).reshape(-1,1)
                    state, reward = self.env.new_step(state, qps_input, torch.Tensor.cpu(action))
                    # print(len(self.replayBuffer),'Reward: {}'.format(reward))             
                    
                    if CUDA:
                        nextState = Variable(obs2state(state)).cuda()
                    else:
                        nextState = Variable(obs2state(state))
                    ep_reward = ep_reward + reward

                    if CUDA:
                        reward = torch.FloatTensor([reward]).view(-1,1).cuda()
                    else:
                        reward = torch.FloatTensor([reward]).view(-1,1)
                    # Update replay bufer
                    self.replayBuffer.append((currStateTensor, actionToBuffer, nextState, reward))
                    
                # Training loop
                curStateBatch, actionBatch, nextStateBatch, rewardBatch = self.replayBuffer.sample_batch(self.batchSize)
                curStateBatch = torch.cat(curStateBatch)
                actionBatch = torch.cat(actionBatch)
                rewardBatch = torch.cat(rewardBatch)
                nextStateBatch = torch.cat(nextStateBatch)

                qPredBatch = self.critic(curStateBatch, actionBatch)
                qTargetBatch = self.getQTarget(nextStateBatch, rewardBatch)
                criticLoss = self.criticLoss(qPredBatch, qTargetBatch)

                # Critic update
                self.criticOptim.zero_grad()
                # print('Critic Loss: {}'.format(criticLoss))
                criticLoss.backward(retain_graph=True)
                self.criticOptim.step()

                _curStateBatch = curStateBatch.detach()
                a = self.actor(_curStateBatch)
                q = self.critic(_curStateBatch, a)
                actorLoss = -torch.mean(q)

                # Actor update
                self.actorOptim.zero_grad()
                # print('Actor Loss: {}'.format(actorLoss))
                actorLoss.backward(retain_graph=True)
                self.actorOptim.step()

                # Update Targets                        
                # self.updateTargets(self.targetActor, self.actor)
                # self.updateTargets(self.targetCritic, self.critic)
                if episode % 5 == 0:
                    self.targetActor.load_state_dict(self.actor.state_dict())
                    self.targetCritic.load_state_dict(self.critic.state_dict())
                self.epsilon -= self.epsilon_decay
                
                # test
                if episode % 10 == 0:
                    ep_reward = self.test()
                    print("EP -", episode, "| Total Reward -", ep_reward.detach().numpy())
                    self.rewardgraph.append(ep_reward)
    
                # save to checkpoints
                if episode % 100 == 0:
                    self.save_checkpoint(episode)

                if PLOT_FIG:
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
        svcs = self.env.svcs
        state = self.env.get_state()
        qps_df = pd.concat(qps_dfs)[svcs].dropna(axis=0,how='any').astype('float')
        qps_df = qps_df[::10]
        for _, qps_row in qps_df.iterrows():
            # get maximizing action
            self.actor.eval()
            _state = state.view(1,-1).cuda()
            action_prob = self.actor(_state).view(-1,1)
            action_prob = nn.Softmax(dim=-1)(action_prob.view(len(self.env.svcs), 8))
            action = torch.argmax(action_prob, dim=-1, keepdim=False) + 1
            action = action.view(-1,1)
            # step episode
            qps_input = torch.from_numpy(qps_row[svcs].values).reshape(-1,1)
            next_state, reward = self.env.new_step(state, qps_input, torch.Tensor.cpu(action))
            state = next_state
            ep_reward += reward
        return ep_reward

    def save_checkpoint(self, episode_num):
        checkpointName = self.checkpoint_dir + 'ep{}.pth.tar'.format(episode_num)
        checkpoint = {
            'episode': episode_num,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'targetActor': self.targetActor.state_dict(),
            'targetCritic': self.targetCritic.state_dict(),
            'actorOpt': self.actorOptim.state_dict(),
            'criticOpt': self.criticOptim.state_dict(),
            'replayBuffer': self.replayBuffer,
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
            self.replayBuffer = checkpoint['replayBuffer']
            self.rewardgraph = checkpoint['rewardgraph']
            self.epsilon = checkpoint['epsilon']
            print('Checkpoint loaded')
        else:
            raise OSError('Checkpoint not found')
