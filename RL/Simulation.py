import math
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import numpy as np
import networkx as nx
import pandas as pd
import random
from RL.common.StateModel import STATE_MODEL

from config.Config import Config
from util.PrometheusClient import PrometheusClient

'''
    Simulate the environment based on real trace data
'''

state_model_path = 'RL/state.pt'

STATE_SIZE = 7
STATE_LR = 0.001
STATE_EPOCH = 500

torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Simulation:
    def __init__(self, train=False):
        # dependency graph, adjacency matrix
        self.DG, self.edge_index = self.get_call()
        self.svcs = list(self.DG.nodes)
        # handle data
        self.data_df = pd.read_csv('RL/real_trace.csv').dropna(axis=0,how='any')
        self.handle_data()

        if train:
            # train and save the model
            self.train_state_model()
        
        # load the model
        self.state_model = STATE_MODEL(5).to(device)
        self.state_model = torch.load(state_model_path)

        # frontend service
        self.frontend_svc = 'frontend'
        # SLO latency limit
        self.SLO = 500
        # max pod
        self.max_pod = len(self.svcs) * 8
        self.min_pod = len(self.svcs) * 1

    def get_call(self):
        # get the dependence
        prom_util = PrometheusClient(Config())
        DG = prom_util.get_call()
        if DG.has_node('unknown'):
            DG.remove_node('unknown')
        DG.remove_node('redis-cart')
        DG = DG.reverse()
        # print(list(DG))
        adj = nx.to_scipy_sparse_matrix(DG).tocoo()
        row = adj.row.astype(np.int64)
        col = adj.col.astype(np.int64)
        edge_index = np.stack([row, col], axis=0)
        return DG, edge_index

    def get_state(self):
        df = self.data_df
        index = 100
        row = df.iloc[index]
        state = []
        for i in range(len(self.svcs)):
            svc = self.svcs[i]
            if svc == self.frontend_svc:
                self.frontend_index = i
            state.append([row[svc + '&cpu_usage_percentage'],
                    row[svc + '&mem_usage_percentage'],
                    row[svc + '&p90'],
                    row[svc+'&qps'],
                    row[svc+'&count']
                ])
        return torch.tensor(state, dtype=torch.float)

    def new_step(self, old_state, qps, count):
        data = torch.cat((old_state, qps, count), dim=1).float().to(device)
        # normalize
        for i in range(len(self.svcs)):
            svc = self.svcs[i]
            data[i, 0] = self.max_min_normalization(data[i,0], self.max_min_dic[svc+'&cpu_usage_percentage&min'], self.max_min_dic[svc+'&cpu_usage_percentage&max'])
            data[i, 1] = self.max_min_normalization(data[i,1], self.max_min_dic[svc+'&mem_usage_percentage&min'], self.max_min_dic[svc+'&mem_usage_percentage&max'])
            data[i, 2] = self.max_min_normalization(data[i,2], self.max_min_dic[svc+'&p90&min'], self.max_min_dic[svc+'&p90&max'])
            data[i, 3] = self.max_min_normalization(data[i,3], self.max_min_dic[svc+'&qps&min'], self.max_min_dic[svc+'&qps&max'])
            data[i, 4] = self.max_min_normalization(data[i,4], self.max_min_dic[svc+'&count&min'], self.max_min_dic[svc+'&count&max'])
            data[i, 5] = self.max_min_normalization(data[i,5], self.max_min_dic[svc+'&qps&min'], self.max_min_dic[svc+'&qps&max'])
            data[i, 6] = self.max_min_normalization(data[i,6], self.max_min_dic[svc+'&count&min'], self.max_min_dic[svc+'&count&max'])

        edge_index = torch.from_numpy(self.edge_index).to(device)
        # simulate state transition
        self.state_model.eval()
        cpu, mem, p90 = self.state_model(data, edge_index)
        next_state = torch.cat((torch.Tensor.cpu(cpu), torch.Tensor.cpu(mem), torch.Tensor.cpu(p90), qps, count), dim=1).float()
        # back to real value
        for i in range(len(self.svcs)):
            svc = self.svcs[i]
            next_state[i, 0] = self.back_real_value(next_state[i,0], self.max_min_dic[svc+'&cpu_usage_percentage&min'], self.max_min_dic[svc+'&cpu_usage_percentage&max'])
            next_state[i, 1] = self.back_real_value(next_state[i,1], self.max_min_dic[svc+'&mem_usage_percentage&min'], self.max_min_dic[svc+'&mem_usage_percentage&max'])
            next_state[i, 2] = self.back_real_value(next_state[i,2], self.max_min_dic[svc+'&p90&min'], self.max_min_dic[svc+'&p90&max'])
        return next_state.detach(), self.cal_reward(next_state)
    

    def cal_reward(self, state):
        # calculate the reward of latency and the num_pods
        pod_num = torch.sum(state[:,-1], dim=0)
        latency_value = torch.max(state[:,2])
        pod_reward = 1 - ((pod_num - self.min_pod) / (self.max_pod - self.min_pod)) # 0~1
        # if latency_value <= self.SLO:
        #     latency_reward = math.exp((latency_value / self.SLO) -1) # 0~1
        # elif latency_value > self.SLO * 0.8 and latency_value < self.SLO:
        #     latency_reward = math.exp(0.8 - 10 * ((latency_value / self.SLO) - 0.8)) # 
        if latency_value <= self.SLO:
            latency_reward = 1
        else:
            latency_reward = 0
        return (pod_reward + latency_reward).detach()


    def handle_data(self):
        for svc in self.svcs:
            self.data_df[svc + '&cpu_usage_percentage'] = self.data_df[svc+'&cpu_usage'] / self.data_df[svc+'&cpu_limit']
            self.data_df[svc + '&mem_usage_percentage'] = self.data_df[svc+'&mem_usage'] / self.data_df[svc+'&mem_limit']
        # normalize
        self.max_min_dic = {}
        for col in self.data_df.columns:
            if col != 'timestamp':
                self.max_min_dic[col + '&max'] = max(self.data_df[col])
                self.max_min_dic[col + '&min'] = min(self.data_df[col])
                # self.data_df[col] = self.max_min_normalization(self.data_df[col], self.max_min_dic[col + '&min'],self.max_min_dic[col + '&max'])

    def max_min_normalization(self, x, min_value, max_value):
        return (x - min_value) / (max_value - min_value)

    def back_real_value(self, norm_value, min_value, max_value):
        return norm_value * (max_value - min_value) + min_value

    def build_graph_dataset(self, df):
        # build dataset
        datas = []
        for index, row in df.iterrows():
            x = []
            y = []
            if index < len(df)-1:
                next_row = df.iloc[index+1]
                for svc in self.svcs:
                    x.append(
                        [row[svc + '&cpu_usage_percentage'],
                        row[svc + '&mem_usage_percentage'],
                        row[svc + '&p90'],
                        row[svc+'&qps'],
                        row[svc+'&count'],
                        next_row[svc+'&qps'],
                        next_row[svc+'&count']
                        ])
                    y.append([next_row[svc + '&cpu_usage_percentage'],next_row[svc + '&mem_usage_percentage'],next_row[svc + '&p90']])
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                datas.append(Data(x=x, y=y, edge_index=self.edge_index))
        random.shuffle(datas)
        return datas


    def train_state_model(self):
        datas = self.build_graph_dataset(self.data_df)
        train_data = datas[0:int(len(datas)*0.8)]
        test_data = datas[-int(len(datas)*0.2):]
        train_loader = DataLoader(dataset=train_data, batch_size=512,shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=512,shuffle=False)
        
        model = STATE_MODEL(7).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=STATE_LR)

        loss_func = nn.MSELoss()

        # training and testing
        for epoch in range(STATE_EPOCH):
            for step, data in enumerate(train_loader):
                model.train()
                data.to(device)
                cpu, mem, p90 = model(data.x, data.edge_index)
                cpu_loss = torch.sqrt(loss_func(cpu, data.y[:,0].reshape(-1,1).to(device)))
                mem_loss = torch.sqrt(loss_func(mem, data.y[:,1].reshape(-1,1).to(device)))
                p90_loss = torch.sqrt(loss_func(p90, data.y[:,2].reshape(-1,1).to(device)))
                loss = cpu_loss + mem_loss + p90_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss = loss.item()

            for step, data in enumerate(test_loader):
                model.eval()
                data.to(device)
                cpu, mem, p90 = model(data.x, data.edge_index)
                cpu_loss = torch.sqrt(loss_func(cpu, data.y[:,0].reshape(-1,1).to(device)))
                mem_loss = torch.sqrt(loss_func(mem, data.y[:,1].reshape(-1,1).to(device)))
                p90_loss = torch.sqrt(loss_func(p90, data.y[:,2].reshape(-1,1).to(device)))
                loss = cpu_loss + mem_loss + p90_loss
            test_loss = loss.item()
                
            print(f'Epoch: {epoch:03d}, Train_loss: {train_loss:10.8f}, test_loss: {test_loss:10.8f}')

        torch.save(model, 'RL/state.pt')
        