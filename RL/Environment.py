import pandas as pd
import numpy as np
from config.Config import Config
from util.PrometheusClient import PrometheusClient
from util.KubernetesClient import KubernetesClient
import time
import networkx as nx
import torch


NUM_RESOURCES = 2

class Environment():
    def __init__(self, config: Config):
        self.config = config
        self.prom_util = PrometheusClient(config)
        self.k8s_util = KubernetesClient(config)
        # dependency graph, adjacency matrix
        self.DG, self.edge_index = self.get_call()
        # svcs
        self.svcs = list(self.DG.nodes)
        # frontend service
        self.frontend_svc = 'frontend'
        # SLO latency limit
        self.SLO = 200
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
        print(list(DG))
        adj = nx.to_scipy_sparse_matrix(DG).tocoo()
        row = adj.row.astype(np.int64)
        col = adj.col.astype(np.int64)
        edge_index = np.stack([row, col], axis=0)
        return DG, edge_index

    def get_state(self):
        self.prom_util.set_time_range(int(round(time.time())) - 30, int(round(time.time())))
        # get state from the cluster
        latency = self.prom_util.get_svc_latency()
        metric = self.prom_util.get_svc_metric().T.to_dict()[0]
        qps = self.prom_util.get_svc_qps()
        count = self.k8s_util.get_svcs_counts()

        state_dic = {}
        
        state = []
        for i in range(len(self.svcs)):
            svc = self.svcs[i]
            if svc == self.frontend_svc:
                self.frontend_index = i
            svc_state = []
            svc_state.append(metric[svc+'&cpu_usage']/metric[svc+'&cpu_limit'])
            svc_state.append(metric[svc+'&mem_usage']/metric[svc+'&mem_limit'])  
            svc_state.append(latency[svc + '&p90'])
            svc_state.append(qps[svc + '&qps'])
            svc_state.append(count[svc])
                
            state_dic[svc+'&cpu_usage_percentage'] = svc_state[0]
            state_dic[svc+'&mem_usage_percentage'] = svc_state[1]
            state_dic[svc+'&p90'] = svc_state[2]
            state_dic[svc+'&qps'] = svc_state[3]
            state_dic[svc+'&count'] = svc_state[4]

            state.append(svc_state)
        
        return torch.FloatTensor(state), pd.DataFrame([state_dic])

    def new_step(self, action):
        # execute action
        action = action.squeeze().numpy().tolist()
        for i in range(len(self.svcs)):
            self.k8s_util.patch_scale(self.svcs[i], action[i])
        while True:
            time.sleep(1)
            if self.k8s_util.all_avaliable():
                break
        state, state_df = self.get_state()
        return state, state_df, self.cal_reward(state)

    def cal_reward(self, state):
        # calculate the reward of latency and the num_pods
        pod_num = torch.sum(state[:,-1], dim=0)
        latency_value = state[self.frontend_index,2]
        pod_reward = 1 - ((pod_num - self.min_pod) / (self.max_pod - self.min_pod)) # 0~1
        if latency_value <= self.SLO:
            latency_reward = 1
        else:
            latency_reward = 0
        return (pod_reward + latency_reward).detach()
