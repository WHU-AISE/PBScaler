import math
import time
import numpy as np
import schedule
from copy import deepcopy
import networkx as nx
import scipy.stats
import joblib
from config.Config import Config
from util.GA import *
import warnings
from util.KubernetesClient import KubernetesClient
from util.PrometheusClient import PrometheusClient

warnings.filterwarnings("ignore")

def coast_time(func):
    def fun(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        print(f'func {func.__name__} coast time:{time.perf_counter() - t:.8f} s')
        return result
    return fun

CONF = 0.05
# anomaly detection window -- 15 seconds
AB_CHECK_INTERVAL = 15
# waste detection window -- 120 seconds
WASTE_CHECK_INTERVAL = 120
ALPHA = 0.2
BETA = 0.9
K = 2


class PBScaler:
    def __init__(self, config: Config, simulation_model_path: str):
        # the prometheus client && k8s client
        self.config = config
        self.prom_util = PrometheusClient(config)
        self.k8s_util = KubernetesClient(config)
        # simulation environment
        self.predictor = joblib.load(simulation_model_path)
        # args
        self.SLO = config.SLO
        self.max_num = config.max_pod
        self.min_num = config.min_pod
        # microservices
        self.mss = self.k8s_util.get_svcs_without_state()
        self.roots = None
        self.svc_counts = None
        

    @coast_time
    def anomaly_detect(self):
        """ SLO check
        """
        # get service count
        self.svc_counts = self.k8s_util.get_svcs_counts()
        ab_calls = self.get_abnormal_calls()
        # abnormal
        if len(ab_calls) > 0:
            self.root_analysis(ab_calls=ab_calls)

    @coast_time
    def waste_detection(self):
        """waste detection
        """
        # check the front SLO
        ab_calls = self.get_abnormal_calls()
        if len(ab_calls) > 0:
            return
        # check qps
        cur_time = int(round(time.time()))
        self.prom_util.set_time_range(cur_time - 60, cur_time)
        now_qps_df = self.prom_util.get_svc_qps_range()

        self.prom_util.set_time_range(cur_time - 300, cur_time - 60)
        old_qps_df = self.prom_util.get_svc_qps_range()

        # select the waste pods
        waste_mss = []

        '''Hypothesis µ testing
        a: now_qps µ
        b: old_qps µ0
        H0: µ >= µ0
        H1: µ < µ0
        '''
        for svc in self.mss:
            if svc in now_qps_df.columns and svc in old_qps_df.columns:
                t, p = scipy.stats.ttest_ind(now_qps_df[svc], old_qps_df[svc] * BETA, equal_var=False)
                if t < 0 and p <= CONF:
                    waste_mss.append(svc)
        self.roots = list(filter(lambda ms: self.svc_counts[ms] > self.min_num, waste_mss))
        if len(self.roots) != 0:
            self.choose_action(option='reduce')

    def get_abnormal_calls(self):
        # get all call latency last 1min
        begin = int(round((time.time() - 60)))
        end = int(round(time.time()))
        self.prom_util.set_time_range(begin, end)
        # slo Hypothesis testing
        ab_calls = []
        call_latency = self.prom_util.get_call_latency()
        for call, latency in call_latency.items():
            if latency > self.SLO * (1 + ALPHA / 2):
                ab_calls.append(call)
        # call_latency = self.prom_util.get_call_p90_latency_range()
        # for call, latency in call_latency.iteritems():
        #     if call != 'timestamp':
        #         _, p = scipy.stats.ttest_1samp(latency.values, self.SLO * self.alpha, alternative='greater')
        #         if p < self.conf:
        #             ab_calls.append(call)
        return ab_calls

    @coast_time
    def root_analysis(self, ab_calls):
        """ locate the root cause
        1. build the abnormal subgraph
        2. calculate pr score
        3. sort and return the root cause
        Args:
            :param ab_calls: abnormal call edge
            :param n: top n root causes
        """
        ab_dg, personal_array = self.build_abnormal_subgraph(ab_calls)
        nodes = [node for node in ab_dg.nodes]
        if len(nodes) == 1:
            res = [(nodes[0], 1)]
        else:
            res = nx.pagerank(ab_dg, alpha=0.85, personalization=personal_array, max_iter=1000)
            res = sorted(res.items(), key=lambda x: x[1], reverse=True)
        res = [ms for ms, _ in res]
        self.roots = list(filter(lambda root: self.svc_counts[root] + 1 < self.max_num, res))[:K]
        # trigger the choose_action
        if len(self.roots) != 0:
            self.choose_action('add')

    @coast_time
    def choose_action(self, option='add'):
        """ choose action
        Args:
            :param option: [add, reduce], add or reduce the replica
        """
        mss = deepcopy(self.mss)
        roots = deepcopy(self.roots)
        r = deepcopy(self.svc_counts)
        workloads = []

        dim = len(roots)
        if option == 'add':
            print('begin scale out')
            min_array, max_array = [r[t]+1 for t in self.roots if r[t] < self.max_num] , [self.max_num] * dim
        elif option == 'reduce':
            print('begin scale in')
            min_array, max_array = [self.min_num] * dim, [r[t] for t in roots if r[t] > self.min_num]
            thesold_array = np.array(max_array) - 1
            min_array = np.maximum(min_array, thesold_array)
        else:
            raise NotImplementedError()
        qps = self.prom_util.get_svc_qps()
        for ms in mss:
            if ms + '&qps' in qps.keys():
                workloads.append(qps[ms + '&qps'])
            else:
                workloads.append(0)
        '''
        optimizing with genetic algorithms
        only optimize the root services
        TODO: base the root score
        '''
        opter = GA('/home/ubuntu/xsy/experiment/autoscaling/simulation/train_ticket/RandomForestClassify.model', dim, min_array, max_array, 'max', size_pop=50, max_iter=5, prob_cross=0.9, prob_mut=0.01, precision=1, encoding='BG', selectStyle='tour', recStyle='xovdp', mutStyle='mutbin', seed=1)

        opter.set_env(workloads, mss, roots, r)
        res = opter.evolve()

        actions = deepcopy(self.svc_counts)
        for i in range(dim):
            svc = self.roots[i]
            actions[svc] = res[i]

        self.execute_task(actions)

    # def fitness(self, action):
    #     """
    #     calculate the reward
    #     :param action: the iterative result
    #     :return: reward
    #     """
    #     root_mss = copy.deepcopy(self.roots)
    #     x = []
    #     for i in range(len(self.mss)):
    #         ms = self.mss[i]
    #         ms_qps_key = ms + '&qps'
    #         ms_qps = 0 if ms_qps_key not in self.qps.keys() else self.qps[ms + '&qps']
    #         if ms in root_mss:
    #             # root services: the iterative result
    #             x.extend([i, ms_qps, action[root_mss.index(ms)]])
    #         else:
    #             # the initial count
    #             x.extend([i, ms_qps, self.svc_counts[ms]])
    #     x = np.array(x).reshape(1, -1)
    #     slo_reward = self.predictor.predict(x).tolist()[0]
    #     cost_reward = 1 - (np.sum(x) / (self.max_num * len(self.mss)))
    #     return - (slo_reward + cost_reward)
        # return - self.predictor.predict(x).tolist()[0]

    def build_abnormal_subgraph(self, ab_calls):
        """
            1. collect metrics for all abnormal services
            2. build the abnormal subgraph with abnormal calls
            3. weight the c by Pearson correlation coefficient
        """
        ab_sets = set()
        for ab_call in ab_calls:
            ab_sets.update(ab_call.split('_'))
        if 'unknown' in ab_sets: ab_sets.remove('unknown')
        if 'istio-ingressgateway' in ab_sets: ab_sets.remove('istio-ingressgateway')
        ab_mss = list(ab_sets)
        ab_mss.sort()
        begin = int(round((time.time() - 60)))
        end = int(round(time.time()))
        self.prom_util.set_time_range(begin, end)
        ab_metric_df = self.prom_util.get_svc_metric_range()
        ab_svc_latency_df = self.prom_util.get_svc_p90_latency_range()
        ab_svc_latency_df = ab_svc_latency_df[[col for col in ab_svc_latency_df.columns if col in ab_mss]]

        ab_dg = nx.DiGraph()
        ab_dg.add_nodes_from(ab_mss)
        edges = []
        for ab_call in ab_calls:
            edge = ab_call.split('_')
            if 'unknown' in edge or 'istio-ingressgateway' in edge:
                continue
            metric_df = ab_metric_df[[col for col in ab_metric_df.columns if col.startswith(edge[1])]]
            edges.append((edge[0], edge[1], self.cal_weight(ab_svc_latency_df[edge[0]], metric_df)))
        ab_dg.add_weighted_edges_from(edges)

        # calculate topology potential
        anomaly_score_map = {}
        for node in ab_mss:
            e_latency_array = ab_svc_latency_df[node]
            ef = e_latency_array[e_latency_array > self.SLO].count()
            anomaly_score_map[node] = ef
        personal_array = self.cal_topology_potential(ab_dg, anomaly_score_map)

        return ab_dg, personal_array

    def cal_weight(self, latency_array, metric_df):
        max_corr = 0
        for col in metric_df.columns:
            temp = abs(metric_df[col].corr(latency_array))
            if temp > max_corr:
                max_corr = temp
        return max_corr

    # def get_neigbors(self, g, node, depth=1):
    #     output = {}
    #     layers = dict(nx.bfs_successors(g, source=node, depth_limit=depth))
    #     nodes = [node]
    #     for i in range(1, depth + 1):
    #         output[i] = []
    #         for x in nodes:
    #             output[i].extend(layers.get(x, []))
    #         nodes = output[i]
    #     return output
    
    def cal_topology_potential(self, ab_DG, anomaly_score_map: dict):
        personal_array = {}
        for node in ab_DG.nodes:
            # calculate topological potential
            sigma = 1
            potential = anomaly_score_map[node]
            pre_nodes = ab_DG.predecessors(node)
            for pre_node in pre_nodes:
                potential += (anomaly_score_map[pre_node] * math.exp(-1 * math.pow(1/sigma, 2)))
                for pre2_node in ab_DG.predecessors(pre_node):
                    if pre2_node != node:
                        potential += (anomaly_score_map[pre2_node] * math.exp(-1 * math.pow(2 / sigma, 2)))
            personal_array[node] = potential
        return personal_array

    # AAMR http://ksiresearch.org/seke/seke21paper/paper091.pdf
    # def cal_ixscore(self, dg, ab_dg, anomaly_score_map: dict):
    #     ix_all = {}
    #     scores = []
    #     for ms in ab_dg.nodes:
    #         x = anomaly_score_map[ms]
    #         neighbors = list(dg.neighbors(ms))
    #         AANs = 0
    #         for AAN in neighbors:
    #             AS = anomaly_score_map[AAN] if AAN in anomaly_score_map else 1
    #             AANs += AS
    #         i_score = AANs / dg.degree(ms)

    #         n2 = self.get_neigbors(dg, ms, 2)[2]
    #         NHANs = 0
    #         degree2sum = 1
    #         for NHAN in n2:
    #             AS = anomaly_score_map[NHAN] if NHAN in anomaly_score_map else 1
    #             degree2sum += dg.degree(NHAN)
    #             NHANs += AS
    #         x_score = x / dg.degree(ms) - NHANs / degree2sum
    #         ix_score = i_score + x_score
    #         ix_all[ms] = ix_score
    #         scores.append(ix_score)
    #     for svc, score in ix_all.items():
    #         score = (score - min(scores)) / (max(scores) - min(scores) + 1e-11)
    #         ix_all[svc] = score
    #     return ix_all

    def execute_task(self, actions):
        for ms in self.mss:
            # self.k8s_util.patch_scale(ms, int(actions[ms]))
            print('scale {} to {}'.format(ms, int(actions[ms])))

    def start(self):
        print("PBScaler is running...")
        schedule.clear()
        schedule.every(AB_CHECK_INTERVAL).seconds.do(self.anomaly_detect)
        schedule.every(WASTE_CHECK_INTERVAL).seconds.do(self.waste_detection)
        time_start = time.time()
        
        while True:
            time_c = time.time() - time_start
            if time_c > self.config.duration:
                schedule.clear()
                break
            schedule.run_pending()


# if __name__ == '__main__':
#     config = Config()
#     data_path = 'output/PBScaler/'
#     simulation_model_path = '/home/ubuntu/xsy/experiment/autoscaling/simulation/train_ticket/RandomForestClassify.model'
#     scaler = PBScaler(config=config, simulation_model_path=simulation_model_path)
#     scaler.start()
#     config.end = int(round(time.time()))
#     config.start = config.end - config.duration
#     collect(config, data_path)
