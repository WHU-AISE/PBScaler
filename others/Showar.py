from cmath import nan
import time
import numpy as np
import schedule
from util.KubernetesClient import KubernetesClient
from util.PrometheusClient import PrometheusClient
from config.Config import Config
from simple_pid import PID


class Showar:
    def __init__(self, config: Config):
        self.config = config
        self.k8s_util = KubernetesClient(config)
        self.prom_util = PrometheusClient(config)
        self.SLO_target = config.SLO
        self.max_pod = config.max_pod
        self.min_pod = config.min_pod

        self.mss = self.k8s_util.get_svcs_without_state()
        # 为每一个微服务初始化controller
        self.controller_map = {}
        for ms in self.mss:
            self.controller_map[ms] = PID(1 / 3, 1 / 3, 1 / 3, setpoint=self.SLO_target)

        self.beta = 0.1
        self.alpha = 0.2

    def PID_score(self, ms):
        # 获取 runq_latency
        p90 = self.prom_util.p90(ms)
        if np.isnan(p90):
            return self.SLO_target
        # 计算PID score
        pid = self.controller_map[ms]
        output = pid(p90)
        return -1 * output

    def horizontal_scale(self):
        # 获取所有服务的PID分数
        ms_score_map = {}
        for ms in self.mss:
            ms_score_map[ms] = self.PID_score(ms)
        print(ms_score_map)
        # 从高到低排序
        ranks = sorted(ms_score_map.items(), key=lambda x: x[1], reverse=True)
        # 实时获取服务之间的调用关系
        DG = self.prom_util.get_call()
        for pair in ranks:
            ms = pair[0]
            output = pair[1]
            # 等依赖的服务全部扩容完
            son_mss = list(DG.successors(ms))
            while True:
                if self.k8s_util.svcs_avaliable(son_mss):
                    break
            RM = self.k8s_util.get_svc_count(ms)
            if output > self.SLO_target * (1 + self.alpha / 2):
                RM = int(RM + max(1, RM * self.beta))
            elif output < self.SLO_target * (1 - self.alpha / 2):
                RM = int(RM - max(1, RM * self.beta))
            else:
                continue
            if self.min_pod <= RM <= self.max_pod:
                print(ms, '扩容到', RM)
                self.k8s_util.patch_scale(ms, RM)

    def start(self):
        print('Showar is running')
        self.horizontal_scale()
        schedule.every(15).seconds.do(self.horizontal_scale)
        time_start = time.time()  # 开始计时
        while True:
            time_c = time.time() - time_start
            if time_c > self.config.duration:
                # 超过指定运行时间，退出
                schedule.clear()
                break
            schedule.run_pending()
