import time
import numpy as np
import schedule
from bayes_opt import BayesianOptimization
from util.KubernetesClient import KubernetesClient
from util.PrometheusClient import PrometheusClient
from config.Config import Config
import threading


class MicroScaler:
    def __init__(self, config: Config):
        self.prom_url = config.prom_no_range_url
        self.config_path = config.k8s_config
        self.namespace = config.namespace

        self.p_min = 0.7
        self.p_max = 0.833
        self.n_iter = 3
        self.latency_max = config.SLO
        self.pod_max = config.max_pod

        self.duration = config.duration
        self.k8s_util = KubernetesClient(config)
        self.prom_util = PrometheusClient(config)

        # 当前需要改变的服务
        self.mss = self.k8s_util.get_svcs()
        self.so = set()
        self.si = set()

    # 容器的成本
    def price(self, pod_count):
        return pod_count

    # p=P50/P90
    def p_value(self, svc):
        begin = int(round((time.time() - 30)))
        end = int(round(time.time()))
        self.prom_util.set_time_range(begin, end)
        p90 = self.prom_util.p90(svc)
        p50 = self.prom_util.p50(svc)
        if p90 == 0:
            return np.NaN
        else:
            return float(p50) / float(p90)

    # 检测机
    def detector(self):
        svc_count_dic = self.k8s_util.get_svcs_counts()
        [print(svc, svc_count_dic[svc]) for svc in svc_count_dic.keys() if svc_count_dic[svc] != 1]
        # 检测异常服务，获取异常服务列表
        svcs = self.k8s_util.get_svcs()
        ab_svcs = []
        for svc in svcs:
            begin = int(round((time.time() - 30)))
            end = int(round(time.time()))
            self.prom_util.set_time_range(begin, end)
            t = self.prom_util.p90(svc)
            # print(svc, t)
            if t > self.latency_max:
                ab_svcs.append(svc)

        self.service_power(ab_svcs)

    # 决定哪些服务需要伸缩
    def service_power(self, ab_svcs):
        for ab_svc in ab_svcs:
            p = self.p_value(ab_svc)
            if np.isnan(p):
                continue
            elif p > self.p_max:
                self.si.add(ab_svc)
            # elif p < self.p_min:
            else:
                self.so.add(ab_svc)

    # 决定服务伸缩的程度
    # Auto-scale Decision
    def auto_scale(self):
        for svc in self.so:
            # 开始扩容
            origin_pod_count = self.k8s_util.get_svc_count(svc)
            if origin_pod_count == self.pod_max:
                continue
            index = self.mss.index(svc)
            pbounds = {'x': (origin_pod_count, self.pod_max), 'index': [index, index]}
            t = threading.Thread(target=self.BO, args=(self.scale, pbounds))
            t.setDaemon(True)
            t.start()
        for svc in self.si:
            # 开始缩容
            origin_pod_count = self.k8s_util.get_svc_count(svc)
            index = self.mss.index(svc)
            if origin_pod_count == 1:
                continue
            pbounds = {'x': (1, origin_pod_count), 'index': [index, index]}
            t = threading.Thread(target=self.BO, args=(self.scale, pbounds))
            t.setDaemon(True)
            t.start()
        self.so.clear()
        self.si.clear()

    # 扩容具体操作
    def scale(self, x, index):
        # 扩容到具体数量
        svc = self.mss[int(index)]
        self.k8s_util.patch_scale(svc, int(x))
        print(svc, '扩容到', int(x))
        # 循环确认pod是否都avaliable了
        while True:
            if self.k8s_util.all_avaliable():
                break

        time.sleep(30)

        # 计算当前分数
        svcs_counts = self.k8s_util.get_svcs_counts()
        for svc in svcs_counts.keys():
            begin = int(round((time.time() - 30)))
            end = int(round(time.time()))
            self.prom_util.set_time_range(begin, end)
            P90 = self.prom_util.p90(svc)
            score = 0
            if P90 > self.latency_max:
                score = -P90 * self.price(svcs_counts[svc]) - P90 * 10
            else:
                score = -P90 * self.price(svcs_counts[svc]) + P90 * 10
            return score

    # 贝叶斯优化    
    def BO(self, f, pbounds):
        optimizer = BayesianOptimization(
            f=f,
            pbounds=pbounds,
            random_state=1,
        )
        gp_param = {'kernel': None}
        optimizer.maximize(
            **gp_param,
            n_iter=self.n_iter
        )

    # 5分钟的滑动窗口
    def auto_task(self):
        print("microscaler is running...")
        self.detector()
        schedule.clear()
        schedule.every(30).seconds.do(self.detector)
        schedule.every(2).minutes.do(self.auto_scale)
        time_start = time.time()  # 开始计时
        while True:
            time_c = time.time() - time_start
            if time_c > self.duration:
                # 超过指定运行时间，退出
                schedule.clear()
                break
            schedule.run_pending()

    def start(self):
        self.auto_task()
