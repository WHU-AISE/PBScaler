'''
    kubernetes HPA
'''
import os
from sched import scheduler
import time

from config.Config import Config
from util.KubernetesClient import KubernetesClient
import schedule


class KHPA:
    def __init__(self, config: Config):
        self.config = config
        self.k8s_util = KubernetesClient(config)
        self.duration = config.duration

    def start(self):
        # 需要在管理员权限下使用
        print('启动 KHPA')
        os.system('sh ./others/HPA/%s.sh'%self.config.namespace)
        time.sleep(self.duration)
