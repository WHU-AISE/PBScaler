import time

import schedule
from sklearn.preprocessing import scale

from config.Config import Config
from util.KubernetesClient import KubernetesClient
import numpy as np
import random

class RandomController:
    def __init__(self, config: Config):
        self.config = config
        self.kube_util = KubernetesClient(config)

    def scale(self):
        # 随机选择两个服务进行数量更改
        # svcs = ['ts-assurance-service', 'ts-auth-service', 'ts-basic-service', 'ts-config-service', 'ts-consign-price-service', 'ts-consign-service', 'ts-contacts-service', 'ts-execute-service', 'ts-food-map-service', 'ts-food-service', 'ts-inside-payment-service', 'ts-order-other-service', 'ts-order-service', 'ts-payment-service', 'ts-preserve-service', 'ts-price-service', 'ts-route-service', 'ts-seat-service', 'ts-security-service', 'ts-station-service', 'ts-ticketinfo-service', 'ts-train-service', 'ts-travel-service', 'ts-travel2-service', 'ts-ui-dashboard', 'ts-user-service']
        svcs = ['adservice', 'cartservice', 'checkoutservice', 'currencyservice', 'emailservice', 'frontend', 'paymentservice', 'productcatalogservice', 'recommendationservice', 'shippingservice']
        mss = np.array([ms for ms in svcs if 'redis' not in ms and 'mongo' not in ms and 'mysql' not in ms and 'rabbitmq' not in ms])
        indexes = np.random.randint(len(mss), size=2)

        for ms in mss[indexes]:
            new_num = random.randint(self.config.min_pod, self.config.max_pod)
            self.kube_util.patch_scale(ms, new_num)

    def start(self):
        print("RandomController is running...")
        schedule.clear()
        schedule.every(2).minutes.do(self.scale)
        time_start = time.time()
        while True:
            time_c = time.time() - time_start
            if time_c > self.config.duration:
                schedule.clear()
                break
            schedule.run_pending()
