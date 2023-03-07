import time
from others.KHPA import KHPA
from others.MicroScaler import MicroScaler
from others.Showar import Showar
from others.NoneController import NoneController
from others.RandomController import RandomController
from PBScaler import PBScaler
from monitor import MetricCollect
from config.Config import Config
import warnings
warnings.filterwarnings("ignore")


def initController(name: str, config: Config):
    if name == 'MicroScaler':
        return MicroScaler(config)
    elif name == 'SHOWAR':
        return Showar(config)
    elif name == 'KHPA':
        return KHPA(config)
    elif name == 'random':
        return RandomController(config)
    elif name == 'PBScaler':
        simulation_model_path = '/home/ubuntu/xsy/experiment/autoscaling/simulation/train_ticket/RandomForestClassify.model'
        return PBScaler(config, simulation_model_path)
    else:
        raise NotImplementedError() 


if __name__ == '__main__':
    config = Config()

    controller = initController('PBScaler', config)
    controller.start()

    # 收集指标
    data_path = './output'
    MetricCollect.collect(config, data_path)