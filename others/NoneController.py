from config.Config import Config
import time


class NoneController:
    def __init__(self, config: Config):
        self.config = config

    def start(self):
        print('NoneController is running...')
        time.sleep(self.config.duration)
