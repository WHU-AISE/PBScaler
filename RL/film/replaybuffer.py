import random
from collections import deque
MINIBATCH_SIZE = 64

class Buffer:
    def __init__(self, buffer_size):
        self.limit = buffer_size
        self.data = deque(maxlen=self.limit)
        
    def __len__(self):
        return len(self.data)
    
    def sample_batch(self, batchSize):
        if len(self.data) < batchSize:
            print('Not enough entries to sample without replacement.')
            return None
        else:
            batch = random.sample(self.data, batchSize)
            curState = [element[0] for element in batch]
            action = [element[1] for element in batch]
            nextState = [element[2] for element in batch]
            reward = [element[3] for element in batch]
        return curState, action, nextState, reward
                  
    def append(self, element):
        self.data.append(element)  
