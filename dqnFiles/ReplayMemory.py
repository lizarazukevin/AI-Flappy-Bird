from collections import deque
import random


class ReplayMemory:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = deque(maxlen=self.memory_size)

    def sample(self, batch_size):
        if len(self.memory) > 32:
            batch = random.sample(self.memory, batch_size)

        else: batch = random.sample(self.memory, len(self.memory))

        return batch

        # inputs = []
        # targets = []
        #
        # for experience in batch:
        #     state, action, reward, next_state = experience

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

        if self.__len__() > self.memory_size:
            self.memory.pop()

    def __len__(self):
        return len(self.memory)
