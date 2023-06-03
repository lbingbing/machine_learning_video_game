import random

class ReplayMemory:
    def __init__(self, size):
        self.size = size
        self.memory = []
        self.wr_ptr = 0

    def resize(self, size):
        if size != self.size:
            self.size = size
            del self.memory[self.size:]
            if self.wr_ptr >= self.size:
                self.wr_ptr = 0

    def record(self, data):
        if len(self.memory) == self.size:
            self.memory[self.wr_ptr] = data
            self.wr_ptr = (self.wr_ptr + 1) % self.size
        else:
            self.memory.append(data)

    def sample(self, batch_size):
        if len(self.memory) > batch_size:
            return random.sample(self.memory, batch_size)
        else:
            return self.memory
