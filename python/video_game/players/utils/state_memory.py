import random

class StateMemory:
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

    def has_age(self, age):
        return any(e.get_age() >= age for e in self.memory)

    def sample(self, age):
        states = [e for e in self.memory if e.get_age() >= age]
        return random.choice(states)
