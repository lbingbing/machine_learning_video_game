import numpy as np

class VTable:
    def __init__(self, state):
        self.state_dim = state.get_state_dim()
        self.table = np.zeros((self.state_dim, ), dtype=np.float32)

    def get_V(self, state_index):
        return self.table[state_index]

    def set_V(self, state_index, V):
        self.table[state_index] = V

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            np.save(f, self.table)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.table = np.load(f)
        assert self.table.shape == (self.state_dim, )
