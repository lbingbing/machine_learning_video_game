import numpy as np

class QTable:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.table = None

    def initialize(self):
        self.table = np.zeros((self.state_dim, self.action_dim), dtype=np.float32)

    def get_entry_number(self):
        return self.state_dim * self.action_dim

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            np.save(f, self.table)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.table = np.load(f)
        assert self.table.shape == (self.state_dim, self.action_dim)

    def get_Q(self, state_index, action_index):
        return self.table[state_index][action_index]

    def get_Q_m(self, state_index):
        return self.table[state_index]

    def set_Q(self, state_index, action_index, Q):
        self.table[state_index][action_index] = Q
