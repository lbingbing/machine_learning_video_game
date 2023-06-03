import numpy as np

class PVTable:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ptable = None
        self.vtable = None

    def initialize(self):
        self.ptable = np.zeros((self.state_dim, self.action_dim), dtype=np.float32)
        self.vtable = np.zeros((self.state_dim, ), dtype=np.float32)

    def get_entry_number(self):
        return self.state_dim * self.action_dim + self.state_dim

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            np.savez_compressed(f, ptable=self.ptable, vtable=self.vtable)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            tables = np.load(f)
            self.ptable = tables['ptable']
            self.vtable = tables['vtable']
        assert self.ptable.shape == (self.state_dim, self.action_dim)
        assert self.vtable.shape == (self.state_dim, )

    def get_P_logits_m(self, state_index):
        return self.ptable[state_index]

    def set_P_logits_m(self, state_index, P_logits):
        self.ptable[state_index] = P_logits

    def get_V(self, state_index):
        return self.vtable[state_index]

    def set_V(self, state_index, V):
        self.vtable[state_index] = V
