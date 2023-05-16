import numpy as np

class PVTable:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.vtable = None
        self.ptable = None

    def initialize(self):
        self.vtable = np.zeros((self.state_dim, ), dtype=np.float32)
        self.ptable = np.zeros((self.state_dim, self.action_dim), dtype=np.float32)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            np.savez_compressed(f, ptable=self.ptable, vtable=self.vtable)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            tables = np.load(f)
            self.vtable = tables['vtable']
            self.ptable = tables['ptable']
        assert self.vtable.shape == (self.state_dim, )
        assert self.ptable.shape == (self.state_dim, self.action_dim)

    def get_V(self, state_index):
        return self.vtable[state_index]

    def set_V(self, state_index, V):
        self.vtable[state_index] = V

    def get_P_logits_m(self, state_index):
        return self.ptable[state_index]

    def set_P_logits_m(self, state_index, P_logits):
        self.ptable[state_index] = P_logits
