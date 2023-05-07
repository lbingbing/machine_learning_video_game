import numpy as np

class PTable:
    def __init__(self, state):
        self.state_dim = state.get_state_dim()
        self.action_dim = state.get_action_dim()
        self.table = np.ones((self.state_dim, self.action_dim), dtype=np.float32) / self.action_dim

    def get_P_logits_m(self, state_index):
        return self.table[state_index]

    def set_P_logits_m(self, state_index, P_logits):
        self.table[state_index] = P_logits

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            np.save(f, self.table)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.table = np.load(f)
        assert self.table.shape == (self.state_dim, self.action_dim)
