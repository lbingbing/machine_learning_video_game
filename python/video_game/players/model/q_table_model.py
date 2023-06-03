import numpy as np

from . import table_model
from . import q_model
from . import q_table

class QTableModel(table_model.TableModel, q_model.QModel):
    def __init__(self, game_name, state_dim, action_dim):
        table_model.TableModel.__init__(self, game_name, q_table.QTable(state_dim, action_dim))

    def train(self, batch, learning_rate):
        square_errors = []
        for state, action, target_Q in batch:
            equivalent_state_indexes = state.get_equivalent_state_indexes(state.to_state_index())
            equivalent_action_indexes = state.get_equivalent_action_indexes(state.action_to_action_index(action))
            for state_index, action_index in zip(equivalent_state_indexes, equivalent_action_indexes):
                Q = self.table.get_Q(state_index, action_index)
                loss = (Q - target_Q) ** 2
                dloss = 2 * (Q - target_Q)
                Q -= dloss * learning_rate
                self.table.set_Q(state_index, action_index, Q)
                square_errors.append(loss)
        return sum(square_errors) / len(square_errors)

    def get_Q_m(self, state):
        state_index = state.to_state_index()
        Q_m = self.table.get_Q_m(state_index)
        return Q_m

    def get_action_Q(self, state, action):
        state_index = state.to_state_index()
        action_index = state.action_index_to_action(action)
        Q = self.table.get_Q(state_index, action_index)
        return Q

    def get_legal_Q_m(self, state):
        Q_m = self.get_Q_m(state)
        legal_action_mask_m = state.get_legal_action_mask_numpy()
        legal_Q_m = np.where(legal_action_mask_m, Q_m, -np.inf)
        return legal_Q_m

    def get_max_Q(self, state):
        legal_Q_m = self.get_legal_Q_m(state)
        return np.max(legal_Q_m)

    def get_action(self, state):
        legal_Q_m = self.get_legal_Q_m(state)
        opt_action_index = np.argmax(legal_Q_m)
        action = state.action_index_to_action(opt_action_index)
        return action
