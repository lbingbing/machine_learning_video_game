import numpy as np

from . import table_model
from . import pv_model
from . import pv_table

class PVTableModel(table_model.TableModel, pv_model.PVModel):
    def __init__(self, game_name, state_dim, action_dim):
        table_model.TableModel.__init__(self, game_name, pv_table.PVTable(state_dim, action_dim))

        self.softmax_temperature = 1 / 3
        self.softmax_temperature_training = 1
        self.focal_loss_factor = 0

    def create_table(self, state):
        return pv_table.PVTable(state.get_state_dim(), state.get_action_dim())

    def train(self, batch, learning_rate, vloss_factor):
        plosses = []
        vlosses = []
        for state, target_P, p_factor, target_V in batch:
            equivalent_state_indexes = state.get_equivalent_state_indexes(state.to_state_index())
            equivalent_target_P_m = state.get_equivalent_action_numpy(np.array(target_P, dtype=np.float32).reshape(1, state.get_action_dim()))
            for i, state_index in enumerate(equivalent_state_indexes):
                P_logits_m = self.table.get_P_logits_m(state_index)
                P_m = self.softmax(P_logits_m)
                ploss = -p_factor * np.sum(np.power(np.abs(P_m - equivalent_target_P_m[i]), self.focal_loss_factor) * equivalent_target_P_m[i] * np.log(P_m + 1e-6))
                dploss = p_factor * (P_m - equivalent_target_P_m[i]) * np.power(np.abs(P_m - equivalent_target_P_m[i]), self.focal_loss_factor)
                P_logits_m = P_logits_m - dploss * learning_rate
                self.table.set_P_logits_m(state_index, P_logits_m)
                V = self.table.get_V(state_index)
                vloss = (V - target_V) ** 2 * vloss_factor
                dvloss = 2 * (V - target_V) * vloss_factor
                V -= dvloss * learning_rate
                self.table.set_V(state_index, V)
                plosses.append(ploss)
                vlosses.append(vloss)
        return sum(plosses) / len(plosses), sum(vlosses) / len(vlosses)

    def get_V(self, state):
        state_index = state.to_state_index()
        V = self.table.get_V(state_index)
        return V
        
    def get_legal_P_logits_m(self, state):
        state_index = state.to_state_index()
        P_logits_m = self.table.get_P_logits_m(state_index)
        legal_P_logits_m = P_logits_m[state.get_legal_action_indexes()]
        return legal_P_logits_m

    def get_legal_P_logit_range(self, state):
        legal_P_logits_m = self.get_legal_P_logits_m(state)
        return np.min(legal_P_logits_m), np.max(legal_P_logits_m)

    def softmax(self, P_logits_m):
        P_m = P_logits_m.copy()
        P_m /= self.softmax_temperature_training if self.is_training else self.softmax_temperature
        P_m -= np.max(P_m)
        P_m = np.exp(P_m)
        P_m = P_m / np.sum(P_m)
        return P_m

    def get_legal_P_m(self, state):
        legal_P_logits_m = self.get_legal_P_logits_m(state)
        legal_P_m = self.softmax(legal_P_logits_m)
        return legal_P_m

    def get_P_m(self, state):
        legal_P_m = self.get_legal_P_m(state)
        P_m = np.zeros(state.get_action_dim(), dtype=np.float32)
        P_m[state.get_legal_action_indexes()] = legal_P_m
        return P_m

    def get_P(self, state):
        return self.get_P_m(state).tolist()

    def get_legal_P_range(self, state):
        legal_P_m = self.get_legal_P_m(state)
        return np.min(legal_P_m), np.max(legal_P_m)

    def get_action(self, state):
        P_m = self.get_P_m(state)
        action_index = np.random.choice(state.get_action_dim(), p=P_m)
        action = state.action_index_to_action(action_index)
        return action
