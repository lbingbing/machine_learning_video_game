import numpy as np

from . import table_model
from . import pv_model
from . import pv_table

class PVTableModel(table_model.TableModel, pv_model.PVModel):
    def create_table(self, state):
        return pv_table.PVTable(state)

    def get_softmax_temperature(self):
        raise NotImplementedError()

    def train(self, batch, learning_rate, vloss_factor):
        vlosses = []
        plosses = []
        for state, action, target_V, factor in batch:
            equivalent_state_indexes = state.get_equivalent_state_indexes(state.to_state_index())
            equivalent_action_m = state.get_equivalent_action_numpy(state.action_to_action_numpy(action)).astype(dtype=np.float32)
            for i, state_index in enumerate(equivalent_state_indexes):
                V = self.table.get_V(state_index)
                vloss = (V - target_V) ** 2 * vloss_factor
                dvloss = 2 * (V - target_V) * vloss_factor
                V -= dvloss * learning_rate
                self.table.set_V(state_index, V)
                P_logits_m = self.table.get_P_logits_m(state_index)
                P_m = self.softmax(P_logits_m)
                ploss = -factor * np.sum(equivalent_action_m[i] * np.log(P_m + 1e-6) + (1 - equivalent_action_m[i]) * np.log(1 - P_m + 1e-6))
                dploss = factor * (P_m - equivalent_action_m[i]) * np.abs(P_m - equivalent_action_m[i])
                P_logits_m = P_logits_m - dploss * learning_rate
                self.table.set_P_logits_m(state_index, P_logits_m)
                vlosses.append(vloss)
                plosses.append(ploss)
        return sum(vlosses) / len(vlosses), sum(plosses) / len(plosses)

    def get_V(self, state):
        state_index = state.to_state_index()
        V = self.table.get_V(state_index)
        return V
        
    def get_P_logits_m(self, state):
        state_index = state.to_state_index()
        P_logits_m = self.table.get_P_logits_m(state_index)
        return P_logits_m

    def get_P_logit_range(self, state):
        P_logits_m = self.get_P_logits_m(state)
        return np.min(P_logits_m), np.max(P_logits_m)

    def softmax(self, P_logits_m):
        P_m = P_logits_m.copy()
        P_m /= self.get_softmax_temperature()
        P_m -= np.max(P_m)
        P_m = np.exp(P_m)
        P_m = P_m / np.sum(P_m)
        return P_m

    def get_P_m(self, state):
        P_logits_m = self.get_P_logits_m(state)
        P_m = self.softmax(P_logits_m)
        return P_m

    def get_legal_P_m(self, state):
        P_m = self.get_P_m(state)
        legal_action_mask_m = state.get_legal_action_mask_numpy().reshape(-1)
        legal_P_m = np.where(legal_action_mask_m, P_m, 0)
        legal_P_m /= np.sum(legal_P_m)
        return legal_P_m

    def get_P_range(self, state):
        legal_P_m = self.get_legal_P_m(state)
        return np.min(legal_P_m), np.max(legal_P_m)

    def get_action(self, state):
        legal_P_m = self.get_legal_P_m(state)
        action_index = np.random.choice(state.get_action_dim(), p=legal_P_m)
        action = state.action_index_to_action(action_index)
        return action