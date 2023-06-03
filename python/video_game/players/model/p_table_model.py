import numpy as np

from . import table_model
from . import p_model
from . import p_table

class PTableModel(table_model.TableModel, p_model.PModel):
    def __init__(self, game_name, state_dim, action_dim):
        table_model.TableModel.__init__(self, game_name, p_table.PTable(state_dim, action_dim))

        self.softmax_temperature = 1 / 3
        self.softmax_temperature_training = 1
        self.focal_loss_factor = 0

    def train(self, batch, learning_rate):
        cross_entropy_errors = []
        for state, target_P, p_factor in batch:
            equivalent_state_indexes = state.get_equivalent_state_indexes(state.to_state_index())
            equivalent_target_P_m = state.get_equivalent_action_numpy(np.array(target_P, dtype=np.float32).reshape(1, state.get_action_dim()))
            for i, state_index in enumerate(equivalent_state_indexes):
                P_logits_m = self.table.get_P_logits_m(state_index)
                P_m = self.softmax(P_logits_m)
                loss = -p_factor * np.sum(np.power(np.abs(P_m - equivalent_target_P_m[i]), self.focal_loss_factor) * equivalent_target_P_m[i] * np.log(P_m + 1e-6))
                dloss = p_factor * (P_m - equivalent_target_P_m[i]) * np.power(np.abs(P_m - equivalent_target_P_m[i]), self.focal_loss_factor)
                P_logits_m = P_logits_m - dloss * learning_rate
                self.table.set_P_logits_m(state_index, P_logits_m)
                cross_entropy_errors.append(loss)
        return sum(cross_entropy_errors) / len(cross_entropy_errors)
        
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
