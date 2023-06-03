import numpy as np
import torch

from . import torch_nn_model
from . import q_model

class QTorchNNModel(torch_nn_model.TorchNNModel, q_model.QModel):
    def __init__(self, game_name, network):
        torch_nn_model.TorchNNModel.__init__(self, game_name, network)

    def train(self, batch, learning_rate):
        states = []
        actions = []
        target_Qs = []
        for state, action, target_Q in batch:
            states.append(state)
            actions.append(action)
            target_Qs.append(target_Q)
        state_m = np.concatenate([state.get_equivalent_state_numpy(state.to_state_numpy()) for state in states], axis=0)
        action_m = np.concatenate([state.get_equivalent_action_numpy(state.action_to_action_numpy(action)) for action in actions], axis=0)
        target_Q_m = np.concatenate([np.full((state.get_equivalent_num(), 1), target_Q, dtype=np.float32) for target_Q in target_Qs], axis=0)
        state_t = torch.tensor(state_m).to(self.device)
        action_t = torch.tensor(action_m).to(self.device)
        target_Q_t = torch.tensor(target_Q_m).to(self.device)
        Q_t = self.network(state_t)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(torch.where(action_t, Q_t, 0), torch.mul(target_Q_t, action_t))
        optimizer = torch.optim.Adam(self.network.parameters(), learning_rate, weight_decay=1e-4)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def get_Q_t(self, state):
        with torch.no_grad():
            state_m = state.to_state_numpy()
            state_t = torch.tensor(state_m).to(self.device)
            Q_t = self.network(state_t)
            return Q_t

    def get_action_Q(self, state, action):
        with torch.no_grad():
            Q_t = self.get_Q_t(state)
            action_m = state.action_to_action_numpy(action)
            action_t = torch.tensor(action_m).to(self.device)
            return torch.max(torch.where(action_t, Q_t, -np.inf)).item()

    def get_legal_Q_t(self, state):
        with torch.no_grad():
            Q_t = self.get_Q_t(state)
            legal_action_mask_m = state.get_legal_action_mask_numpy()
            legal_action_mask_t = torch.tensor(legal_action_mask_m).to(self.device)
            legal_Q_t = torch.where(legal_action_mask_t, Q_t, -np.inf)
            return legal_Q_t

    def get_max_Q(self, state):
        with torch.no_grad():
            legal_Q_t = self.get_legal_Q_t(state)
            return torch.max(legal_Q_t).item()

    def get_action(self, state):
        with torch.no_grad():
            legal_Q_t = self.get_legal_Q_t(state)
            opt_action_index = torch.argmax(legal_Q_t).item()
            action = state.action_index_to_action(opt_action_index)
            return action
