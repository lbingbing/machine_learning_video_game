import numpy as np
import torch

from . import torch_nn_model
from . import pv_model

class PVTorchNNModel(torch_nn_model.TorchNNModel, pv_model.PVModel):
    def __init__(self, game_name, network):
        torch_nn_model.TorchNNModel.__init__(self, game_name, network)

        self.softmax_temperature = 1 / 3
        self.softmax_temperature_training = 1
        self.focal_loss_factor = 0

    def train(self, batch, learning_rate, vloss_factor):
        states = []
        target_Ps = []
        target_Vs = []
        p_factors = []
        for state, target_P, p_factor, target_V in batch:
            states.append(state)
            target_Ps.append(target_P)
            target_Vs.append(target_V)
            p_factors.append(p_factor)
        state_m = np.concatenate([state.get_equivalent_state_numpy(state.to_state_numpy()) for state in states], axis=0)
        target_P_m = np.concatenate([state.get_equivalent_action_numpy(np.array(target_P, dtype=np.float32).reshape(1, state.get_action_dim())) for target_P, state in zip(target_Ps, states)], axis=0)
        target_V_m = np.concatenate([np.full((state.get_equivalent_num(), 1), target_V, dtype=np.float32) for target_V in target_Vs], axis=0)
        p_factor_m = np.concatenate([np.full((state.get_equivalent_num(), 1), p_factor, dtype=np.float32) for p_factor in p_factors], axis=0)
        state_t = torch.tensor(state_m).to(self.device)
        target_P_t = torch.tensor(target_P_m).to(self.device)
        target_V_t = torch.tensor(target_V_m).to(self.device)
        p_factor_t = torch.tensor(p_factor_m).to(self.device)
        P_logits_t, V_t = self.network(state_t)
        P_t = torch.nn.Softmax(dim=1)(P_logits_t / self.softmax_temperature_training)
        ploss = -torch.mean(p_factor_t * torch.sum(torch.pow(torch.abs(P_t - target_P_t), self.focal_loss_factor) * target_P_t * torch.log(P_t + 1e-6), dim=1))
        vloss = torch.nn.MSELoss()(V_t, target_V_t) * vloss_factor
        loss = ploss + vloss
        optimizer = torch.optim.Adam(self.network.parameters(), learning_rate, weight_decay=1e-4)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return ploss.item(), vloss.item()

    def get_V(self, state):
        with torch.no_grad():
            state_m = state.to_state_numpy()
            state_t = torch.tensor(state_m).to(self.device)
            P_logits_t, V_t = self.network(state_t)
            return V_t.item()

    def get_legal_P_logits_t(self, state):
        with torch.no_grad():
            state_m = state.to_state_numpy()
            state_t = torch.tensor(state_m).to(self.device)
            P_logits_t, V_t = self.network(state_t)
            P_logits_t = P_logits_t.reshape(-1)
            legal_P_logits_t = P_logits_t[state.get_legal_action_indexes()]
            return legal_P_logits_t

    def get_legal_P_logit_range(self, state):
        with torch.no_grad():
            legal_P_logits_t = self.get_legal_P_logits_t(state)
            return torch.min(legal_P_logits_t).item(), torch.max(legal_P_logits_t).item()

    def get_legal_P_t(self, state):
        with torch.no_grad():
            legal_P_logits_t = self.get_legal_P_logits_t(state)
            legal_P_t = torch.nn.Softmax(dim=0)(legal_P_logits_t / (self.softmax_temperature_training if self.is_training else self.softmax_temperature))
            return legal_P_t

    def get_P_t(self, state):
        with torch.no_grad():
            legal_P_t = self.get_legal_P_t(state)
            P_t = torch.zeros(state.get_action_dim(), dtype=torch.float32).to(self.device)
            P_t[state.get_legal_action_indexes()] = legal_P_t
            return P_t

    def get_P(self, state):
        with torch.no_grad():
            return self.get_P_t(state).cpu().detach().numpy().tolist()

    def get_legal_P_range(self, state):
        with torch.no_grad():
            legal_P_t = self.get_legal_P_t(state)
            return torch.min(legal_P_t).item(), torch.max(legal_P_t).item()

    def get_action(self, state):
        with torch.no_grad():
            P_m = self.get_P_t(state).cpu().detach().numpy()
            action_index = np.random.choice(state.get_action_dim(), p=P_m)
            action = state.action_index_to_action(action_index)
            return action
