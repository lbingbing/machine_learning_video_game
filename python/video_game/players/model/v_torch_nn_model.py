import numpy as np
import torch

from . import torch_nn_model
from . import v_model

class VTorchNNModel(torch_nn_model.TorchNNModel, v_model.VModel):
    def __init__(self, game_name, network):
        torch_nn_model.TorchNNModel.__init__(self, game_name, network)

    def train(self, batch, learning_rate):
        states = []
        target_Vs = []
        for state, target_V in batch:
            states.append(state)
            target_Vs.append(target_V)
        state_m = np.concatenate([state.get_equivalent_state_numpy(state.to_state_numpy()) for state in states], axis=0)
        target_V_m = np.concatenate([np.full((state.get_equivalent_num(), 1), target_V, dtype=np.float32) for target_V in target_Vs], axis=0)
        state_t = torch.tensor(state_m).to(self.device)
        target_V_t = torch.tensor(target_V_m).to(self.device)
        V_t = self.network(state_t)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(V_t, target_V_t)
        optimizer = torch.optim.Adam(self.network.parameters(), learning_rate, weight_decay=1e-4)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def get_V(self, state):
        with torch.no_grad():
            state_m = state.to_state_numpy()
            state_t = torch.tensor(state_m).to(self.device)
            V_t = self.network(state_t)
            return V_t.item()

