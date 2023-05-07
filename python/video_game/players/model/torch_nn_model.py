import torch

from . import nn_model

class TorchNNModel(nn_model.NNModel):
    def __init__(self, state):
        super().__init__(state)

        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.network = self.create_network(state).to(self.device)

    def get_nn_framework(self):
        return 'torch'

    def create_network(self, state):
        raise NotImplementedError()

    def save(self):
        torch.save(self.network.state_dict(), self.get_model_path())

    def load(self):
        self.network.load_state_dict(torch.load(self.get_model_path()))
