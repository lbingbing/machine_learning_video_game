import torch

from . import nn_model

class TorchNNModel(nn_model.NNModel):
    def __init__(self, game_name, network):
        super().__init__(game_name)

        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.network = network.to(self.device)

    def set_device(self, device):
        super().set_device(device)
        self.network.to(device)

    def set_training(self, b):
        super().set_training(b)
        self.network.train(b)

    def get_nn_framework(self):
        return 'torch'

    def initialize(self):
        def init_parameters(m):
            if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

        self.network.apply(init_parameters)

    def get_parameter_number(self):
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)

    def save(self):
        torch.save(self.network.state_dict(), self.get_model_path())

    def load(self):
        self.network.load_state_dict(torch.load(self.get_model_path()))
