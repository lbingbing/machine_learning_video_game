import functools
import operator

import torch

from . import sgql_torch_nn_model
from ..model import model_player

class Network(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        sequential1 = torch.nn.Sequential(
            torch.nn.Conv2d(state.get_state_numpy_shape()[1], 16, 3, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3, padding=2),
            torch.nn.ReLU(),
            )
        dim1 = functools.reduce(operator.mul, sequential1(torch.rand(*state.get_state_numpy_shape())).shape)
        dim2 = state.get_state_numpy_shape()[2] * state.get_state_numpy_shape()[3]
        sequential2 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(dim1, dim2),
            torch.nn.ReLU(),
            torch.nn.Linear(dim2, state.get_action_dim()),
            )
        self.layers = torch.nn.Sequential(
            sequential1,
            sequential2,
            )

    def forward(self, x):
        return self.layers(x)

class PackManSGQLTorchNNModel(sgql_torch_nn_model.SGQLTorchNNModel):
    def create_network(self, state):
        return Network(state)

class PackManSGQLTorchNNPlayer(model_player.ModelPlayer):
    def create_model(self, state):
        return PackManSGQLTorchNNModel(state)
