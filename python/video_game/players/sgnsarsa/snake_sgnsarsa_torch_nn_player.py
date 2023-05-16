import functools
import operator

import torch

from . import sgnsarsa_torch_nn_model
from ..model import model_player

class Network(torch.nn.Module):
    def __init__(self, state_shape, action_dim):
        super().__init__()
        sequential1 = torch.nn.Sequential(
            torch.nn.Conv2d(state_shape[1], 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1),
            torch.nn.ReLU(),
            )
        dim1 = functools.reduce(operator.mul, sequential1(torch.rand(*state_shape)).shape)
        dim2 = state_shape[2] * state_shape[3]
        sequential2 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(dim1, dim2),
            torch.nn.ReLU(),
            torch.nn.Linear(dim2, action_dim),
            )
        self.layers = torch.nn.Sequential(
            sequential1,
            sequential2,
            )

    def forward(self, x):
        return self.layers(x)

class SnakeSGNSarsaTorchNNModel(sgnsarsa_torch_nn_model.SGNSarsaTorchNNModel):
    def create_network(self, state):
        return Network(state.get_state_numpy_shape(), state.get_action_dim())

class SnakeSGNSarsaTorchNNPlayer(model_player.ModelPlayer):
    def create_model(self, state):
        return SnakeSGNSarsaTorchNNModel(state)
