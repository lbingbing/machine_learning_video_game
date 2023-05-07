import functools
import operator

import torch

from . import sgnsarsa_torch_nn_model
from ..model import model_player

class Network(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        dim1 = state.get_state_numpy_shape()[2] * state.get_state_numpy_shape()[3]
        dim2 =  dim1 * state.get_action_dim()
        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(dim1, dim2),
            torch.nn.ReLU(),
            torch.nn.Linear(dim2, dim2),
            torch.nn.ReLU(),
            torch.nn.Linear(dim2, state.get_action_dim()),
            )

    def forward(self, x):
        return self.layers(x)

class GridWalkSGNSarsaTorchNNModel(sgnsarsa_torch_nn_model.SGNSarsaTorchNNModel):
    def create_network(self, state):
        return Network(state)

class GridWalkSGNSarsaTorchNNPlayer(model_player.ModelPlayer):
    def create_model(self, state):
        return GridWalkSGNSarsaTorchNNModel(state)
