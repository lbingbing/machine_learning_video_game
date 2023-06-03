import torch

from . import torch_network

class Backbone(torch.nn.Module):
    def __init__(self, state_shape, action_dim):
        super().__init__()

        dim1 = state_shape[1] * state_shape[2] * state_shape[3]
        dim2 = dim1 * action_dim
        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(dim1, dim2),
            torch.nn.ReLU(),
            torch_network.FcResidualBlock(dim2),
            torch_network.FcResidualBlock(dim2),
            )

    def forward(self, x):
        return self.layers(x)

class QNetwork(torch_network.QNetwork):
    def __init__(self, state_shape, action_dim):
        super().__init__(state_shape, action_dim, Backbone(state_shape, action_dim))

class PNetwork(torch_network.PNetwork):
    def __init__(self, state_shape, action_dim):
        super().__init__(state_shape, action_dim, Backbone(state_shape, action_dim))

class PVNetwork(torch_network.PVNetwork):
    def __init__(self, state_shape, action_dim):
        super().__init__(state_shape, action_dim, Backbone(state_shape, action_dim))
