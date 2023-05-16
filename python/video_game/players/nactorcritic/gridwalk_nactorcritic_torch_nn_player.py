import torch

from . import nactorcritic_torch_nn_model
from ..model import model_player

class Network(torch.nn.Module):
    def __init__(self, state_shape, action_dim):
        super().__init__()
        dim1 = state_shape[2] * state_shape[3]
        dim2 =  dim1 * action_dim
        self.sequential = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(dim1, dim2),
            torch.nn.ReLU(),
            torch.nn.Linear(dim2, dim2),
            torch.nn.ReLU(),
            )
        self.vhead = torch.nn.Sequential(
            torch.nn.Linear(dim2, 1),
            )
        self.phead = torch.nn.Sequential(
            torch.nn.Linear(dim2, action_dim),
            )

    def forward(self, x):
        x = self.sequential(x)
        return self.phead(x), self.vhead(x)

class GridWalkNActorCriticTorchNNModel(nactorcritic_torch_nn_model.NActorCriticTorchNNModel):
    def create_network(self, state):
        return Network(state.get_state_numpy_shape(), state.get_action_dim())

class GridWalkNActorCriticTorchNNPlayer(model_player.ModelPlayer):
    def create_model(self, state):
        return GridWalkNActorCriticTorchNNModel(state)
