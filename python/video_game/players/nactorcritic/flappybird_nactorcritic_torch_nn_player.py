from . import nactorcritic_torch_nn_model
from ..model.torch import flappybird_torch_network
from ..model import model_player

def create_model(state):
    return nactorcritic_torch_nn_model.NActorCriticTorchNNModel(state.get_name(), flappybird_torch_network.PVNetwork(state.get_state_numpy_shape(), state.get_action_dim()))

def create_player(state):
    return model_player.ModelPlayer(create_model(state))
