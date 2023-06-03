from . import mcpgc_torch_nn_model
from ..model.torch import tetris_torch_network
from ..model import model_player

def create_model(state):
    return mcpgc_torch_nn_model.MCPGCTorchNNModel(state.get_name(), tetris_torch_network.PNetwork(state.get_state_numpy_shape(), state.get_action_dim()))

def create_player(state):
    return model_player.ModelPlayer(create_model(state))
