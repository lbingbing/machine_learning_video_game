from . import mcpgc_table_model
from ..model import model_player

def create_model(state):
    return mcpgc_table_model.MCPGCTableModel(state.get_name(), state.get_state_dim(), state.get_action_dim())

def create_player(state):
    return model_player.ModelPlayer(create_model(state))
