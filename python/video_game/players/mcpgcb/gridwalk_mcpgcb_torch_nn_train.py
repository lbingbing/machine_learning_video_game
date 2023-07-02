from ...states import gridwalk_state
from . import gridwalk_mcpgcb_torch_nn_player
from . import mcpgcb_model_train

state = gridwalk_state.create_state()
model = gridwalk_mcpgcb_torch_nn_player.create_model(state)

configs = mcpgcb_model_train.get_default_configs()

mcpgcb_model_train.main(state, model, configs)
