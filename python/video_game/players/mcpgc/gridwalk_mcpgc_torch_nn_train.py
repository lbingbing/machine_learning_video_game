from ...states import gridwalk_state
from . import gridwalk_mcpgc_torch_nn_player
from . import mcpgc_model_train

state = gridwalk_state.create_state()
model = gridwalk_mcpgc_torch_nn_player.create_model(state)

configs = mcpgc_model_train.get_default_configs()

mcpgc_model_train.main(state, model, configs)
