from ...states import packman_state
from . import packman_mcpgc_torch_nn_player
from . import mcpgc_model_train

state = packman_state.create_state()
model = packman_mcpgc_torch_nn_player.create_model(state)

configs = mcpgc_model_train.get_default_configs()

mcpgc_model_train.main(state, model, configs)
