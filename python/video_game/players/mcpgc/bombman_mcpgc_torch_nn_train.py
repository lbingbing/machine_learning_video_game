from ...states import bombman_state
from . import bombman_mcpgc_torch_nn_player
from . import mcpgc_model_train

state = bombman_state.create_state()
model = bombman_mcpgc_torch_nn_player.create_model(state)

configs = mcpgc_model_train.get_default_configs()

mcpgc_model_train.main(state, model, configs)
