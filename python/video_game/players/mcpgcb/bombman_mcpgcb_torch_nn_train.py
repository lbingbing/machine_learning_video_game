from ...states import bombman_state
from . import bombman_mcpgcb_torch_nn_player
from . import mcpgcb_model_train

state = bombman_state.create_state()
model = bombman_mcpgcb_torch_nn_player.create_model(state)

configs = mcpgcb_model_train.get_default_configs()

mcpgcb_model_train.main(state, model, configs)
