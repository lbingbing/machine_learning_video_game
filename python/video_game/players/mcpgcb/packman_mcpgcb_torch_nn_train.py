from ...states import packman_state
from . import packman_mcpgcb_torch_nn_player
from . import mcpgcb_model_train

state = packman_state.create_state()
model = packman_mcpgcb_torch_nn_player.create_model(state)

configs = mcpgcb_model_train.get_default_configs()

mcpgcb_model_train.main(state, model, configs)
