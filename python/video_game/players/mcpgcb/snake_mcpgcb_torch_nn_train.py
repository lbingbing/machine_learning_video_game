from ...states import snake_state
from . import snake_mcpgcb_torch_nn_player
from . import mcpgcb_model_train

state = snake_state.create_state()
model = snake_mcpgcb_torch_nn_player.create_model(state)

configs = mcpgcb_model_train.get_default_configs()

mcpgcb_model_train.main(state, model, configs)
