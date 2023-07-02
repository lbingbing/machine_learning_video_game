from ...states import snake_state
from . import snake_mcpgc_torch_nn_player
from . import mcpgc_model_train

state = snake_state.create_state()
model = snake_mcpgc_torch_nn_player.create_model(state)

configs = mcpgc_model_train.get_default_configs()

mcpgc_model_train.main(state, model, configs)
