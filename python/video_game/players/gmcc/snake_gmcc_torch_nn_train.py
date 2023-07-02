from ...states import snake_state
from . import snake_gmcc_torch_nn_player
from . import gmcc_model_train

state = snake_state.create_state()
model = snake_gmcc_torch_nn_player.create_model(state)

configs = gmcc_model_train.get_default_configs()

gmcc_model_train.main(state, model, configs)
