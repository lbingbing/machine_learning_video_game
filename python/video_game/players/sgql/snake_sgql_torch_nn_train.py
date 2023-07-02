from ...states import snake_state
from . import snake_sgql_torch_nn_player
from . import sgql_model_train

state = snake_state.create_state()
model = snake_sgql_torch_nn_player.create_model(state)

configs = sgql_model_train.get_default_configs()

sgql_model_train.main(state, model, configs)
