from ...states import bombman_state
from . import bombman_sgql_torch_nn_player
from . import sgql_model_train

state = bombman_state.create_state()
model = bombman_sgql_torch_nn_player.create_model(state)

configs = sgql_model_train.get_default_configs()

sgql_model_train.main(state, model, configs)
