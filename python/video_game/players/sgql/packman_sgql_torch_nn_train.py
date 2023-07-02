from ...states import packman_state
from . import packman_sgql_torch_nn_player
from . import sgql_model_train

state = packman_state.create_state()
model = packman_sgql_torch_nn_player.create_model(state)

configs = sgql_model_train.get_default_configs()

sgql_model_train.main(state, model, configs)
