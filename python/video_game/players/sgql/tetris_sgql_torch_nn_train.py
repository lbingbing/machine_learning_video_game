from ...states import tetris_state
from . import tetris_sgql_torch_nn_player
from . import sgql_model_train

state = tetris_state.create_state()
model = tetris_sgql_torch_nn_player.create_model(state)

configs = sgql_model_train.get_default_configs()

sgql_model_train.main(state, model, configs)
