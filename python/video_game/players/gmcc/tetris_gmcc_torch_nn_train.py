from ...states import tetris_state
from . import tetris_gmcc_torch_nn_player
from . import gmcc_model_train

state = tetris_state.create_state()
model = tetris_gmcc_torch_nn_player.create_model(state)

configs = gmcc_model_train.get_default_configs()

gmcc_model_train.main(state, model, configs)
