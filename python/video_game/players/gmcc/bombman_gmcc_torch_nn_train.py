from ...states import bombman_state
from . import bombman_gmcc_torch_nn_player
from . import gmcc_model_train

state = bombman_state.create_state()
model = bombman_gmcc_torch_nn_player.create_model(state)

configs = gmcc_model_train.get_default_configs()

gmcc_model_train.main(state, model, configs)
