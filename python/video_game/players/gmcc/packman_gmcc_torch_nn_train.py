from ...states import packman_state
from . import packman_gmcc_torch_nn_player
from . import gmcc_model_train

state = packman_state.create_state()
model = packman_gmcc_torch_nn_player.create_model(state)

configs = gmcc_model_train.get_default_configs()

gmcc_model_train.main(state, model, configs)
