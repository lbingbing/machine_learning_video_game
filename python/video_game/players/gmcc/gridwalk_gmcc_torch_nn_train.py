from ...states import gridwalk_state
from . import gridwalk_gmcc_torch_nn_player
from . import gmcc_model_train

state = gridwalk_state.create_state()
model = gridwalk_gmcc_torch_nn_player.create_model(state)

configs = gmcc_model_train.get_default_configs()

gmcc_model_train.main(state, model, configs)
