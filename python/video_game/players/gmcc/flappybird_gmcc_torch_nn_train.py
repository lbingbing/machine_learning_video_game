from ...states import flappybird_state
from . import flappybird_gmcc_torch_nn_player
from . import gmcc_model_train

state = flappybird_state.create_state()
model = flappybird_gmcc_torch_nn_player.create_model(state)

configs = gmcc_model_train.get_default_configs()

gmcc_model_train.main(state, model, configs)
