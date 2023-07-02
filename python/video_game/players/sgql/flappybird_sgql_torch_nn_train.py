from ...states import flappybird_state
from . import flappybird_sgql_torch_nn_player
from . import sgql_model_train

state = flappybird_state.create_state()
model = flappybird_sgql_torch_nn_player.create_model(state)

configs = sgql_model_train.get_default_configs()

sgql_model_train.main(state, model, configs)
