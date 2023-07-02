from ...states import snake_state
from . import snake_sgnsarsa_torch_nn_player
from . import sgnsarsa_model_train

state = snake_state.create_state()
model = snake_sgnsarsa_torch_nn_player.create_model(state)

configs = sgnsarsa_model_train.get_default_configs()

sgnsarsa_model_train.main(state, model, configs)
