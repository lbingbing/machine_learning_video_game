from ...states import bombman_state
from . import bombman_sgnsarsa_torch_nn_player
from . import sgnsarsa_model_train

state = bombman_state.create_state()
model = bombman_sgnsarsa_torch_nn_player.create_model(state)

configs = sgnsarsa_model_train.get_default_configs()

sgnsarsa_model_train.main(state, model, configs)
