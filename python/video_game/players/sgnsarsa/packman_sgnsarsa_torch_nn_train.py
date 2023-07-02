from ...states import packman_state
from . import packman_sgnsarsa_torch_nn_player
from . import sgnsarsa_model_train

state = packman_state.create_state()
model = packman_sgnsarsa_torch_nn_player.create_model(state)

configs = sgnsarsa_model_train.get_default_configs()

sgnsarsa_model_train.main(state, model, configs)
