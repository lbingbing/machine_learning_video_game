from ...states import tetris_state
from . import tetris_sgnsarsa_torch_nn_player
from . import sgnsarsa_model_train

state = tetris_state.create_state()
model = tetris_sgnsarsa_torch_nn_player.create_model(state)

configs = sgnsarsa_model_train.get_default_configs()

sgnsarsa_model_train.main(state, model, configs)
