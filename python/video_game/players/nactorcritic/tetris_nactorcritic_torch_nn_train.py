from ...states import tetris_state
from . import tetris_nactorcritic_torch_nn_player
from . import nactorcritic_model_train

state = tetris_state.create_state()
model = tetris_nactorcritic_torch_nn_player.create_model(state)

configs = nactorcritic_model_train.get_default_configs()

nactorcritic_model_train.main(state, model, configs)
