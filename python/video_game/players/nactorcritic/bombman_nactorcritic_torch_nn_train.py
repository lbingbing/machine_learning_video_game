from ...states import bombman_state
from . import bombman_nactorcritic_torch_nn_player
from . import nactorcritic_model_train

state = bombman_state.create_state()
model = bombman_nactorcritic_torch_nn_player.create_model(state)

configs = nactorcritic_model_train.get_default_configs()

nactorcritic_model_train.main(state, model, configs)
