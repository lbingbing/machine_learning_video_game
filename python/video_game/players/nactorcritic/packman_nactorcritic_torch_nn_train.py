from ...states import packman_state
from . import packman_nactorcritic_torch_nn_player
from . import nactorcritic_model_train

state = packman_state.create_state()
model = packman_nactorcritic_torch_nn_player.create_model(state)

configs = nactorcritic_model_train.get_default_configs()

nactorcritic_model_train.main(state, model, configs)
