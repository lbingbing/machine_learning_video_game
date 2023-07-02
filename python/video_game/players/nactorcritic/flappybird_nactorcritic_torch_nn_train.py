from ...states import flappybird_state
from . import flappybird_nactorcritic_torch_nn_player
from . import nactorcritic_model_train

state = flappybird_state.create_state()
model = flappybird_nactorcritic_torch_nn_player.create_model(state)

configs = nactorcritic_model_train.get_default_configs()

nactorcritic_model_train.main(state, model, configs)
