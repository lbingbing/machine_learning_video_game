from ...states import snake_state
from . import snake_nactorcritic_torch_nn_player
from . import nactorcritic_model_train

state = snake_state.create_state()
model = snake_nactorcritic_torch_nn_player.create_model(state)

configs = nactorcritic_model_train.get_default_configs()

nactorcritic_model_train.main(state, model, configs)
