from ...states import flappybird_state
from . import flappybird_mcpgc_torch_nn_player
from . import mcpgc_model_train

state = flappybird_state.create_state()
model = flappybird_mcpgc_torch_nn_player.create_model(state)

configs = mcpgc_model_train.get_default_configs()

mcpgc_model_train.main(state, model, configs)
