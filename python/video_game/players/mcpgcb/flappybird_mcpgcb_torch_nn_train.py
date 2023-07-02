from ...states import flappybird_state
from . import flappybird_mcpgcb_torch_nn_player
from . import mcpgcb_model_train

state = flappybird_state.create_state()
model = flappybird_mcpgcb_torch_nn_player.create_model(state)

configs = mcpgcb_model_train.get_default_configs()

mcpgcb_model_train.main(state, model, configs)
