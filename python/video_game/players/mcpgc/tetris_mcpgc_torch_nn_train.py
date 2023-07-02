from ...states import tetris_state
from . import tetris_mcpgc_torch_nn_player
from . import mcpgc_model_train

state = tetris_state.create_state()
model = tetris_mcpgc_torch_nn_player.create_model(state)

configs = mcpgc_model_train.get_default_configs()

mcpgc_model_train.main(state, model, configs)
