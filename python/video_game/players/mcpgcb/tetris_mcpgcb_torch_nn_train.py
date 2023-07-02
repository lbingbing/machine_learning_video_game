from ...states import tetris_state
from . import tetris_mcpgcb_torch_nn_player
from . import mcpgcb_model_train

state = tetris_state.create_state()
model = tetris_mcpgcb_torch_nn_player.create_model(state)

configs = mcpgcb_model_train.get_default_configs()

mcpgcb_model_train.main(state, model, configs)
