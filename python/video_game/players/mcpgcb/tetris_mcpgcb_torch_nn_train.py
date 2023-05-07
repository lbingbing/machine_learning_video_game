from ...states import tetris_state
from . import tetris_mcpgcb_torch_nn_player
from . import mcpgcb_model_train

state = tetris_state.create_state()
model = tetris_mcpgcb_torch_nn_player.TetrisMCPGCBTorchNNModel(state)

configs = {
    'check_interval': 500,
    'save_model_interval': 100000,
    'episode_num_per_iteration': 2,
    'discount': 0.99,
    'replay_memory_size': 4096,
    'batch_num_per_iteration': 2,
    'batch_size': 32,
    'learning_rate': 0.0001,
    'vloss_factor': 1,
    }

mcpgcb_model_train.main(state, model, configs)
