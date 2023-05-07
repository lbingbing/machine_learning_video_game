from ...states import packman_state
from . import packman_mcpgc_torch_nn_player
from . import mcpgc_model_train

state = packman_state.create_state()
model = packman_mcpgc_torch_nn_player.PackManMCPGCTorchNNModel(state)

configs = {
    'check_interval': 500,
    'save_model_interval': 100000,
    'episode_num_per_iteration': 2,
    'discount': 0.99,
    'replay_memory_size': 4096,
    'batch_num_per_iteration': 2,
    'batch_size': 32,
    'learning_rate': 0.0001,
   }

mcpgc_model_train.main(state, model, configs)
