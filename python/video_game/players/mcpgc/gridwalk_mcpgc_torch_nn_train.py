from ...states import gridwalk_state
from . import gridwalk_mcpgc_torch_nn_player
from . import mcpgc_model_train

state = gridwalk_state.create_state()
model = gridwalk_mcpgc_torch_nn_player.GridWalkMCPGCTorchNNModel(state)

configs = {
    'check_interval': 500,
    'save_model_interval': 100000,
    'episode_num_per_iteration': 2,
    'discount': 0.99,
    'replay_memory_size': 4096,
    'batch_num_per_iteration': 2,
    'batch_size': 32,
    'dynamic_learning_rate': [0.001, 0.001, 100000],
    }

mcpgc_model_train.main(state, model, configs)
