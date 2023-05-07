from ...states import gridwalk_state
from . import gridwalk_gmcc_torch_nn_player
from . import gmcc_model_train

state = gridwalk_state.create_state()
model = gridwalk_gmcc_torch_nn_player.GridWalkGMCCTorchNNModel(state)

configs = {
    'check_interval': 500,
    'save_model_interval': 100000,
    'episode_num_per_iteration': 2,
    'dynamic_epsilon': [0.1, 0.1, 25],
    'discount': 0.99,
    'replay_memory_size': 4096,
    'batch_num_per_iteration': 2,
    'batch_size': 32,
    'learning_rate': 0.01,
    }

gmcc_model_train.main(state, model, configs)
