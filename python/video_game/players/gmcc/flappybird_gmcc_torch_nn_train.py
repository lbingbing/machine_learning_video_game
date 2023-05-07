from ...states import flappybird_state
from . import flappybird_gmcc_torch_nn_player
from . import gmcc_model_train

state = flappybird_state.create_state()
model = flappybird_gmcc_torch_nn_player.FlappyBirdGMCCTorchNNModel(state)

configs = {
    'check_interval': 500,
    'save_model_interval': 100000,
    'episode_num_per_iteration': 2,
    'dynamic_epsilon': [0.1, 0.1, 100],
    'discount': 0.99,
    'replay_memory_size': 4096,
    'batch_num_per_iteration': 2,
    'batch_size': 32,
    'learning_rate': 0.01,
    }

gmcc_model_train.main(state, model, configs)
