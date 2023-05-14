from ...states import snake_state
from . import snake_mcpgc_torch_nn_player
from . import mcpgc_model_train

state = snake_state.create_state()
model = snake_mcpgc_torch_nn_player.SnakeMCPGCTorchNNModel(state)

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
