from ...states import snake_state
from . import snake_sgnsarsa_torch_nn_player
from . import sgnsarsa_model_train

state = snake_state.create_state()
model = snake_sgnsarsa_torch_nn_player.SnakeSGNSarsaTorchNNModel(state)

configs = {
    'check_interval': 500,
    'save_model_interval': 100000,
    'episode_num_per_iteration': 2,
    'step_num': 4,
    'dynamic_epsilon': [0.1, 0.1, 25],
    'discount': 0.99,
    'replay_memory_size': 4096,
    'batch_num_per_iteration': 2,
    'batch_size': 32,
    'learning_rate': 0.01,
    }

sgnsarsa_model_train.main(state, model, configs)
