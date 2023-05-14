from ...states import packman_state
from . import packman_sgnsarsa_torch_nn_player
from . import sgnsarsa_model_train

state = packman_state.create_state()
model = packman_sgnsarsa_torch_nn_player.PackManSGNSarsaTorchNNModel(state)

configs = {
    'check_interval': 500,
    'save_model_interval': 100000,
    'episode_num_per_iteration': 2,
    'step_num': 4,
    'dynamic_epsilon': [0.1, 0.1, 100],
    'discount': 0.99,
    'replay_memory_size': 4096,
    'batch_num_per_iteration': 2,
    'batch_size': 32,
    'dynamic_learning_rate': [0.001, 0.001, 100000],
    }

sgnsarsa_model_train.main(state, model, configs)
