from ...states import flappybird_state
from . import flappybird_sgnsarsa_torch_nn_player
from . import sgnsarsa_model_train

state = flappybird_state.create_state()
model = flappybird_sgnsarsa_torch_nn_player.create_model(state)

configs = {
    'episode_num_per_iteration': 2,
    'step_num': 4,
    'dynamic_epsilon': 0.1,
    'discount': 0.99,
    'replay_memory_size': 4096,
    'batch_num_per_iteration': 2,
    'batch_size': 32,
    'dynamic_learning_rate': 0.001,
    }

sgnsarsa_model_train.main(state, model, configs)
