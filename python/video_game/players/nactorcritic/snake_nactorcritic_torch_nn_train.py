from ...states import snake_state
from . import snake_nactorcritic_torch_nn_player
from . import nactorcritic_model_train

state = snake_state.create_state()
model = snake_nactorcritic_torch_nn_player.SnakeNActorCriticTorchNNModel(state)

configs = {
    'check_interval': 500,
    'save_model_interval': 100000,
    'episode_num_per_iteration': 2,
    'step_num': 4,
    'discount': 0.99,
    'replay_memory_size': 4096,
    'batch_num_per_iteration': 2,
    'batch_size': 32,
    'learning_rate': 0.0001,
    'vloss_factor': 1,
    }

nactorcritic_model_train.main(state, model, configs)
