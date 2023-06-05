from ...states import packman_state
from . import packman_nactorcritic_torch_nn_player
from . import nactorcritic_model_train

state = packman_state.create_state()
model = packman_nactorcritic_torch_nn_player.create_model(state)

configs = {
    'episode_num_per_iteration': 2,
    'step_num': 4,
    'discount': 0.99,
    'replay_memory_size': 4096,
    'batch_num_per_iteration': 2,
    'batch_size': 32,
    'dynamic_learning_rate': 0.001,
    'vloss_factor': 1,
    }

nactorcritic_model_train.main(state, model, configs)
