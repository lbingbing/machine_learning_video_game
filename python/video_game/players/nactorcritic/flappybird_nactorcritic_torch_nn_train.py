from ...states import flappybird_state
from . import flappybird_nactorcritic_torch_nn_player
from . import nactorcritic_model_train

state = flappybird_state.create_state()
model = flappybird_nactorcritic_torch_nn_player.FlappyBirdNActorCriticTorchNNModel(state)

configs = {
    'check_interval': 500,
    'save_model_interval': 100000,
    'episode_num_per_iteration': 2,
    'step_num': 4,
    'discount': 0.99,
    'replay_memory_size': 4096,
    'batch_num_per_iteration': 2,
    'batch_size': 32,
    'dynamic_learning_rate': [0.001, 0.001, 100000],
    'vloss_factor': 1,
    }

nactorcritic_model_train.main(state, model, configs)
