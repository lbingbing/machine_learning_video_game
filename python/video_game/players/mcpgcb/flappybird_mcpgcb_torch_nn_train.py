from ...states import flappybird_state
from . import flappybird_mcpgcb_torch_nn_player
from . import mcpgcb_model_train

state = flappybird_state.create_state()
model = flappybird_mcpgcb_torch_nn_player.create_model(state)

configs = {
    'check_interval': 500,
    'save_model_interval': 100000,
    'episode_num_per_iteration': 2,
    'discount': 0.99,
    'replay_memory_size': 4096,
    'batch_num_per_iteration': 2,
    'batch_size': 32,
    'dynamic_learning_rate': 0.001,
    'vloss_factor': 1,
    }

mcpgcb_model_train.main(state, model, configs)
