from ...states import gridwalk_state
from . import gridwalk_sgql_torch_nn_player
from . import sgql_model_train

state = gridwalk_state.create_state()
model = gridwalk_sgql_torch_nn_player.create_model(state)

configs = {
    'episode_num_per_iteration': 2,
    'dynamic_epsilon': 0.1,
    'discount': 0.99,
    'replay_memory_size': 1024,
    'batch_num_per_iteration': 2,
    'batch_size': 32,
    'dynamic_learning_rate': 0.001,
    }

sgql_model_train.main(state, model, configs)
