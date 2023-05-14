from ...states import gridwalk_state
from . import gridwalk_sgql_table_player
from . import sgql_model_train

state = gridwalk_state.create_state()
model = gridwalk_sgql_table_player.GridWalkSGQLTableModel(state)

configs = {
    'check_interval': 5000,
    'save_model_interval': 1000000,
    'episode_num_per_iteration': 2,
    'dynamic_epsilon': [0.1, 0.1, 25],
    'discount': 0.99,
    'replay_memory_size': 4096,
    'batch_num_per_iteration': 2,
    'batch_size': 32,
    'dynamic_learning_rate': [0.001, 0.001, 1000000],
    }

sgql_model_train.main(state, model, configs)
