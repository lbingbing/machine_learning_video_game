from ...states import gridwalk_state
from . import gridwalk_nactorcritic_table_player
from . import nactorcritic_model_train

state = gridwalk_state.create_state()
model = gridwalk_nactorcritic_table_player.GridWalkNActorCriticTableModel(state)

configs = {
    'check_interval': 5000,
    'save_model_interval': 1000000,
    'episode_num_per_iteration': 2,
    'step_num': 4,
    'discount': 0.99,
    'replay_memory_size': 4096,
    'batch_num_per_iteration': 2,
    'batch_size': 32,
    'dynamic_learning_rate': [0.001, 0.001, 1000000],
    'vloss_factor': 1,
    }

nactorcritic_model_train.main(state, model, configs)
