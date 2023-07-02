from ...states import gridwalk_state
from . import gridwalk_sgnsarsa_table_player
from . import sgnsarsa_model_train

state = gridwalk_state.create_state()
model = gridwalk_sgnsarsa_table_player.create_model(state)

configs = sgnsarsa_model_train.get_default_configs()

sgnsarsa_model_train.main(state, model, configs)
