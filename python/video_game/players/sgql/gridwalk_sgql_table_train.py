from ...states import gridwalk_state
from . import gridwalk_sgql_table_player
from . import sgql_model_train

state = gridwalk_state.create_state()
model = gridwalk_sgql_table_player.create_model(state)

configs = sgql_model_train.get_default_configs()

sgql_model_train.main(state, model, configs)
