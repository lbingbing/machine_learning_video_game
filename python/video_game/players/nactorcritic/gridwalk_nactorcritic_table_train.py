from ...states import gridwalk_state
from . import gridwalk_nactorcritic_table_player
from . import nactorcritic_model_train

state = gridwalk_state.create_state()
model = gridwalk_nactorcritic_table_player.create_model(state)

configs = nactorcritic_model_train.get_default_configs()

nactorcritic_model_train.main(state, model, configs)
