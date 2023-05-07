from ..states import gridwalk_state
from ..players import gridwalk_player
from . import game_regression

state = gridwalk_state.create_state()
game_regression.main(state, gridwalk_player.create_player)
