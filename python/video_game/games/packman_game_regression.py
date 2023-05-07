from ..states import packman_state
from ..players import packman_player
from . import game_regression

state = packman_state.create_state()
game_regression.main(state, packman_player.create_player)
