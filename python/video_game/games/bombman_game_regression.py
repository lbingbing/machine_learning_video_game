from ..states import bombman_state
from ..players import bombman_player
from . import game_regression

state = bombman_state.create_state()
game_regression.main(state, bombman_player.create_player)
