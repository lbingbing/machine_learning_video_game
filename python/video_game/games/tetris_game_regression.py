from ..states import tetris_state
from ..players import tetris_player
from . import game_regression

state = tetris_state.create_state()
game_regression.main(state, tetris_player.create_player)
