from ..states import snake_state
from ..players import snake_player
from . import game_regression

state = snake_state.create_state()
game_regression.main(state, snake_player.create_player)
