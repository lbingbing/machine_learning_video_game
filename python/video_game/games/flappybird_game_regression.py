from ..states import flappybird_state
from ..players import flappybird_player
from . import game_regression

state = flappybird_state.create_state()
game_regression.main(state, flappybird_player.create_player)
