from PySide6 import QtCore, QtGui

from ..states import flappybird_state
from . import flappybird_qt_game_utils
from . import qt_game

class FlappyBirdGameWidget(qt_game.GameWidget):
    def create_state(self):
        return flappybird_qt_game_utils.create_state()

    def create_player(self, state, player_type):
        return flappybird_qt_game_utils.create_player(state, player_type)

    def get_unit_size(self):
        return flappybird_qt_game_utils.get_unit_size()

    def get_state_update_interval(self):
        return flappybird_qt_game_utils.get_state_update_interval(self.is_human_player())

    def draw_canvas(self, painter):
        flappybird_qt_game_utils.draw_canvas(self.state, self.unit_size, painter)

    def handle_human_player_events(self, event):
        if event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Space:
                if not self.state.get_action_done():
                    self.state.do_action(flappybird_state.FLY)

qt_game.main(FlappyBirdGameWidget)
