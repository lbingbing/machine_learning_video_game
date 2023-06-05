from PySide6 import QtCore, QtGui

from ..states import snake_state
from . import snake_qt_game_utils
from . import qt_game

class SnakeGameWidget(qt_game.GameWidget):
    def create_state(self):
        return snake_qt_game_utils.create_state()

    def create_player(self, state, player_type):
        return snake_qt_game_utils.create_player(state, player_type)

    def get_unit_size(self):
        return snake_qt_game_utils.get_unit_size()

    def get_state_update_interval(self):
        return snake_qt_game_utils.get_state_update_interval(self.is_human_player())

    def draw_canvas(self, painter):
        snake_qt_game_utils.draw_canvas(self.state, self.unit_size, painter)

    def handle_human_player_events(self, event):
        if event.type() == QtCore.QEvent.KeyPress:
            if not self.state.get_action_done():
                if event.key() == QtCore.Qt.Key_Up:
                    self.state.do_action(snake_state.UP)
                if event.key() == QtCore.Qt.Key_Down:
                    self.state.do_action(snake_state.DOWN)
                if event.key() == QtCore.Qt.Key_Left:
                    self.state.do_action(snake_state.LEFT)
                if event.key() == QtCore.Qt.Key_Right:
                    self.state.do_action(snake_state.RIGHT)

qt_game.main(SnakeGameWidget)
