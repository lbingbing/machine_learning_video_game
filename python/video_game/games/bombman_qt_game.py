from PySide6 import QtCore, QtGui

from ..states import bombman_state
from . import bombman_qt_game_utils
from . import qt_game

class BombManGameWidget(qt_game.GameWidget):
    def create_state(self):
        return bombman_qt_game_utils.create_state()

    def create_player(self, state, player_type):
        return bombman_qt_game_utils.create_player(state, player_type)

    def get_unit_size(self):
        return bombman_qt_game_utils.get_unit_size()

    def get_state_update_interval(self):
        return bombman_qt_game_utils.get_state_update_interval(self.is_human_player())

    def draw_canvas(self, painter):
        bombman_qt_game_utils.draw_canvas(self.state, self.unit_size, painter)

    def handle_human_player_events(self, event):
        if event.type() == QtCore.QEvent.KeyPress:
            if not self.state.get_action_done():
                if event.key() == QtCore.Qt.Key_Up:
                    if bombman_state.UP in self.state.get_legal_actions():
                        self.state.do_action(bombman_state.UP)
                if event.key() == QtCore.Qt.Key_Down:
                    if bombman_state.DOWN in self.state.get_legal_actions():
                        self.state.do_action(bombman_state.DOWN)
                if event.key() == QtCore.Qt.Key_Left:
                    if bombman_state.LEFT in self.state.get_legal_actions():
                        self.state.do_action(bombman_state.LEFT)
                if event.key() == QtCore.Qt.Key_Right:
                    if bombman_state.RIGHT in self.state.get_legal_actions():
                        self.state.do_action(bombman_state.RIGHT)
                if event.key() == QtCore.Qt.Key_Space:
                    if bombman_state.PLANT_BOMB in self.state.get_legal_actions():
                        self.state.do_action(bombman_state.PLANT_BOMB)

qt_game.main(BombManGameWidget)
