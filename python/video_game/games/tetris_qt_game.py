from PySide6 import QtCore, QtGui

from ..states import tetris_state
from . import tetris_qt_game_utils
from . import qt_game

class TetrisGameWidget(qt_game.GameWidget):
    def create_state(self):
        return tetris_qt_game_utils.create_state()

    def create_player(self, state, player_type):
        return tetris_qt_game_utils.create_player(state, player_type)

    def get_unit_size(self):
        return tetris_qt_game_utils.get_unit_size()

    def get_state_update_interval(self):
        return tetris_qt_game_utils.get_state_update_interval(self.is_human_player())

    def draw_canvas(self, painter):
        tetris_qt_game_utils.draw_canvas(self.state, self.unit_size, painter)

    def handle_human_player_events(self, event):
        if event.type() == QtCore.QEvent.KeyPress:
            if not self.state.get_action_done():
                if event.key() == QtCore.Qt.Key_Left:
                    if tetris_state.LEFT in self.state.get_legal_actions():
                        self.state.do_action(tetris_state.LEFT)
                if event.key() == QtCore.Qt.Key_Right:
                    if tetris_state.RIGHT in self.state.get_legal_actions():
                        self.state.do_action(tetris_state.RIGHT)
                if event.key() == QtCore.Qt.Key_Up:
                    if tetris_state.ROTATE in self.state.get_legal_actions():
                        self.state.do_action(tetris_state.ROTATE)
                if event.key() == QtCore.Qt.Key_Down:
                    if tetris_state.FALL in self.state.get_legal_actions():
                        self.state.do_action(tetris_state.FALL)
                if event.key() == QtCore.Qt.Key_A:
                    if tetris_state.FIRE in self.state.get_legal_actions():
                        self.state.do_action(tetris_state.FIRE)
                if event.key() == QtCore.Qt.Key_Space:
                    if tetris_state.LAND in self.state.get_legal_actions():
                        self.state.do_action(tetris_state.LAND)

qt_game.main(TetrisGameWidget)
