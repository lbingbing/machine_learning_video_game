from PySide6 import QtCore, QtGui

from ..states import tetris_state
from ..players import tetris_player
from . import qt_game

class TetrisGameWidget(qt_game.GameWidget):
    def init_state(self):
        self.state = tetris_state.create_state()

    def create_player(self, state, player_type):
        return tetris_player.create_player(state, player_type)

    def init_gui_parameters(self):
        self.unit_size = 20

    def init_state_update_interval(self):
        if self.is_human_player():
            self.state_update_interval = 0.3
        else:
            self.state_update_interval = 0.1

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

    def draw_canvas(self, painter):
        for i in range(self.state.get_canvas_shape()[0]):
            for j in range(self.state.get_canvas_shape()[1]):
                rect = ((j + 1) * self.unit_size, (i + 1) * self.unit_size, self.unit_size, self.unit_size)
                if self.state.canvas[i][j] == tetris_state.BACKGROUND:
                    painter.setBrush(QtGui.QColor(255, 255, 255))
                elif self.state.canvas[i][j] == tetris_state.LANDED_UNIT:
                    painter.setBrush(QtGui.QColor(0, 0, 255))
                elif self.state.canvas[i][j] == tetris_state.FALLING_UNIT:
                    painter.setBrush(QtGui.QColor(255, 0, 0))
                else:
                    assert False
                painter.drawRect(*rect)

qt_game.main(TetrisGameWidget)
