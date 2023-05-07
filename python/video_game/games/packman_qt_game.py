from PySide6 import QtCore, QtGui

from ..states import packman_state
from ..players import packman_player
from . import qt_game

class PackManGameWidget(qt_game.GameWidget):
    def init_state(self):
        self.state = packman_state.create_state()

    def create_player(self, state, player_type):
        return packman_player.create_player(state, player_type)

    def init_gui_parameters(self):
        self.unit_size = 30

    def init_state_update_interval(self):
        self.state_update_interval = 0.3

    def handle_human_player_events(self, event):
        if event.type() == QtCore.QEvent.KeyPress:
            if not self.state.get_action_done():
                if event.key() == QtCore.Qt.Key_Up:
                    if packman_state.UP in self.state.get_legal_actions():
                        self.state.do_action(packman_state.UP)
                if event.key() == QtCore.Qt.Key_Down:
                    if packman_state.DOWN in self.state.get_legal_actions():
                        self.state.do_action(packman_state.DOWN)
                if event.key() == QtCore.Qt.Key_Left:
                    if packman_state.LEFT in self.state.get_legal_actions():
                        self.state.do_action(packman_state.LEFT)
                if event.key() == QtCore.Qt.Key_Right:
                    if packman_state.RIGHT in self.state.get_legal_actions():
                        self.state.do_action(packman_state.RIGHT)

    def draw_canvas(self, painter):
        for i in range(self.state.get_canvas_shape()[0]):
            for j in range(self.state.get_canvas_shape()[1]):
                rect = ((j + 1) * self.unit_size, (i + 1) * self.unit_size, self.unit_size, self.unit_size)
                if self.state.canvas[i][j] == packman_state.BACKGROUND:
                    painter.setBrush(QtGui.QColor(255, 255, 255))
                elif self.state.canvas[i][j] == packman_state.WALL:
                    painter.setBrush(QtGui.QColor(0, 0, 0))
                elif self.state.canvas[i][j] == packman_state.BEAN:
                    painter.setBrush(QtGui.QColor(255, 255, 0))
                elif self.state.canvas[i][j] == packman_state.ENEMY:
                    painter.setBrush(QtGui.QColor(0, 255, 0))
                elif self.state.canvas[i][j] == packman_state.PACKMAN:
                    painter.setBrush(QtGui.QColor(255, 0, 0))
                else:
                    assert False
                painter.drawRect(*rect)

qt_game.main(PackManGameWidget)
