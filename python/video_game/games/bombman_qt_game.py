from PySide6 import QtCore, QtGui

from ..states import bombman_state
from ..players import bombman_player
from . import qt_game

class BombManGameWidget(qt_game.GameWidget):
    def init_state(self):
        self.state = bombman_state.create_state()

    def create_player(self, state, player_type):
        return bombman_player.create_player(state, player_type)

    def init_gui_parameters(self):
        self.unit_size = 30

    def init_state_update_interval(self):
        if self.is_human_player():
            self.state_update_interval = 0.3
        else:
            self.state_update_interval = 0.1

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

    def draw_canvas(self, painter):
        for i in range(self.state.get_canvas_shape()[0]):
            for j in range(self.state.get_canvas_shape()[1]):
                rect = ((j + 1) * self.unit_size, (i + 1) * self.unit_size, self.unit_size, self.unit_size)
                if self.state.canvas[i][j] == bombman_state.BACKGROUND:
                    painter.setBrush(QtGui.QColor(255, 255, 255))
                elif self.state.canvas[i][j] == bombman_state.STEEL:
                    painter.setBrush(QtGui.QColor(0, 0, 0))
                elif self.state.canvas[i][j] == bombman_state.WALL:
                    painter.setBrush(QtGui.QColor(128, 128, 128))
                elif self.state.canvas[i][j] == bombman_state.GATE:
                    painter.setBrush(QtGui.QColor(0, 0, 255))
                elif self.state.canvas[i][j] == bombman_state.BOMBMAN or (self.state.canvas[i][j] >= bombman_state.BOMBMAN_BOMB_BASE and self.state.canvas[i][j] < bombman_state.BOMBMAN_BOMB_BASE + bombman_state.BOMB_TRIGGER_TIME):
                    painter.setBrush(QtGui.QColor(255, 0, 0))
                elif self.state.canvas[i][j] >= bombman_state.BOMB_BASE and self.state.canvas[i][j] < bombman_state.BOMB_BASE + bombman_state.BOMB_TRIGGER_TIME:
                    painter.setBrush(QtGui.QColor(0, 255, 0))
                else:
                    assert False
                painter.drawRect(*rect)
                bomb_timer = None
                if self.state.canvas[i][j] >= bombman_state.BOMB_BASE and self.state.canvas[i][j] < bombman_state.BOMB_BASE + bombman_state.BOMB_TRIGGER_TIME:
                    bomb_timer = self.state.canvas[i][j] - bombman_state.BOMB_BASE
                if self.state.canvas[i][j] >= bombman_state.BOMBMAN_BOMB_BASE and self.state.canvas[i][j] < bombman_state.BOMBMAN_BOMB_BASE + bombman_state.BOMB_TRIGGER_TIME:
                    bomb_timer = self.state.canvas[i][j] - bombman_state.BOMBMAN_BOMB_BASE
                if bomb_timer is not None:
                    painter.setBrush(QtGui.QColor(0, 0, 0))
                    painter.drawText(*rect, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter, str(bomb_timer))

qt_game.main(BombManGameWidget)
