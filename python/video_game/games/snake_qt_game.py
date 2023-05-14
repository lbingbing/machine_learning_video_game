from PySide6 import QtCore, QtGui

from ..states import snake_state
from ..players import snake_player
from . import qt_game

class SnakeGameWidget(qt_game.GameWidget):
    def init_state(self):
        self.state = snake_state.create_state()

    def create_player(self, state, player_type):
        return snake_player.create_player(state, player_type)

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
                    self.state.do_action(snake_state.UP)
                if event.key() == QtCore.Qt.Key_Down:
                    self.state.do_action(snake_state.DOWN)
                if event.key() == QtCore.Qt.Key_Left:
                    self.state.do_action(snake_state.LEFT)
                if event.key() == QtCore.Qt.Key_Right:
                    self.state.do_action(snake_state.RIGHT)

    def draw_canvas(self, painter):
        for i in range(self.state.get_canvas_shape()[0]):
            for j in range(self.state.get_canvas_shape()[1]):
                rect = ((j + 1) * self.unit_size, (i + 1) * self.unit_size, self.unit_size, self.unit_size)
                if self.state.canvas[i][j] == snake_state.BACKGROUND:
                    painter.setBrush(QtGui.QColor(255, 255, 255))
                elif self.state.canvas[i][j] == snake_state.SNAKE_HEAD:
                    painter.setBrush(QtGui.QColor(128, 0, 0))
                elif self.state.canvas[i][j] == snake_state.SNAKE_BODY:
                    painter.setBrush(QtGui.QColor(255, 0, 0))
                elif self.state.canvas[i][j] == snake_state.TARGET:
                    painter.setBrush(QtGui.QColor(0, 0, 255))
                else:
                    assert False
                painter.drawRect(*rect)

qt_game.main(SnakeGameWidget)
