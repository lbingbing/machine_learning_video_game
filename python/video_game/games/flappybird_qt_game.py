from PySide6 import QtCore, QtGui

from ..states import flappybird_state
from ..players import flappybird_player
from . import qt_game

class FlappyBirdGameWidget(qt_game.GameWidget):
    def init_state(self):
        self.state = flappybird_state.create_state()

    def create_player(self, state, player_type):
        return flappybird_player.create_player(state, player_type)

    def init_gui_parameters(self):
        self.unit_size = 15

    def init_state_update_interval(self):
        if self.is_human_player():
            self.state_update_interval = 0.1
        else:
            self.state_update_interval = 0.1

    def handle_human_player_events(self, event):
        if event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Space:
                if not self.state.get_action_done():
                    self.state.do_action(flappybird_state.FLY)

    def draw_canvas(self, painter):
        for i in range(self.state.get_canvas_shape()[0]):
            for j in range(self.state.get_canvas_shape()[1]):
                rect = ((j + 1) * self.unit_size, (i + 1) * self.unit_size, self.unit_size, self.unit_size)
                if self.state.canvas[i][j] == flappybird_state.BACKGROUND:
                    painter.setBrush(QtGui.QColor(255, 255, 255))
                elif self.state.canvas[i][j] == flappybird_state.BIRD:
                    painter.setBrush(QtGui.QColor(255, 0, 0))
                elif self.state.canvas[i][j] == flappybird_state.WALL:
                    painter.setBrush(QtGui.QColor(0, 0, 0))
                elif self.state.canvas[i][j] == flappybird_state.TARGET:
                    painter.setBrush(QtGui.QColor(0, 0, 255))
                else:
                    assert False
                painter.drawRect(*rect)

qt_game.main(FlappyBirdGameWidget)
