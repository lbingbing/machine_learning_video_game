from PySide6 import QtCore, QtGui

from ..states import gridwalk_state
from ..players import gridwalk_player
from . import qt_game

class GridWalkGameWidget(qt_game.GameWidget):
    def init_state(self):
        self.state = gridwalk_state.create_state()

    def create_player(self, state, player_type):
        return gridwalk_player.create_player(state, player_type)

    def init_gui_parameters(self):
        self.unit_size = 30

    def init_state_update_interval(self):
        self.state_update_interval = 0.3

    def handle_human_player_events(self, event):
        if event.type() == QtCore.QEvent.KeyPress:
            if not self.state.get_action_done():
                if event.key() == QtCore.Qt.Key_Up:
                    self.state.do_action(gridwalk_state.UP)
                if event.key() == QtCore.Qt.Key_Down:
                    self.state.do_action(gridwalk_state.DOWN)
                if event.key() == QtCore.Qt.Key_Left:
                    self.state.do_action(gridwalk_state.LEFT)
                if event.key() == QtCore.Qt.Key_Right:
                    self.state.do_action(gridwalk_state.RIGHT)

    def draw_canvas(self, painter):
        for i in range(self.state.get_canvas_shape()[0]):
            for j in range(self.state.get_canvas_shape()[1]):
                rect = ((j + 1) * self.unit_size, (i + 1) * self.unit_size, self.unit_size, self.unit_size)
                if self.state.canvas[i][j] == gridwalk_state.BACKGROUND:
                    painter.setBrush(QtGui.QColor(255, 255, 255))
                elif self.state.canvas[i][j] == gridwalk_state.TARGET:
                    painter.setBrush(QtGui.QColor(0, 0, 255))
                elif self.state.canvas[i][j] == gridwalk_state.WALKER:
                    painter.setBrush(QtGui.QColor(255, 0, 0))
                else:
                    assert False
                painter.drawRect(*rect)

qt_game.main(GridWalkGameWidget)
