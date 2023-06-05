from PySide6 import QtCore, QtGui

from ..states import flappybird_state
from ..players import flappybird_player

def create_state():
    return flappybird_state.create_state()

def create_player(state, player_type):
    return flappybird_player.create_player(state, player_type)

def get_unit_size():
    return 15

def get_state_update_interval(is_human_player):
    return 0.2 if is_human_player else 0.1

def draw_canvas(state, unit_size, painter):
    for i in range(state.get_canvas_shape()[0]):
        for j in range(state.get_canvas_shape()[1]):
            rect = ((j + 1) * unit_size, (i + 1) * unit_size, unit_size, unit_size)
            if state.canvas[i][j] == flappybird_state.BACKGROUND:
                painter.setBrush(QtGui.QColor(255, 255, 255))
            elif state.canvas[i][j] == flappybird_state.BIRD:
                painter.setBrush(QtGui.QColor(255, 0, 0))
            elif state.canvas[i][j] == flappybird_state.WALL:
                painter.setBrush(QtGui.QColor(0, 0, 0))
            elif state.canvas[i][j] == flappybird_state.TARGET:
                painter.setBrush(QtGui.QColor(0, 0, 255))
            else:
                assert False
            painter.drawRect(*rect)
