from PySide6 import QtCore, QtGui

from ..states import tetris_state
from ..players import tetris_player
from . import qt_game

def create_state():
    return tetris_state.create_state()

def create_player(state, player_type):
    return tetris_player.create_player(state, player_type)

def get_unit_size():
    return 20

def get_state_update_interval(is_human_player):
    return 0.3 if is_human_player else 0.1

def draw_canvas(state, unit_size, painter):
    for i in range(state.get_canvas_shape()[0]):
        for j in range(state.get_canvas_shape()[1]):
            rect = ((j + 1) * unit_size, (i + 1) * unit_size, unit_size, unit_size)
            if state.canvas[i][j] == tetris_state.BACKGROUND:
                painter.setBrush(QtGui.QColor(255, 255, 255))
            elif state.canvas[i][j] == tetris_state.LANDED_UNIT:
                painter.setBrush(QtGui.QColor(0, 0, 255))
            elif state.canvas[i][j] == tetris_state.FALLING_UNIT:
                painter.setBrush(QtGui.QColor(255, 0, 0))
            else:
                assert False
            painter.drawRect(*rect)
