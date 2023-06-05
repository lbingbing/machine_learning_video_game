from PySide6 import QtCore, QtGui

from ..states import bombman_state
from ..players import bombman_player

def create_state():
    return bombman_state.create_state()

def create_player(state, player_type):
    return bombman_player.create_player(state, player_type)

def get_unit_size():
    return 30

def get_state_update_interval(is_human_player):
    return 0.3 if is_human_player else 0.1

def draw_canvas(state, unit_size, painter):
    for i in range(state.get_canvas_shape()[0]):
        for j in range(state.get_canvas_shape()[1]):
            rect = ((j + 1) * unit_size, (i + 1) * unit_size, unit_size, unit_size)
            if state.canvas[i][j] == bombman_state.BACKGROUND:
                painter.setBrush(QtGui.QColor(255, 255, 255))
            elif state.canvas[i][j] == bombman_state.STEEL:
                painter.setBrush(QtGui.QColor(0, 0, 0))
            elif state.canvas[i][j] == bombman_state.WALL:
                painter.setBrush(QtGui.QColor(128, 128, 128))
            elif state.canvas[i][j] == bombman_state.GATE:
                painter.setBrush(QtGui.QColor(0, 0, 255))
            elif state.canvas[i][j] == bombman_state.BOMBMAN or (state.canvas[i][j] >= bombman_state.BOMBMAN_BOMB_BASE and state.canvas[i][j] < bombman_state.BOMBMAN_BOMB_BASE + bombman_state.BOMB_TRIGGER_TIME):
                painter.setBrush(QtGui.QColor(255, 0, 0))
            elif state.canvas[i][j] >= bombman_state.BOMB_BASE and state.canvas[i][j] < bombman_state.BOMB_BASE + bombman_state.BOMB_TRIGGER_TIME:
                painter.setBrush(QtGui.QColor(0, 255, 0))
            else:
                assert False
            painter.drawRect(*rect)
            bomb_timer = None
            if state.canvas[i][j] >= bombman_state.BOMB_BASE and state.canvas[i][j] < bombman_state.BOMB_BASE + bombman_state.BOMB_TRIGGER_TIME:
                bomb_timer = state.canvas[i][j] - bombman_state.BOMB_BASE
            if state.canvas[i][j] >= bombman_state.BOMBMAN_BOMB_BASE and state.canvas[i][j] < bombman_state.BOMBMAN_BOMB_BASE + bombman_state.BOMB_TRIGGER_TIME:
                bomb_timer = state.canvas[i][j] - bombman_state.BOMBMAN_BOMB_BASE
            if bomb_timer is not None:
                painter.setBrush(QtGui.QColor(0, 0, 0))
                painter.drawText(*rect, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter, str(bomb_timer))
