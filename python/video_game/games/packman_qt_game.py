from PySide6 import QtCore, QtGui

from ..states import packman_state
from . import packman_qt_game_utils
from . import qt_game

class PackManGameWidget(qt_game.GameWidget):
    def create_state(self):
        return packman_qt_game_utils.create_state()

    def create_player(self, state, player_type):
        return packman_qt_game_utils.create_player(state, player_type)

    def get_unit_size(self):
        return packman_qt_game_utils.get_unit_size()

    def get_state_update_interval(self):
        return packman_qt_game_utils.get_state_update_interval(self.is_human_player())

    def draw_canvas(self, painter):
        packman_qt_game_utils.draw_canvas(self.state, self.unit_size, painter)

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

qt_game.main(PackManGameWidget)
