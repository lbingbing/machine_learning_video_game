from . import packman_qt_game_utils
from . import qt_game_monitor

class PackManGameMonitorWidget(qt_game_monitor.GameMonitorWidget):
    def create_state(self):
        return packman_qt_game_utils.create_state()

    def get_unit_size(self):
        return packman_qt_game_utils.get_unit_size()

    def get_state_update_interval(self):
        return packman_qt_game_utils.get_state_update_interval(False)

    def draw_canvas(self, painter):
        packman_qt_game_utils.draw_canvas(self.state, self.unit_size, painter)

qt_game_monitor.main(PackManGameMonitorWidget)
