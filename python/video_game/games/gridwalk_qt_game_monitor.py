from . import gridwalk_qt_game_utils
from . import qt_game_monitor

class GridWalkGameMonitorWidget(qt_game_monitor.GameMonitorWidget):
    def create_state(self):
        return gridwalk_qt_game_utils.create_state()

    def get_unit_size(self):
        return gridwalk_qt_game_utils.get_unit_size()

    def get_state_update_interval(self):
        return gridwalk_qt_game_utils.get_state_update_interval(False)

    def draw_canvas(self, painter):
        gridwalk_qt_game_utils.draw_canvas(self.state, self.unit_size, painter)

qt_game_monitor.main(GridWalkGameMonitorWidget)
