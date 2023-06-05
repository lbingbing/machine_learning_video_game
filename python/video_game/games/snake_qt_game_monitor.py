from . import snake_qt_game_utils
from . import qt_game_monitor

class SnakeGameMonitorWidget(qt_game_monitor.GameMonitorWidget):
    def create_state(self):
        return snake_qt_game_utils.create_state()

    def get_unit_size(self):
        return snake_qt_game_utils.get_unit_size()

    def get_state_update_interval(self):
        return snake_qt_game_utils.get_state_update_interval(False)

    def draw_canvas(self, painter):
        snake_qt_game_utils.draw_canvas(self.state, self.unit_size, painter)

qt_game_monitor.main(SnakeGameMonitorWidget)
