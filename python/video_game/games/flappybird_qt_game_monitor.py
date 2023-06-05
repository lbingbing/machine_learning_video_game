from . import flappybird_qt_game_utils
from . import qt_game_monitor

class FlappyBirdGameMonitorWidget(qt_game_monitor.GameMonitorWidget):
    def create_state(self):
        return flappybird_qt_game_utils.create_state()

    def get_unit_size(self):
        return flappybird_qt_game_utils.get_unit_size()

    def get_state_update_interval(self):
        return flappybird_qt_game_utils.get_state_update_interval(False)

    def draw_canvas(self, painter):
        flappybird_qt_game_utils.draw_canvas(self.state, self.unit_size, painter)

qt_game_monitor.main(FlappyBirdGameMonitorWidget)
