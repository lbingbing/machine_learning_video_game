import argparse
import sys
import threading
import queue

from PySide6 import QtCore, QtGui, QtWidgets

from ..players import player

class GameWidget(QtWidgets.QWidget):
    def __init__(self, player_type):
        super().__init__()

        self.state = self.create_state()
        self.player = self.create_player(self.state, player_type)

        self.unit_size = self.get_unit_size()
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setFixedSize(self.unit_size * (self.state.get_canvas_shape()[1] + 2), self.unit_size * (self.state.get_canvas_shape()[0] + 2))
        self.installEventFilter(self)

        self.init_state_update_timer()
        self.init_get_action_worker()

        self.game_num = 0
        
        self.reset()

    def create_state(self):
        raise NotImplementedError()

    def create_player(self, state, player_type):
        raise NotImplementedError()

    def is_human_player(self):
        return player.is_human(self.player)

    def get_unit_size(self):
        raise NotImplementedError()

    def init_state_update_timer(self):
        self.state_update_interval = self.get_state_update_interval()

        self.state_update_timer = QtCore.QTimer(self)
        self.state_update_timer.setInterval(self.state_update_interval * 1000)
        self.state_update_timer.setSingleShot(True)
        self.state_update_timer.timeout.connect(self.update_state)

    def get_state_update_interval(self):
        raise NotImplementedError()

    def stop_state_update(self):
        if self.state_update_timer.isActive():
            self.state_update_timer.stop()

    def init_get_action_worker(self):
        self.action_polling_interval = 0.05

        self.action_requested = False

        self.get_action_request_queue = queue.Queue()
        self.get_action_response_queue = queue.Queue()
        self.get_action_worker_thread = threading.Thread(target=self.get_action_worker)
        self.get_action_worker_thread.start()

        self.poll_action_timer = QtCore.QTimer(self)
        self.poll_action_timer.setInterval(self.action_polling_interval * 1000)
        self.poll_action_timer.setSingleShot(True)
        self.poll_action_timer.timeout.connect(self.poll_action)

    def get_action_worker(self):
        while True:
            item = self.get_action_request_queue.get()
            if item is None:
                break
            func, arg = item
            action = func(arg)
            self.get_action_response_queue.put(action)

    def stop_get_action_worker(self):
        self.get_action_request_queue.put(None)
        self.get_action_worker_thread.join()

    def stop_action_request(self):
        if self.poll_action_timer.isActive():
            self.poll_action_timer.stop()
        if self.action_requested:
            self.get_action_response_queue.get()
            self.action_requested = False

    def reset(self):
        self.stop_state_update()
        self.stop_action_request()

        self.state.reset()
        self.update()

    def start(self):
        if self.is_human_player():
            self.schedule_state_update()
        else:
            self.request_action()

    def schedule_state_update(self):
        self.state_update_timer.start()

    def request_action(self):
        self.get_action_request_queue.put((self.player.get_action, self.state))
        self.action_requested = True
        self.poll_action_timer.start()

    def poll_action(self):
        try:
            item = self.get_action_response_queue.get(block=False)
            self.action_requested = False
            action = item
            self.state.do_action(action)
            self.schedule_state_update()
        except queue.Empty:
            self.poll_action_timer.start()

    def update_state(self):
        self.state.update()
        self.update()
        if self.state.is_end():
            self.game_num += 1
            print('game {} score: {} age: {}'.format(self.game_num, self.state.get_score(), self.state.get_age()))
        else:
            if self.is_human_player():
                self.schedule_state_update()
            else:
                self.request_action()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        self.draw_background(painter)
        self.draw_canvas(painter)

    def draw_background(self, painter):
        painter.setPen(QtGui.QColor(0, 0, 0))
        painter.setBrush(QtCore.Qt.NoBrush)
        canvas_shape = self.state.get_canvas_shape()
        canvas_height = (canvas_shape[0] + 2) * self.unit_size
        canvas_width = (canvas_shape[1] + 2) * self.unit_size
        for i in range(1, canvas_shape[0]+1):
            x0 = self.unit_size
            x1 = canvas_shape[1] * self.unit_size
            y = i * self.unit_size
            painter.drawLine(x0, y, x1, y)
        for j in range(1, canvas_shape[1]+1):
            x = j * self.unit_size
            y0 = self.unit_size
            y1 = canvas_shape[0] * self.unit_size
            painter.drawLine(x, y0, x, y1)

    def draw_canvas(self, painter):
        raise NotImplementedError()

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Return:
                self.reset()
                self.start()
        if self.is_human_player():
           self.handle_human_player_events(event)
        return super().eventFilter(obj, event)

    def handle_human_player_events(self, event):
        raise NotImplementedError()

    def closeEvent(self, event):
        self.stop_state_update()
        self.stop_action_request()
        self.stop_get_action_worker()

def main(Widget):
    parser = argparse.ArgumentParser()
    player.add_player_options(parser)
    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    widget = Widget(args.player_type)
    widget.show()
    sys.exit(app.exec())
