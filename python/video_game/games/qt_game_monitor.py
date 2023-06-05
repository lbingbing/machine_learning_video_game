import argparse
import sys
import threading
import time
import socket
import struct
import pickle

from PySide6 import QtCore, QtGui, QtWidgets

class GameMonitorWidget(QtWidgets.QWidget):
    def __init__(self, port):
        super().__init__()

        self.port = port
        self.state = self.create_state()

        self.unit_size = self.get_unit_size()
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setFixedSize(self.unit_size * (self.state.get_canvas_shape()[1] + 2), self.unit_size * (self.state.get_canvas_shape()[0] + 2))

        self.init_state_update_timer()
        self.init_get_state_worker()

        self.game_num = 0

        self.start()

    def create_state(self):
        raise NotImplementedError()

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

    def init_get_state_worker(self):
        self.state_polling_interval = 0.05

        self.get_state_worker_running = True
        self.get_state_worker_cur_episode = []
        self.get_state_worker_cur_episode_end = False
        self.get_state_worker_next_episode = []
        self.get_state_worker_lock = threading.Lock()
        self.get_state_worker_thread = threading.Thread(target=self.get_state_worker)
        self.get_state_worker_thread.start()

        self.poll_state_timer = QtCore.QTimer(self)
        self.poll_state_timer.setInterval(self.state_polling_interval * 1000)
        self.poll_state_timer.setSingleShot(True)
        self.poll_state_timer.timeout.connect(self.poll_state)

    def get_state_worker(self):
        while True:
            with self.get_state_worker_lock:
                if not self.get_state_worker_running:
                    break
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect(('127.0.0.1', self.port))
                    print('monitor connected')
                    while True:
                        with self.get_state_worker_lock:
                            if not self.get_state_worker_running:
                                break
                        len_bytes = bytearray()
                        while len(len_bytes) < 8:
                            b = s.recv(8 - len(len_bytes))
                            if not b:
                                raise Exception('recv fails')
                            len_bytes += b
                        state_pickle_byte_len = struct.unpack('<Q', len_bytes)[0]
                        state_pickle_bytes = bytearray()
                        while len(state_pickle_bytes) < state_pickle_byte_len:
                            b = s.recv(state_pickle_byte_len - len(state_pickle_bytes))
                            if not b:
                                raise Exception('recv fails')
                            state_pickle_bytes += b
                        state = pickle.loads(state_pickle_bytes)
                        with self.get_state_worker_lock:
                            if self.get_state_worker_cur_episode_end:
                                self.get_state_worker_next_episode.append(state)
                                if state.is_end():
                                    self.get_state_worker_next_episode.clear()
                            else:
                                self.get_state_worker_cur_episode.append(state)
                                if state.is_end():
                                    self.get_state_worker_cur_episode_end = True
            except Exception as e:
                print('monitor disconnected')
            with self.get_state_worker_lock:
                self.get_state_worker_cur_episode = []
                self.get_state_worker_cur_episode_end = False
                self.get_state_worker_next_episode = []
            time.sleep(1)

    def stop_get_state_worker(self):
        with self.get_state_worker_lock:
            self.get_state_worker_running = False
        self.get_state_worker_thread.join()

    def stop_state_request(self):
        if self.poll_state_timer.isActive():
            self.poll_state_timer.stop()

    def start(self):
        self.update()
        self.request_state()

    def schedule_state_update(self):
        self.state_update_timer.start()

    def request_state(self):
        self.poll_state_timer.start()

    def poll_state(self):
        self.get_state_worker_lock.acquire()
        if self.get_state_worker_cur_episode:
            self.state = self.get_state_worker_cur_episode.pop(0)
            if self.state.is_end():
                assert not self.get_state_worker_cur_episode
                self.get_state_worker_cur_episode, self.get_state_worker_next_episode = self.get_state_worker_next_episode, self.get_state_worker_cur_episode
                self.get_state_worker_cur_episode_end = False
            self.get_state_worker_lock.release()
            self.schedule_state_update()
        else:
            self.get_state_worker_lock.release()
            self.poll_state_timer.start()

    def update_state(self):
        self.update()
        if self.state.is_end():
            self.game_num += 1
            print('game {} score: {} age: {}'.format(self.game_num, self.state.get_score(), self.state.get_age()))
        self.request_state()

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

    def closeEvent(self, event):
        self.stop_state_update()
        self.stop_state_request()
        self.stop_get_state_worker()

def main(Widget):
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int, help='port')
    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    widget = Widget(args.port)
    widget.show()
    sys.exit(app.exec())
