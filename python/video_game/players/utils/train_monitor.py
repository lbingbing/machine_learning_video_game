import struct
import threading
import socket
import pickle
import contextlib

class Monitor:
    def __init__(self, port):
        self.port = port
        self.running = True
        self.cur_episode = []
        self.cur_episode_end = False
        self.next_episode = []
        self.cv = threading.Condition()
        self.worker_thread = threading.Thread(target=self.worker)
        self.worker_thread.start()

    def send_state(self, state):
        with self.cv:
            if self.cur_episode_end:
                self.next_episode.append(state)
                if state.is_end():
                    self.next_episode.clear()
            else:
                self.cur_episode.append(state)
                if state.is_end():
                    self.cur_episode_end = True
                self.cv.notify()

    def worker(self):
        while True:
            with self.cv:
                if not self.running:
                    break
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', self.port))
                    s.listen(1)
                    conn, addr = s.accept()
                    print('monitor connected')
                    with conn:
                        while True:
                            with self.cv:
                                self.cv.wait_for(lambda: not self.running or self.cur_episode)
                                if not self.running:
                                    break
                                state = self.cur_episode.pop(0)
                                if state.is_end():
                                    assert not self.cur_episode
                                    self.cur_episode, self.next_episode = self.next_episode, self.cur_episode
                                    self.cur_episode_end = False
                            state_pickle_bytes = pickle.dumps(state)
                            conn.sendall(struct.pack('<Q', len(state_pickle_bytes)))
                            conn.sendall(state_pickle_bytes)
            except Exception as e:
                print('monitor disconnected')

    def close(self):
        was_running = False
        with self.cv:
            if self.running:
                was_running = True
                self.running = False
                self.cv.notify()
        if was_running:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect(('127.0.0.1', self.port))
            except Exception as e:
                pass
            self.worker_thread.join()

@contextlib.contextmanager
def create_training_monitor(monitor_port):
    if monitor_port:
        monitor = Monitor(monitor_port)
        try:
            yield monitor
        finally:
            monitor.close()
    else:
        yield None
