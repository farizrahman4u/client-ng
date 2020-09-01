import logging
import os
from subprocess import PIPE, Popen, STDOUT
import threading
import time


logger = logging.getLogger(__name__)


class TPUProfiler(object):

    def __init__(self):
        try:
            import cloud_tpu_profiler
            del cloud_tpu_profiler  # flake8
            self._enabled = True
        except ImportError:
            logger.warn("cloud_tpu_profiler is not installed. "
                        "TPU stats will not be captured.")
            self._enabled = False
            return
        self._tpu_utilization = -1.
        self._time = time.time()
        self._validity_timeout = 10  # seconds
        self.start()

    def start(self):
        self._start_capture_process()
        self._stop_thread = False
        self._thread = threading.Thread(target=self._thread_body)
        self._thread.start()

    def _start_capture_process(self):
        args = ["capture_tpu_profile",
                "--tpu=" + os.environ['TPU_NAME'],
                "--monitoring_level=2"]
        self._capture_process = Popen(args,
                                      stdout=PIPE,
                                      stderr=STDOUT,
                                      universal_newlines=True)

    def _is_capture_process_alive(self):
        return self._capture_process.poll() is None

    def _kill_capture_process(self):
        try:
            print("Killing capture process..")
            self._capture_process.kill()
            print("Killed.")
        except Exception as e:
            print("Error killing capture process: " + str(e))

    def _thread_body(self):
        while not self._stop_thread:
            if not self._is_capture_process_alive():
                self._start_capture_process()
            watchdog = _WatchdogTimer(timeout=10,
                                        callback=self._kill_capture_process,
                                        daemon=True)
            watchdog.start()
            for line in self._capture_process.stdout:
                if line.startswith("Utilization "):
                    self._tpu_utilization = float(line.split(': ')[1].split('%')[0])
                    print(self._tpu_utilization)
                    self._time = time.time()
                    with self._watchdog.blocked:
                        self._watchdog.restart()
            watchdog.cancel()
            self._kill_capture_process()
            self._start_capture_process()

    def _is_valid(self):
        return time.time() - self._time < self._validity_timeout

    def get_tpu_utilization(self):
        if self._enabled and self._is_valid():
            return self._tpu_utilization
        return 0.

    def stop(self):
        if self._enabled:
            self._stop_thread = True
            self._thread.join()
            self._kill_capture_process()

    def is_enabled(self):
        return self._enabled


class _WatchdogTimer(threading.Thread):
    """Run *callback* in *timeout* seconds unless the timer is restarted."""

    def __init__(self, timeout, callback, *args, timer=time.monotonic, **kwargs):
        super().__init__(**kwargs)
        self.timeout = timeout
        self.callback = callback
        self.args = args
        self.timer = timer
        self.cancelled = threading.Event()
        self.blocked = threading.Lock()

    def run(self):
        self.restart()
        while not self.cancelled.wait(self.deadline - self.timer()):
            with self.blocked:
                if self.deadline <= self.timer() and not self.cancelled.is_set():
                    return self.callback(*self.args)

    def restart(self):
        self.deadline = self.timer() + self.timeout

    def cancel(self):
        self.cancelled.set()
