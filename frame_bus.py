# frame_bus.py
# -*- coding: utf-8 -*-
"""Thread-safe 'latest frame' bus to share camera frames across workers."""

import threading

class _FrameBus:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def publish(self, frame):
        """Publisher stores a copy of the latest frame (BGR)."""
        with self._lock:
            self._frame = frame.copy()

    def latest(self):
        """Consumer gets a copy of the latest frame or None."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

BUS = _FrameBus()
