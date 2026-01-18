import time
from contextlib import ContextDecorator


class Timer(ContextDecorator):
    """Context manager / decorator to measure elapsed time.

    Usage:
        with Timer() as t:
            do_work()
        print(t.elapsed_seconds)

    Or as decorator:
        @Timer()
        def f():
            pass
    """

    def __init__(self, start_now=True):
        self.start = None
        self.end = None
        self.elapsed = None
        if start_now:
            self.start_now()

    def start_now(self):
        self.start = time.time()
        self.end = None
        self.elapsed = None

    def stop(self):
        if self.start is None:
            return None
        self.end = time.time()
        self.elapsed = self.end - self.start
        return self.elapsed

    @property
    def elapsed_seconds(self):
        if self.elapsed is None and self.start is not None:
            return time.time() - self.start
        return self.elapsed

    def __enter__(self):
        if self.start is None:
            self.start_now()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False


def measure_seconds(func, *args, **kwargs):
    t = Timer()
    t.start_now()
    result = func(*args, **kwargs)
    t.stop()
    return t.elapsed, result
