"""Performance monitoring utilities."""

import time
from contextlib import contextmanager

class PerformanceMonitor:
    """Monitors and reports performance metrics"""

    def __init__(self, metrics, log_interval=5):
        self.metrics = {m: [] for m in metrics}
        self.last_log = time.time()
        self.log_interval = log_interval

    def update(self, metric, value):
        """Update metric value"""
        if metric in self.metrics:
            self.metrics[metric].append(value)

    def get_metrics(self):
        """Get current metric averages"""
        return {m: sum(v) / len(v) if v else 0 for m, v in self.metrics.items()}

    def should_log(self):
        """Check if it's time to log metrics"""
        if time.time() - self.last_log >= self.log_interval:
            self.last_log = time.time()
            return True
        return False

    def reset(self):
        """Reset all metrics"""
        self.metrics = {m: [] for m in self.metrics}

    @contextmanager
    def measure(self, metric):
        """Context manager to measure execution time of a block"""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.update(metric, duration)
