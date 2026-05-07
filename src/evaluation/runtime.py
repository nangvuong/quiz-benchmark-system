import time
from contextlib import contextmanager

class RuntimeTracker:
    def __init__(self):
        self.records = {}

    @contextmanager
    def measure(self, name: str):
        start = time.time()
        yield
        end = time.time()
        if name not in self.records:
            self.records[name] = []
        self.records[name].append(end - start)

    def get_average_runtime(self, name: str) -> float:
        if name not in self.records or not self.records[name]:
            return 0.0
        return sum(self.records[name]) / len(self.records[name])
        
    def get_all_averages(self) -> dict:
        return {name: self.get_average_runtime(name) for name in self.records}
