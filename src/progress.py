# src/progress.py
import sys
import time

class ProgressTracker:
    """簡易プログレスバー（TSM, TMCMC 共通）"""
    def __init__(self, total, desc="Progress", bar_length=40):
        self.total = total
        self.desc = desc
        self.bar_length = bar_length
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n=1):
        self.current = min(self.current + n, self.total)
        self._display()
    
    def set(self, n):
        self.current = min(n, self.total)
        self._display()
    
    def _display(self):
        percent = self.current / self.total if self.total > 0 else 1.0
        filled = int(self.bar_length * percent)
        bar = '█' * filled + '░' * (self.bar_length - filled)
        elapsed = time.time() - self.start_time
        eta = (elapsed / self.current * (self.total - self.current)) if self.current > 0 else 0.0
        sys.stdout.write(
            f'\r    {self.desc}: |{bar}| {self.current}/{self.total} '
            f'[{elapsed:.0f}s<{eta:.0f}s]'
        )
        sys.stdout.flush()
    
    def close(self):
        print()
