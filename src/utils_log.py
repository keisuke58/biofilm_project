# src/utils_log.py
from datetime import datetime

def log(*args):
    """全ログにタイムスタンプを付ける print ラッパー"""
    ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(ts, *args)
