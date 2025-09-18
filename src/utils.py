# src/utils.py
r"""
EN: Small utilities used in training scripts.
JA: 学習スクリプトで使う小さなユーティリティ群。
"""

from contextlib import contextmanager
from datetime import datetime, timedelta


def format_td(dt: timedelta) -> str:
    """EN: Human-friendly timedelta as H:MM:SS.
       JA: timedelta を H:MM:SS 形式の文字列に整形。
    """
    total = int(dt.total_seconds())
    h, r = divmod(total, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


class Timer:
    r"""EN: Simple wall-clock timer with checkpoints.
        JA: 壁時計ベースの簡易タイマー（チェックポイント付き）。
    """
    def __init__(self) -> None:
        self.t0 = datetime.now()
        self.last = self.t0

    def elapsed(self) -> timedelta:
        """EN: Time since start. JA: 開始からの経過時間。"""
        return datetime.now() - self.t0

    def lap(self) -> timedelta:
        """EN: Time since previous lap(). JA: 前回 lap() からの経過時間。"""
        now = datetime.now()
        dt = now - self.last
        self.last = now
        return dt

    def eta(self, progress: float) -> timedelta:
        r"""EN: Rough ETA given progress∈(0,1]; returns remaining time.
            JA: 進捗率 progress∈(0,1] からの概算残り時間（ETA）。
        """
        progress = max(1e-8, min(1.0, progress))
        spent = self.elapsed()
        remaining = spent * (1.0 - progress) / progress
        return remaining


@contextmanager
def record_time():
    """EN: Context manager returning (start_time, end_time, timedelta).
       JA: (開始, 終了, 経過) を返すコンテキストマネージャ。
    """
    t0 = datetime.now()
    try:
        yield t0
    finally:
        t1 = datetime.now()
        dt = t1 - t0
        # Could log or print here if desired
