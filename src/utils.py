# src/utils.py
# =============================================================================
# EN: Small utilities for training, logging, and progress display.
# JA: 学習・ログ記録・進捗表示のための小さなユーティリティ群。
# =============================================================================

import os
import sys
import csv
import io
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional, Dict, Any


# -----------------------------------------------------------------------------
# Time formatting / 時間整形
# -----------------------------------------------------------------------------
def format_td(dt: timedelta, show_seconds: bool = True) -> str:
    """
    EN: Format timedelta as H:MM:SS (or H:MM if show_seconds=False).
    JA: timedelta を H:MM:SS （または H:MM）形式で文字列化。

    Args:
        dt: timedelta
        show_seconds: whether to include seconds
    Returns:
        formatted string
    """
    total = int(dt.total_seconds())
    h, r = divmod(total, 3600)
    m, s = divmod(r, 60)
    if show_seconds:
        return f"{h:d}:{m:02d}:{s:02d}"
    else:
        return f"{h:d}:{m:02d}"


def now_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    EN: Current timestamp as string.
    JA: 現在時刻を文字列で返す。
    """
    return datetime.now().strftime(fmt)


# -----------------------------------------------------------------------------
# Timer / タイマー
# -----------------------------------------------------------------------------
class Timer:
    """
    EN: Simple wall-clock timer with checkpoints and ETA estimation.
    JA: 壁時計ベースの簡易タイマー（チェックポイントとETA推定付き）。
    """

    def __init__(self) -> None:
        self.t0 = datetime.now()
        self.last = self.t0

    def reset(self) -> None:
        """EN: Reset start and lap times. JA: 開始・ラップをリセット。"""
        self.t0 = datetime.now()
        self.last = self.t0

    def elapsed(self) -> timedelta:
        """EN: Time since start. JA: 開始からの経過時間。"""
        return datetime.now() - self.t0

    def lap(self) -> timedelta:
        """EN: Time since last lap(). JA: 前回 lap() からの経過時間。"""
        now = datetime.now()
        dt = now - self.last
        self.last = now
        return dt

    def eta(self, progress: float) -> timedelta:
        """
        EN: Estimate remaining time given progress ∈ (0,1].
        JA: 進捗率 progress ∈ (0,1] に基づき、残り時間を概算。

        Args:
            progress: fraction completed
        Returns:
            estimated remaining time
        """
        progress = max(1e-8, min(1.0, progress))
        spent = self.elapsed()
        remaining = spent * (1.0 - progress) / progress
        return remaining


# -----------------------------------------------------------------------------
# Context manager / コンテキストマネージャ
# -----------------------------------------------------------------------------
@contextmanager
def record_time():
    """
    EN: Context manager measuring block execution time.
        Usage:
            with record_time() as rt:
                # work...
            print(rt['start'], rt['end'], rt['delta'])
    JA: ブロック実行時間を測定するコンテキスト。
        使い方:
            with record_time() as rt:
                # 処理...
            print(rt['start'], rt['end'], rt['delta'])
    """
    t0 = datetime.now()
    payload = {"start": t0, "end": None, "delta": None}
    try:
        yield payload
    finally:
        t1 = datetime.now()
        payload["end"] = t1
        payload["delta"] = t1 - t0


# -----------------------------------------------------------------------------
# Filesystem helpers / ファイルシステム補助
# -----------------------------------------------------------------------------
def ensure_dir(path: str) -> str:
    """
    EN: Create directory if it doesn't exist. Returns the path.
    JA: ディレクトリが無ければ作成して、そのパスを返す。
    """
    os.makedirs(path, exist_ok=True)
    return path


# -----------------------------------------------------------------------------
# Simple logger / テキストロガー
# -----------------------------------------------------------------------------
class SimpleLogger:
    """
    EN: Minimal file logger with optional stdout mirroring.
    JA: 最小限のファイルロガー。標準出力への同時出力も可能。

    Example:
        logger = SimpleLogger("runs/train.log", mirror_stdout=True)
        logger.info("hello")
        logger.dict({"lr": 1e-3, "loss": 0.12})
    """

    def __init__(self, filepath: str, mirror_stdout: bool = True, append: bool = True) -> None:
        ensure_dir(os.path.dirname(filepath) or ".")
        mode = "a" if append else "w"
        self.fp = open(filepath, mode, encoding="utf-8")
        self.mirror = mirror_stdout
        self.filepath = filepath

    def close(self) -> None:
        if not self.fp.closed:
            self.fp.close()

    def _write(self, msg: str) -> None:
        ts = now_str()
        line = f"[{ts}] {msg}\n"
        self.fp.write(line)
        self.fp.flush()
        if self.mirror:
            sys.stdout.write(line)
            sys.stdout.flush()

    def info(self, msg: str) -> None:
        """EN/JA: Write a plain info line."""
        self._write(msg)

    def dict(self, kv: Dict[str, Any], prefix: Optional[str] = None) -> None:
        """
        EN: Log key-value pairs as a single line.
        JA: キー値ペアを1行で出力。
        """
        pre = f"{prefix} " if prefix else ""
        body = " ".join(f"{k}={v}" for k, v in kv.items())
        self._write(pre + body)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# CSV logger / CSVロガー
# -----------------------------------------------------------------------------
class CSVLogger:
    """
    EN: Append dict rows to CSV (creates header automatically).
    JA: dict をCSVに追記（ヘッダ自動生成）。

    Example:
        csvlog = CSVLogger("runs/metrics.csv")
        csvlog.write({"epoch": 1, "loss": 0.3})
    """

    def __init__(self, filepath: str) -> None:
        ensure_dir(os.path.dirname(filepath) or ".")
        self.filepath = filepath
        self._has_header = os.path.exists(filepath) and os.path.getsize(filepath) > 0

    def write(self, row: Dict[str, Any]) -> None:
        fieldnames = list(row.keys())
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._has_header:
                writer.writeheader()
                self._has_header = True
            writer.writerow(row)


# -----------------------------------------------------------------------------
# Text progress bar / テキスト進捗バー
# -----------------------------------------------------------------------------
class ProgressBar:
    """
    EN: Minimal text progress bar for loops.
    JA: ループ用の簡易テキスト進捗バー。

    Example:
        pbar = ProgressBar(total=900, width=30, label="train")
        for ep in range(1, 901):
            # ... work ...
            pbar.update(ep)
        pbar.close()
    """

    def __init__(self, total: int, width: int = 40, label: str = "", stream: io.TextIOBase = sys.stdout):
        self.total = max(1, int(total))
        self.width = max(10, int(width))
        self.label = label
        self.stream = stream
        self.start_time = datetime.now()
        self.last_len = 0
        self._render(0)

    def _render(self, n: int) -> None:
        n = max(0, min(n, self.total))
        frac = n / self.total
        filled = int(self.width * frac + 0.5)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = datetime.now() - self.start_time
        # Simple ETA
        eta = format_td(elapsed * (1 - frac) / max(1e-8, frac)) if n > 0 else "?:??:??"
        msg = f"{self.label} [{bar}] {n}/{self.total} ({frac*100:5.1f}%)  ETA {eta}"
        # erase previous line (carriage return), then print
        self.stream.write("\r" + msg + " " * max(0, self.last_len - len(msg)))
        self.stream.flush()
        self.last_len = len(msg)
        if n == self.total:
            self.stream.write("\n")
            self.stream.flush()

    def update(self, n: int) -> None:
        """EN/JA: Update current count (1..total)."""
        self._render(n)

    def close(self) -> None:
        """EN/JA: Finish the bar if not ended."""
        self._render(self.total)


# -----------------------------------------------------------------------------
# Pretty printing dictionaries / 辞書の整形出力
# -----------------------------------------------------------------------------
def kv_str(d: Dict[str, Any], sep: str = " ", kv_join: str = "=") -> str:
    """
    EN: Convert dict to a single-line string "k=v k2=v2 ..."
    JA: 辞書を1行の "k=v k2=v2 ..." 形式に整形。
    """
    return sep.join(f"{k}{kv_join}{v}" for k, v in d.items())
