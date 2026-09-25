# -*- coding: utf-8 -*-
"""通用工具：日志、交易日历判断、原子保存、异常隔离。"""
from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path

from config import config as cfg  # noqa: E402  (由入口 sys.path 注入根目录)


def setup_logger(name: str = "cb_t0") -> logging.Logger:
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = cfg.LOG_DIR / f"{datetime.now():%Y%m%d_%H%M%S}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------- 交易日判断 ----------------
def expected_latest_trade_date(now: datetime | None = None) -> date:
    """
    预期最新交易日：
      周一~周五 15:00 后 -> 当天
      周一~周五 15:00 前 -> 上一个工作日
      周六/周日           -> 上周五
    （仅作为启发式窗口判断；真实以数据源返回为准）
    """
    now = now or datetime.now()
    wd = now.weekday()  # 0=Mon ... 6=Sun
    if wd >= 5:  # 周末
        back = wd - 4  # Sat->1, Sun->2
        return (now - timedelta(days=back)).date()
    # 工作日
    if now.hour >= cfg.MARKET_CLOSE_HOUR:
        return now.date()
    d = now.date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def should_download(local_latest: date | None,
                    expected: date | None = None,
                    now: datetime | None = None) -> bool:
    """
    是否需要发起下载请求。
      - 无本地数据 -> 全量下载
      - 本地已 >= 预期最新交易日 -> 跳过
      - 周末：若本地距预期 <= 周末窗口（3天），避免空检查 -> 跳过
      - 工作日：只要本地落后于预期就尝试（盘前空返回由 data_loader 不写 meta 来兜底）
    """
    now = now or datetime.now()
    expected = expected or expected_latest_trade_date(now)
    if local_latest is None:
        return True
    if local_latest >= expected:
        return False
    if now.weekday() >= 5:
        if (expected - local_latest).days <= cfg.WEEKEND_SKIP_WINDOW:
            return False
    return True


# ---------------- 元数据 ----------------
def load_meta() -> dict:
    if cfg.META_FILE.exists():
        try:
            return json.loads(cfg.META_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_meta(meta: dict) -> None:
    atomic_write_text(json.dumps(meta, ensure_ascii=False, indent=2), cfg.META_FILE)


# ---------------- 原子保存（写临时文件后替换，避免损坏） ----------------
def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os_replace(tmp, path)


def os_replace(src: Path, dst: Path) -> None:
    """跨平台原子替换。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def retry(times: int = 2, sleep: float = 1.0):
    """简单重试装饰器。"""

    def deco(fn):
        def wrapper(*a, **kw):
            last = None
            for i in range(times + 1):
                try:
                    return fn(*a, **kw)
                except Exception as e:  # noqa: BLE001
                    last = e
                    if i < times:
                        time.sleep(sleep)
            raise last

        return wrapper

    return deco
