# -*- coding: utf-8 -*-
"""特征工程：转债量价 + 正股联动 + 转股溢价率 + 流动性。"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 进模型的特征列
FEATURE_COLS = [
    # 原有量价
    "ret1", "ret5", "ret10", "ret20",
    "vol5", "vol10", "vol20",
    "amp5", "amp20",
    "pos20", "pos60",
    "ma5_bias", "ma20_bias", "ma60_bias",
    "vma5_ratio", "vma20_ratio",
    "high20_ratio", "low20_ratio",
    "skew5", "kur5",
    # 新增：正股联动
    "stock_ret1", "stock_ret5", "stock_ret10", "stock_ret20",
    "div1", "div5",
    # 新增：溢价率
    "premium", "premium_chg5",
    # 新增：流动性
    "amount_ratio",
]

# 仅用于回测过滤、不进模型的列
FILTER_COLS = ["premium", "avg_amount_20"]


def _features_one(g: pd.DataFrame, code: str) -> pd.DataFrame:
    g = g.sort_values("trade_date").copy()
    g["code"] = code
    c = g["close"]
    o = g["open"]
    h = g["high"]
    l = g["low"]
    v = g["volume"]

    g["ret1"] = c.pct_change(1)
    g["ret5"] = c.pct_change(5)
    g["ret10"] = c.pct_change(10)
    g["ret20"] = c.pct_change(20)

    r = g["ret1"]
    g["vol5"] = r.rolling(5).std()
    g["vol10"] = r.rolling(10).std()
    g["vol20"] = r.rolling(20).std()

    amp = (h - l) / c
    g["amp5"] = amp.rolling(5).mean()
    g["amp20"] = amp.rolling(20).mean()

    low20 = l.rolling(20).min();
    high20 = h.rolling(20).max()
    low60 = l.rolling(60).min();
    high60 = h.rolling(60).max()
    g["pos20"] = (c - low20) / (high20 - low20 + 1e-9)
    g["pos60"] = (c - low60) / (high60 - low60 + 1e-9)

    g["ma5_bias"] = c / c.rolling(5).mean() - 1
    g["ma20_bias"] = c / c.rolling(20).mean() - 1
    g["ma60_bias"] = c / c.rolling(60).mean() - 1

    vma5 = v.rolling(5).mean();
    vma20 = v.rolling(20).mean()
    g["vma5_ratio"] = v / (vma5 + 1e-9)
    g["vma20_ratio"] = v / (vma20 + 1e-9)

    g["high20_ratio"] = c / high20 - 1
    g["low20_ratio"] = c / low20 - 1

    g["skew5"] = r.rolling(5).skew()
    g["kur5"] = r.rolling(5).kurt()

    # 转债成交额近似（volume 单位张，每张面值100，价格即全价）
    g["amount"] = c * v
    g["avg_amount_20"] = g["amount"].rolling(20).mean()
    g["amount_ratio"] = g["amount"] / (g["amount"].rolling(20).mean() + 1e-9)

    # T+1 日 OHLC（供回测引擎模拟盘中止盈止损）
    g["next_open"] = o.shift(-1)
    g["next_high"] = h.shift(-1)
    g["next_low"] = l.shift(-1)
    # 标签：T日收盘后出信号，T+1开盘买、T+1收盘卖（T+0当日回转）
    g["label"] = c.shift(-1) / o.shift(-1) - 1
    g["next_ret"] = c.shift(-1) / o.shift(-1) - 1
    return g


def build_features(panel: pd.DataFrame,
                   stock_panel: pd.DataFrame | None = None,
                   universe: pd.DataFrame | None = None,
                   drop_label_na: bool = True) -> pd.DataFrame:
    """
    panel: 转债面板（含 code, trade_date, OHLCV）
    stock_panel: 正股面板（stock_code, trade_date, close），可选
    universe: 含 code, stock_code, conv_price，可选
    """
    parts = []
    for code, g in panel.groupby("code"):
        try:
            parts.append(_features_one(g, code))
        except Exception:
            continue
    out = pd.concat(parts, ignore_index=True)

    # ---- merge 正股数据 ----
    if stock_panel is not None and len(stock_panel) and universe is not None:
        try:
            uni = universe[["code", "stock_code", "conv_price"]].copy()
            uni["stock_code"] = uni["stock_code"].astype(str).str.zfill(6)
            out = out.merge(uni, on="code", how="left")
            sp = stock_panel[["stock_code", "trade_date", "close"]].rename(
                columns={"close": "stock_close"})
            out = out.merge(sp, on=["stock_code", "trade_date"], how="left")
            # 正股动量
            out = out.sort_values(["code", "trade_date"]).reset_index(drop=True)
            out["stock_ret1"] = out.groupby("code")["stock_close"].pct_change(1)
            out["stock_ret5"] = out.groupby("code")["stock_close"].pct_change(5)
            out["stock_ret10"] = out.groupby("code")["stock_close"].pct_change(10)
            out["stock_ret20"] = out.groupby("code")["stock_close"].pct_change(20)
            # 转债-正股背离
            out["div1"] = out["ret1"] - out["stock_ret1"]
            out["div5"] = out["ret5"] - out["stock_ret5"]
            # 转股价值 = 100 / 转股价 × 正股价；溢价率 = 转债价/转股价值 - 1
            out["conv_value"] = 100.0 / out["conv_price"] * out["stock_close"]
            out["premium"] = out["close"] / out["conv_value"] - 1.0
            out["premium_chg5"] = out.groupby("code")["premium"].diff(5)
        except Exception:
            for c in ["stock_ret1", "stock_ret5", "stock_ret10", "stock_ret20",
                      "div1", "div5", "premium", "premium_chg5"]:
                if c not in out.columns:
                    out[c] = 0.0

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    if drop_label_na:
        out = out.dropna(subset=["label"]).reset_index(drop=True)
    return out
