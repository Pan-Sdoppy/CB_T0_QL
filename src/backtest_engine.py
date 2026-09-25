# -*- coding: utf-8 -*-
"""回测引擎：日频横截面 Top-N 等权，带流动性/溢价率过滤 + 止损 + 趋势仓位。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import config as cfg


def run_backtest(feat: pd.DataFrame, pred_col: str = "pred") -> dict:
    """
    feat: 含 trade_date, code, next_ret, pred, premium, avg_amount_20
    规则：
      1) 每日先过滤：avg_amount_20 >= MIN_AVG_AMOUNT_YI*1e8，premium 在 [LOW, HIGH]
      2) 选 pred 最高的 Top-N 等权
      3) 昨日组合亏损 > STOP_LOSS_DAILY -> 今日空仓
      4) 池等权指数 < MA20 -> 今日半仓
    """
    cost = (cfg.COMMISSION_RATE + cfg.SLIPPAGE) * 2

    # 每日全池等权指数用于趋势过滤
    daily_pool = feat.groupby("trade_date")["next_ret"].mean().sort_index()
    pool_ma20 = daily_pool.rolling(20).mean()

    rows = []
    prev_ret = 0.0
    for dt, g in feat.groupby("trade_date"):
        g = g.dropna(subset=[pred_col, "next_ret"])

        # 风控仓位
        position = 1.0
        if prev_ret < -cfg.STOP_LOSS_DAILY:
            position = 0.0
        elif cfg.USE_TREND_FILTER:
            ma20 = pool_ma20.get(dt, np.nan)
            cur = daily_pool.get(dt, np.nan)
            if pd.notna(ma20) and pd.notna(cur) and cur < ma20:
                position = cfg.TREND_HALF_POS

        # 流动性 + 溢价率过滤（硬过滤）
        g = g[g["avg_amount_20"] >= cfg.MIN_AVG_AMOUNT_YI * 1e8]
        g = g[(g["premium"] >= cfg.PREMIUM_LOW) & (g["premium"] <= cfg.PREMIUM_HIGH)]

        if len(g) == 0 or position == 0.0:
            rows.append({"trade_date": dt, "ret": 0.0, "n": 0, "pos": position, "names": ""})
            prev_ret = 0.0
            continue

        # 当天涨幅区间过滤：剔除不在区间的票，剩下的选 Top1
        g = g[(g["ret1"] >= cfg.RET1_LOW) & (g["ret1"] <= cfg.RET1_HIGH)]
        if len(g) == 0:
            rows.append({"trade_date": dt, "ret": 0.0, "n": 0, "pos": position, "names": ""})
            prev_ret = 0.0
            continue

        top = g.nlargest(cfg.TOP_N, pred_col)
        # 盘中止盈止损模拟：开盘价买入，按 high/low 判断触发
        tp_price_mult = 1 + cfg.TAKE_PROFIT
        sl_price_mult = 1 - cfg.STOP_LOSS_INTRADAY
        actual_rets = []
        for _, row in top.iterrows():
            o = row.get("next_open", np.nan)
            h = row.get("next_high", np.nan)
            l = row.get("next_low", np.nan)
            if pd.isna(o) or pd.isna(h) or pd.isna(l):
                actual_rets.append(row["next_ret"])
                continue
            sl_price = o * sl_price_mult
            tp_price = o * tp_price_mult
            if l <= sl_price:
                actual_rets.append(-cfg.STOP_LOSS_INTRADAY)
            elif h >= tp_price:
                actual_rets.append(cfg.TAKE_PROFIT)
            else:
                actual_rets.append(row["next_ret"])
        port_ret = (np.mean(actual_rets) - cost) * position
        rows.append({"trade_date": dt, "ret": port_ret, "n": len(top),
                     "pos": position, "names": ",".join(top["code"].astype(str))})
        prev_ret = port_ret

    daily = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    if len(daily) == 0:
        return {"equity": pd.DataFrame(), "metrics": {}, "daily": daily}

    daily["nav"] = (1.0 + daily["ret"]).cumprod()
    total_days = len(daily)
    years = total_days / 252.0
    total_ret = daily["nav"].iloc[-1] - 1
    ann_ret = (daily["nav"].iloc[-1]) ** (1 / years) - 1 if years > 0 else 0.0
    active = daily[daily["ret"] != 0]
    ann_vol = active["ret"].std() * np.sqrt(252) if len(active) else 0.0
    sharpe = ann_ret / ann_vol if ann_vol > 1e-9 else 0.0
    cummax = daily["nav"].cummax()
    max_dd = (daily["nav"] / cummax - 1).min()
    win_rate = (active["ret"] > 0).mean() if len(active) else 0.0

    metrics = {
        "days": total_days,
        "active_days": int(len(active)),
        "total_return": total_ret,
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
    }
    return {"equity": daily[["trade_date", "nav"]], "metrics": metrics, "daily": daily}
