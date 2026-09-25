# -*- coding: utf-8 -*-
"""
backtest.py —— 用 backtest 模型在样本外区间（BACKTEST_TRAIN_END 之后）做回测。
输出 results/backtest_report.csv（每日持仓+收益）与控制台指标。
"""
from __future__ import annotations
import sys, traceback
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import config as cfg
from src import utils, data_loader, features, model as M
from src.backtest_engine import run_backtest


def main():
    logger = utils.setup_logger("backtest")
    logger.info("=" * 60)
    logger.info("backtest.py 启动")

    if not (cfg.BACKTEST_MODEL_DIR / "lgb_model.txt").exists():
        logger.error("未找到 backtest 模型，请先运行 train.py")
        return

    try:
        panel = data_loader.load_panel(logger)
        universe = pd.read_csv(cfg.UNIVERSE_FILE, dtype={"code": str, "stock_code": str})
        stock_panel = data_loader.load_stock_panel(logger)
        feat = features.build_features(panel, stock_panel, universe, drop_label_na=True)
    except Exception as e:
        logger.error(f"数据/特征构建失败：{e}\n{traceback.format_exc()}")
        return

    try:
        booster, fcols = M.load_model(cfg.BACKTEST_MODEL_DIR)
        cut = pd.Timestamp(cfg.BACKTEST_TRAIN_END)
        holdout = feat[feat["trade_date"] > cut].copy()
        holdout["pred"] = booster.predict(holdout[fcols])
        logger.info(f"样本外回测区间: {holdout['trade_date'].min().date()} ~ "
                    f"{holdout['trade_date'].max().date()}，{len(holdout)} 行")
    except Exception as e:
        logger.error(f"模型预测失败：{e}\n{traceback.format_exc()}")
        return

    try:
        res = run_backtest(holdout, pred_col="pred")
        m = res["metrics"]
        if not m:
            logger.warning("回测无有效交易")
            return
        logger.info("=" * 40)
        logger.info(f"回测天数       : {m['days']}（实际持仓 {m.get('active_days', m['days'])}）")
        logger.info(f"累计收益率     : {m['total_return']*100:.2f}%")
        logger.info(f"年化收益率     : {m['annual_return']*100:.2f}%")
        logger.info(f"年化波动       : {m['annual_vol']*100:.2f}%")
        logger.info(f"夏普比率       : {m['sharpe']:.2f}")
        logger.info(f"最大回撤       : {m['max_drawdown']*100:.2f}%")
        logger.info(f"日胜率         : {m['win_rate']*100:.2f}%")
        logger.info("=" * 40)

        # 按年度拆分
        daily = res["daily"].copy()
        daily["trade_date"] = pd.to_datetime(daily["trade_date"])
        daily["year"] = daily["trade_date"].dt.year
        logger.info("年度明细：")
        header = f"{'年份':<6}{'天数':>5}{'持仓':>5}{'收益%':>10}{'波动%':>8}{'夏普':>7}{'回撤%':>9}{'胜率%':>8}"
        logger.info(header)
        for yr, g in daily.groupby("year"):
            active = g[g["ret"] != 0]
            if len(active) == 0:
                continue
            yr_ret = (1 + g["ret"]).prod() - 1
            yr_vol = active["ret"].std() * np.sqrt(252)
            yr_sharpe = (yr_ret) / yr_vol if yr_vol > 1e-9 else 0
            nav = (1 + g["ret"]).cumprod()
            yr_dd = (nav / nav.cummax() - 1).min()
            yr_win = (active["ret"] > 0).mean()
            logger.info(f"{yr:<6}{len(g):>5}{len(active):>5}"
                        f"{yr_ret*100:>9.1f}%{yr_vol*100:>7.1f}%{yr_sharpe:>7.2f}"
                        f"{yr_dd*100:>8.1f}%{yr_win*100:>7.1f}%")

        # 最近一年按月拆分
        last_year = daily["year"].max()
        g26 = daily[daily["year"] == last_year].copy()
        g26["month"] = g26["trade_date"].dt.month
        logger.info(f"{last_year}年月度明细：")
        mheader = f"{'月份':<8}{'天数':>5}{'持仓':>5}{'收益%':>10}{'波动%':>8}{'夏普':>7}{'回撤%':>9}{'胜率%':>8}"
        logger.info(mheader)
        for mo, gm in g26.groupby("month"):
            active = gm[gm["ret"] != 0]
            if len(active) == 0:
                continue
            mo_ret = (1 + gm["ret"]).prod() - 1
            mo_vol = active["ret"].std() * np.sqrt(252)
            mo_sharpe = mo_ret / mo_vol if mo_vol > 1e-9 else 0
            nav = (1 + gm["ret"]).cumprod()
            mo_dd = (nav / nav.cummax() - 1).min()
            mo_win = (active["ret"] > 0).mean()
            logger.info(f"{mo:>2}月     {len(gm):>5}{len(active):>5}"
                        f"{mo_ret*100:>9.1f}%{mo_vol*100:>7.1f}%{mo_sharpe:>7.2f}"
                        f"{mo_dd*100:>8.1f}%{mo_win*100:>7.1f}%")
        logger.info("=" * 40)

        res["daily"].to_csv(cfg.RESULTS_DIR / "backtest_report.csv",
                            index=False, encoding="utf-8-sig")
        logger.info(f"回测明细已保存: {cfg.RESULTS_DIR / 'backtest_report.csv'}")
    except Exception as e:
        logger.error(f"回测执行异常：{e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
