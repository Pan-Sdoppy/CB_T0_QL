# -*- coding: utf-8 -*-
"""
predict.py —— 用 predict 模型对最新一个交易日横截面打分，输出今日 Top-N 推荐。
输出 results/today_picks.csv
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from config import config as cfg
from src import utils, data_loader, features, model as M


def _fmt_row(r, rank):
    """格式化一行股票信息。"""
    code = str(r.get("code", "")).zfill(6)
    name = str(r.get("name", ""))
    stock = str(r.get("stock_name", ""))
    close = r.get("close", float("nan"))
    ret1 = r.get("ret1", float("nan"))
    prem = r.get("premium", float("nan"))
    pred = r.get("pred", float("nan"))
    vma = r.get("vma20_ratio", float("nan"))
    amt = r.get("avg_amount_20", float("nan"))
    close_s = f"{close:.2f}" if pd.notna(close) else "  N/A"
    ret_s = f"{ret1 * 100:+.2f}%" if pd.notna(ret1) else "  N/A"
    prem_s = f"{prem * 100:+.1f}%" if pd.notna(prem) else "  N/A"
    pred_s = f"{pred:.4f}" if pd.notna(pred) else "  N/A"
    vma_s = f"{vma:.2f}" if pd.notna(vma) else " N/A"
    amt_s = f"{amt / 1e8:.1f}亿" if pd.notna(amt) and amt > 0 else "  N/A"
    return (f"  #{rank:<3} {code} {name:<8} {stock:<6} "
            f"价{close_s:>8} 涨幅{ret_s:>8} 溢价{prem_s:>7} "
            f"量比{vma_s:>5} 成交{amt_s:>6} 置信{pred_s}")


def main():
    logger = utils.setup_logger("predict")
    logger.info("=" * 60)
    logger.info("predict.py 启动")

    if not (cfg.PREDICT_MODEL_DIR / "lgb_model.txt").exists():
        logger.error("未找到 predict 模型，请先运行 train.py")
        return

    # 1) 增量更新（失败不阻断，用本地最新数据）
    try:
        data_loader.update_all(logger)
    except Exception as e:
        logger.warning(f"预测前增量更新异常（继续用本地数据）：{e}")

    try:
        panel = data_loader.load_panel(logger)
        universe = pd.read_csv(cfg.UNIVERSE_FILE, dtype={"code": str, "stock_code": str})
        stock_panel = data_loader.load_stock_panel(logger)
        feat = features.build_features(panel, stock_panel, universe, drop_label_na=False)
    except Exception as e:
        logger.error(f"特征构建失败：{e}\n{traceback.format_exc()}")
        return

    try:
        booster, fcols = M.load_model(cfg.PREDICT_MODEL_DIR)
        feat = feat.sort_values("trade_date")
        latest = feat.groupby("code").tail(1).copy()
        global_latest = feat["trade_date"].max()
        latest = latest[latest["trade_date"] == global_latest].copy()

        # 合并名称
        try:
            uni = pd.read_csv(cfg.UNIVERSE_FILE, dtype={"code": str})
            latest = latest.merge(uni[["code", "name", "stock_name"]], on="code", how="left")
        except Exception:
            pass

        # 流动性 + 溢价率过滤
        latest = latest[latest["avg_amount_20"] >= cfg.MIN_AVG_AMOUNT_YI * 1e8]
        latest = latest[(latest["premium"] >= cfg.PREMIUM_LOW) &
                        (latest["premium"] <= cfg.PREMIUM_HIGH)]
        logger.info(f"最新交易日: {pd.to_datetime(global_latest).date()}，"
                    f"流动性+溢价率过滤后 {len(latest)} 只")

        latest["pred"] = booster.predict(latest[fcols])
        latest = latest.sort_values("pred", ascending=False).reset_index(drop=True)

        # ===== 第一步：打印全部按置信度排名 =====
        logger.info("-" * 60)
        logger.info(f"【模型全量排名】共 {len(latest)} 只，按置信度降序：")
        logger.info(f"  {'排名':<4}{'代码':<7}{'转债名':<9}{'正股':<7}"
                    f"{'最新价':>9}{'当天涨幅':>9}{'溢价率':>8}{'量比':>6}{'日均成交':>8}{'置信度':>9}")
        for i, r in latest.iterrows():
            logger.info(_fmt_row(r, i + 1))

        # ===== 第二步：当天涨幅区间过滤 =====
        filtered = latest[(latest["ret1"] >= cfg.RET1_LOW) &
                          (latest["ret1"] <= cfg.RET1_HIGH)].copy()
        logger.info("-" * 60)
        logger.info(f"【区间过滤】当天涨幅 [{cfg.RET1_LOW * 100:.0f}%, {cfg.RET1_HIGH * 100:.0f}%]，"
                    f"剩余 {len(filtered)} 只（剔除 {len(latest) - len(filtered)} 只）")

        if len(filtered) == 0:
            logger.info("区间内无符合条件的券，今日空仓")
            return

        filtered = filtered.sort_values("pred", ascending=False).reset_index(drop=True)

        # ===== 第三步：打印最终前十 =====
        show_n = min(10, len(filtered))
        logger.info(f"【最终排名 Top-{show_n}】：")
        logger.info(f"  {'排名':<4}{'代码':<7}{'转债名':<9}{'正股':<7}"
                    f"{'最新价':>9}{'当天涨幅':>9}{'溢价率':>8}{'量比':>6}{'日均成交':>8}{'置信度':>9}")
        for i in range(show_n):
            logger.info(_fmt_row(filtered.iloc[i], i + 1))

        # ===== 输出 Top-N 推荐 =====
        top = filtered.head(cfg.TOP_N).copy()
        out_cols = ["code", "name", "stock_name", "trade_date", "close", "ret1",
                    "premium", "vma20_ratio", "avg_amount_20", "pred"]
        out_cols = [c for c in out_cols if c in top.columns]
        out = top[out_cols].copy()
        out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.strftime("%Y-%m-%d")
        out["take_profit"] = (out["close"] * (1 + cfg.TAKE_PROFIT)).round(2)
        out["stop_loss"] = (out["close"] * (1 - cfg.STOP_LOSS_INTRADAY)).round(2)
        out.to_csv(cfg.RESULTS_DIR / "today_picks.csv", index=False, encoding="utf-8-sig")

        logger.info("-" * 60)
        logger.info(f"今日 Top-{cfg.TOP_N} 推荐（明日开盘买入，"
                    f"止盈{cfg.TAKE_PROFIT * 100:.0f}%/止损{cfg.STOP_LOSS_INTRADAY * 100:.0f}%）：")
        for _, r in out.iterrows():
            logger.info(f"  {r['code']} {r.get('name', '')} "
                        f"价{r['close']:.2f} 涨幅{r['ret1'] * 100:+.2f}% "
                        f"溢价{r['premium'] * 100:+.1f}% 置信{r['pred']:.4f}")
            logger.info(f"    止盈 {r['take_profit']:.2f} (+{cfg.TAKE_PROFIT * 100:.0f}%) | "
                        f"止损 {r['stop_loss']:.2f} (-{cfg.STOP_LOSS_INTRADAY * 100:.0f}%)")
        logger.info(f"已保存: {cfg.RESULTS_DIR / 'today_picks.csv'}")
    except Exception as e:
        logger.error(f"预测异常：{e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
