# -*- coding: utf-8 -*-
"""
train.py —— 入口：数据更新 -> 特征构建 -> 从头训练两套模型 -> 覆盖保存。

  - backtest 模型：训练数据固定截止 config.BACKTEST_TRAIN_END（默认 2023-12-31），
                   之后全部作为样本外回测区间，无未来函数
  - predict  模型：用全部最新数据训练（实盘用）
  - 训练过程中断点保护：每个模型训练完立即原子覆盖；任一环节 try/except 记录日志后继续。
"""
from __future__ import annotations
import sys, traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import config as cfg
from src import utils, data_loader, features, model as M


def main():
    logger = utils.setup_logger("train")
    logger.info("=" * 60)
    logger.info("train.py 启动：从头训练所有模型")

    # 1) 数据更新（失败不阻断后续：若本地已有数据则用旧数据继续）
    try:
        data_loader.update_all(logger)
    except Exception as e:
        logger.error(f"数据更新环节异常：{e}\n{traceback.format_exc()}")

    # 2) 加载面板 + 正股数据 + 特征
    try:
        panel = data_loader.load_panel(logger)
        universe = pd.read_csv(cfg.UNIVERSE_FILE, dtype={"code": str, "stock_code": str})
        try:
            data_loader.update_stocks(universe, logger)
        except Exception as e:
            logger.warning(f"正股更新异常（用已有缓存继续）：{e}")
        stock_panel = data_loader.load_stock_panel(logger)
        feat = features.build_features(panel, stock_panel, universe, drop_label_na=True)
        logger.info(f"特征面板：{len(feat)} 行，{feat['code'].nunique()} 券")
    except Exception as e:
        logger.error(f"特征构建失败，终止训练：{e}\n{traceback.format_exc()}")
        return

    # 3) 切分：backtest 模型训练数据固定截止 BACKTEST_TRAIN_END，之后全部为样本外回测区间
    try:
        cut = pd.Timestamp(cfg.BACKTEST_TRAIN_END)
        train_feat = feat[feat["trade_date"] <= cut].copy()
        holdout_feat = feat[feat["trade_date"] > cut].copy()
        if len(train_feat) < 1000:
            logger.error(f"backtest 训练数据过少（截止 {cut.date()} 仅 {len(train_feat)} 行），终止")
            return
        logger.info(f"backtest 模型训练区间: {train_feat['trade_date'].min().date()} ~ "
                    f"{train_feat['trade_date'].max().date()}（{len(train_feat)} 行）")
        if len(holdout_feat):
            logger.info(f"样本外回测区间: {holdout_feat['trade_date'].min().date()} ~ "
                        f"{holdout_feat['trade_date'].max().date()}（{len(holdout_feat)} 行）")
        else:
            logger.warning("holdout 区间为空（本地数据未超过截止日）")
    except Exception as e:
        logger.error(f"切分训练/回测区间失败：{e}\n{traceback.format_exc()}")
        return

    fcols = features.FEATURE_COLS

    # 4) 训练 backtest 模型（只用 holdout 之前的数据；按时间顺序 90/10 分割做早停）
    try:
        n = len(train_feat)
        split = int(n * 0.9)
        train_feat = train_feat.sort_values("trade_date").reset_index(drop=True)
        tr = train_feat.iloc[:split]
        va = train_feat.iloc[split:]
        booster_bt = M.train_model(tr[fcols], tr["label"], va[fcols], va["label"], logger)
        M.save_model_atomic(booster_bt, cfg.BACKTEST_MODEL_DIR, fcols, logger)
    except Exception as e:
        logger.error(f"backtest 模型训练异常：{e}\n{traceback.format_exc()}")

    # 5) 训练 predict 模型（用全部最新数据；按时间顺序末尾留 10% 做早停）
    try:
        n = len(feat)
        split = int(n * 0.9)
        feat_sorted = feat.sort_values("trade_date").reset_index(drop=True)
        tr = feat_sorted.iloc[:split]
        va = feat_sorted.iloc[split:]
        booster_pr = M.train_model(tr[fcols], tr["label"], va[fcols], va["label"], logger)
        M.save_model_atomic(booster_pr, cfg.PREDICT_MODEL_DIR, fcols, logger)
    except Exception as e:
        logger.error(f"predict 模型训练异常：{e}\n{traceback.format_exc()}")

    logger.info("train.py 完成")


if __name__ == "__main__":
    main()
