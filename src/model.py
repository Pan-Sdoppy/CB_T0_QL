# -*- coding: utf-8 -*-
"""模型层：LightGBM 回归，GPU/CPU 自适应，原子覆盖保存。"""
from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from config import config as cfg
from src import utils


def _resolve_device(logger: logging.Logger) -> str:
    """
    LightGBM device 选择。
    环境变量 CB_LGB_DEVICE 可强制指定（cuda/gpu/cpu）。
    默认 cpu：此 pip wheel 未编译 CUDA，OpenCL 后端在 Windows 下有路径编码崩溃；
    日频截面样本量（~50 万行）CPU 训练仅需数十秒，收益/时间比最优。
    RTX 4070 Ti Super 16GB 已验证 torch.cuda 可用，留作深度模型扩展。
    """
    import os
    forced = os.environ.get("CB_LGB_DEVICE", "").lower()
    if forced in ("cuda", "gpu", "cpu"):
        logger.info(f"LightGBM device 强制指定：{forced}")
        return forced
    logger.info("LightGBM device = cpu（稳定路径；CB_LGB_DEVICE=gpu 可尝试 OpenCL）")
    return "cpu"


def train_model(X: pd.DataFrame, y: pd.Series,
                X_valid: pd.DataFrame | None = None,
                y_valid: pd.Series | None = None,
                logger: logging.Logger | None = None) -> lgb.Booster:
    log = logger or utils.setup_logger("train")
    device = _resolve_device(log)
    params = dict(cfg.LGB_PARAMS)
    params["device"] = device

    train_set = lgb.Dataset(X, label=y)
    valid_sets = [train_set]
    valid_names = ["train"]
    if X_valid is not None and len(X_valid) > 50:
        vset = lgb.Dataset(X_valid, label=y_valid, reference=train_set)
        valid_sets.append(vset)
        valid_names.append("valid")

    booster = lgb.train(
        params, train_set,
        num_boost_round=cfg.NUM_BOOST_ROUND,
        valid_sets=valid_sets, valid_names=valid_names,
        callbacks=[
            lgb.early_stopping(cfg.EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(50),
        ],
    )
    log.info(f"训练完成：best_iter={booster.best_iteration}, "
             f"best_score={booster.best_score.get('valid', {}).get('rmse', float('nan')):.6f}")
    return booster


def save_model_atomic(booster: lgb.Booster, out_dir: Path,
                      feature_cols: list[str], logger=None) -> None:
    """覆盖保存模型（先写 tmp 再替换），同时保存特征列清单。"""
    log = logger or utils.setup_logger("save")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tmp = out_dir / "model.txt.tmp"
    booster.save_model(str(model_tmp))
    utils.os_replace(model_tmp, out_dir / "model.txt")
    utils.atomic_write_text("\n".join(feature_cols), out_dir / "features.txt")
    log.info(f"模型已覆盖保存到 {out_dir}")


def load_model(out_dir: Path):
    booster = lgb.Booster(model_file=str(out_dir / "model.txt"))
    features = (out_dir / "features.txt").read_text(encoding="utf-8").strip().split("\n")
    return booster, features
