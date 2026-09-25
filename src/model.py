# -*- coding: utf-8 -*-
"""模型层：LightGBM + XGBoost 等权集成，GPU/CPU 自适应，原子覆盖保存。"""
from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd

from config import config as cfg
from src import utils


class Ensemble:
    """LightGBM + XGBoost 等权集成预测器。"""
    def __init__(self, lgb_model, xgb_model):
        self.lgb = lgb_model
        self.xgb = xgb_model

    def predict(self, X):
        p_lgb = self.lgb.predict(X)
        if isinstance(X, pd.DataFrame):
            p_xgb = self.xgb.predict(xgb.DMatrix(X.values, feature_names=list(X.columns)))
        else:
            p_xgb = self.xgb.predict(xgb.DMatrix(X))
        return (p_lgb + p_xgb) / 2.0


def _resolve_device(logger: logging.Logger) -> str:
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
                logger: logging.Logger | None = None) -> Ensemble:
    log = logger or utils.setup_logger("train")
    device = _resolve_device(log)

    # ---------- LightGBM ----------
    lgb_params = dict(cfg.LGB_PARAMS)
    lgb_params["device"] = device
    train_set = lgb.Dataset(X, label=y)
    valid_sets = [train_set]
    valid_names = ["train"]
    if X_valid is not None and len(X_valid) > 50:
        vset = lgb.Dataset(X_valid, label=y_valid, reference=train_set)
        valid_sets.append(vset)
        valid_names.append("valid")
    lgb_model = lgb.train(
        lgb_params, train_set,
        num_boost_round=cfg.NUM_BOOST_ROUND,
        valid_sets=valid_sets, valid_names=valid_names,
        callbacks=[
            lgb.early_stopping(cfg.EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(50),
        ],
    )
    log.info(f"LightGBM 完成：best_iter={lgb_model.best_iteration}, "
             f"best_score={lgb_model.best_score.get('valid', {}).get('rmse', float('nan')):.6f}")

    # ---------- XGBoost ----------
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "seed": cfg.RANDOM_SEED,
        "tree_method": "hist",
    }
    dtr = xgb.DMatrix(X.values, label=y.values, feature_names=list(X.columns))
    evals = [(dtr, "train")]
    if X_valid is not None and len(X_valid) > 50:
        dval = xgb.DMatrix(X_valid.values, label=y_valid.values, feature_names=list(X.columns))
        evals.append((dval, "valid"))
    xgb_model = xgb.train(
        xgb_params, dtr,
        num_boost_round=cfg.NUM_BOOST_ROUND,
        evals=evals,
        early_stopping_rounds=cfg.EARLY_STOPPING,
        verbose_eval=False,
    )
    log.info(f"XGBoost 完成：best_iter={xgb_model.best_iteration}, "
             f"best_score={xgb_model.best_score:.6f}")

    return Ensemble(lgb_model, xgb_model)


def save_model_atomic(model: Ensemble, out_dir: Path,
                      feature_cols: list[str], logger=None) -> None:
    """覆盖保存 LGB + XGB 双模型（先写 tmp 再替换）。"""
    log = logger or utils.setup_logger("save")
    out_dir.mkdir(parents=True, exist_ok=True)

    lgb_tmp = out_dir / "lgb_model.txt.tmp"
    model.lgb.save_model(str(lgb_tmp))
    utils.os_replace(lgb_tmp, out_dir / "lgb_model.txt")

    xgb_path = out_dir / "xgb_model.json"
    model.xgb.save_model(str(xgb_path))

    utils.atomic_write_text("\n".join(feature_cols), out_dir / "features.txt")
    log.info(f"集成模型已覆盖保存到 {out_dir}（LGB+XGB）")


def load_model(out_dir: Path):
    lgb_model = lgb.Booster(model_file=str(out_dir / "lgb_model.txt"))
    xgb_model = xgb.Booster()
    xgb_model.load_model(str(out_dir / "xgb_model.json"))
    features = (out_dir / "features.txt").read_text(encoding="utf-8").strip().split("\n")
    return Ensemble(lgb_model, xgb_model), features
