# -*- coding: utf-8 -*-
"""
数据层：可转债日线下载、永久缓存、增量更新、异常隔离。

数据源：akshare -> 新浪财经可转债历史日线（sh/sz + 6位代码）。
缓存：每只转债存 data/raw/{code}.parquet；元数据存 data/meta/meta.json。
"""
from __future__ import annotations

import socket

socket.setdefaulttimeout(15)
import time
import traceback
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd
from tqdm import tqdm

from config import config as cfg
from src import utils


def _sina_symbol(code6: str) -> str:
    """6位转债代码 -> 新浪格式 sh11xxxx / sz12xxxx。"""
    code6 = str(code6).zfill(6)
    if code6.startswith(("11", "10", "13")):  # 沪市
        return "sh" + code6
    return "sz" + code6  # 深市 12 开头


# ---------------- 股票池 ----------------
def get_universe(logger=None) -> pd.DataFrame:
    """
    获取全市场可转债列表并过滤：
      - 已上市（上市时间非空）
      - 次新股：上市满 MIN_LISTED_DAYS 个自然日
      - 发行规模 >= MIN_OUTSTANDING_YI 亿
      - 债现价 >= MIN_BOND_PRICE
    返回 DataFrame，至少含 code/name/listed_date/stock_code。
    """
    log = logger or utils.setup_logger("universe")
    raw = ak.bond_zh_cov()
    raw["债券代码"] = raw["债券代码"].astype(str).str.zfill(6)

    df = raw.copy()
    df["上市时间"] = pd.to_datetime(df["上市时间"], errors="coerce")
    n0 = len(df)

    df = df[df["上市时间"].notna()].copy()  # 已上市
    cutoff = datetime.now() - timedelta(days=cfg.MIN_LISTED_DAYS)
    df = df[df["上市时间"] <= cutoff].copy()  # 次新股过滤
    df["发行规模"] = pd.to_numeric(df["发行规模"], errors="coerce")
    df = df[df["发行规模"].fillna(0) >= cfg.MIN_OUTSTANDING_YI]
    df["债现价"] = pd.to_numeric(df["债现价"], errors="coerce")
    df = df[df["债现价"].fillna(0) >= cfg.MIN_BOND_PRICE]
    df = df[~df["债券简称"].str.contains("退", na=False)].copy()  # 剔除退市/到期摘牌死券

    df = df.rename(columns={
        "债券代码": "code", "债券简称": "name", "正股代码": "stock_code",
        "正股简称": "stock_name", "上市时间": "listed_date",
        "转股价": "conv_price", "转股价值": "conv_value",
        "转股溢价率": "premium", "发行规模": "size_yi",
    })
    df = df[["code", "name", "stock_code", "stock_name", "listed_date",
             "conv_price", "conv_value", "premium", "size_yi"]].reset_index(drop=True)

    df.to_csv(cfg.UNIVERSE_FILE, index=False, encoding="utf-8-sig")
    log.info(f"股票池：原始 {n0} -> 过滤后 {len(df)}（次新/规模/价格过滤）")
    return df


# ---------------- 单只下载 ----------------
def _fetch_daily(code: str) -> pd.DataFrame | None:
    """拉单只转债全量日线；失败返回 None（由调用方隔离）。"""
    sym = _sina_symbol(code)
    df = ak.bond_zh_hs_cov_daily(symbol=sym)
    if df is None or len(df) == 0:
        return None
    df = df.rename(columns={"date": "trade_date"})
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("trade_date").reset_index(drop=True)
    return df


def _raw_path(code: str) -> Path:
    return cfg.RAW_DIR / f"{code}.parquet"


def update_all(logger=None) -> dict:
    """
    增量更新全部股票池日线。
    返回统计 dict: {ok, skipped, failed, failed_codes}。
    单股失败不影响全局；空下载不更新 meta（工作日盘前返回空场景）。
    """
    log = logger or utils.setup_logger("data")
    cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
    cfg.META_DIR.mkdir(parents=True, exist_ok=True)

    try:
        universe = get_universe(log)
    except Exception as e:
        log.error(f"获取股票池失败：{e}\n{traceback.format_exc()}")
        universe = pd.read_csv(cfg.UNIVERSE_FILE, dtype={"code": str}) if cfg.UNIVERSE_FILE.exists() else pd.DataFrame()

    if len(universe) == 0:
        log.warning("股票池为空，跳过下载")
        return {"ok": 0, "skipped": 0, "failed": 0, "failed_codes": []}

    meta = utils.load_meta()
    expected = utils.expected_latest_trade_date()
    log.info(f"预期最新交易日：{expected}，股票池 {len(universe)} 只")

    ok, skipped, failed = 0, 0, 0
    failed_codes = []

    pbar = tqdm(universe["code"].tolist(), desc="更新日线", ncols=100)
    for code in pbar:
        code = str(code).zfill(6)
        rec = meta.get(code, {})
        local_latest = None
        if rec.get("latest_date"):
            try:
                local_latest = datetime.strptime(rec["latest_date"], "%Y-%m-%d").date()
            except Exception:
                local_latest = None

        if not utils.should_download(local_latest, expected):
            skipped += 1
            continue

        # 死券：连续失败 >=3 次永久跳过（已退市/接口无数据）
        if rec.get("fail_count", 0) >= 3:
            skipped += 1
            continue

        # 死券冷却：最新数据距今 >180 天视为已退市/长期停牌，7 天内不重复请求
        if local_latest is not None and (expected - local_latest).days > 180:
            last_chk = rec.get("last_checked", "")
            if last_chk:
                try:
                    lc = datetime.strptime(last_chk, "%Y-%m-%d %H:%M:%S")
                    if (datetime.now() - lc).days < 7:
                        skipped += 1
                        continue
                except Exception:
                    pass

        try:
            time.sleep(cfg.DOWNLOAD_SLEEP_SEC)
            new_df = _fetch_daily(code)
            rec["last_checked"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if new_df is None or len(new_df) == 0:
                # 空下载：工作日盘前/节假日，不更新 meta，下次可重试
                log.info(f"{code} 返回空，跳过（不更新元数据）")
                failed += 0  # 不计入失败
                continue

            # 合并本地已有数据，去重，保留更长历史
            path = _raw_path(code)
            if path.exists():
                try:
                    old = pd.read_parquet(path)
                    old["trade_date"] = pd.to_datetime(old["trade_date"]).dt.date
                    combined = pd.concat([old, new_df], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["trade_date"], keep="last")
                    combined = combined.sort_values("trade_date").reset_index(drop=True)
                except Exception:
                    combined = new_df
            else:
                combined = new_df

            # 原子写入：先 tmp 再替换
            tmp = path.with_suffix(".parquet.tmp")
            combined.to_parquet(tmp, index=False)
            utils.os_replace(tmp, path)

            latest = combined["trade_date"].iloc[-1]
            # 只有当数据确实前进了才更新 meta（空/回退不更新）
            meta.setdefault(code, {})
            meta[code]["last_checked"] = rec["last_checked"]
            if (local_latest is None) or (latest > local_latest):
                meta[code].update({
                    "latest_date": latest.strftime("%Y-%m-%d"),
                    "rows": int(len(combined)),
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
            meta[code]["fail_count"] = 0
            ok += 1
            pbar.set_postfix(last=meta[code]["latest_date"])
        except Exception as e:
            failed += 1
            failed_codes.append(code)
            meta.setdefault(code, {})
            meta[code]["fail_count"] = meta[code].get("fail_count", 0) + 1
            log.warning(f"{code} 下载失败：{e}（已隔离，继续，失败{meta[code]['fail_count']}次）")
            continue

    utils.save_meta(meta)
    log.info(
        f"数据更新完成：成功 {ok}，跳过 {skipped}，失败 {failed}（{failed_codes[:10]}{'...' if len(failed_codes) > 10 else ''}）")
    return {"ok": ok, "skipped": skipped, "failed": failed, "failed_codes": failed_codes}


# ---------------- 面板加载 ----------------
def load_panel(logger=None) -> pd.DataFrame:
    """
    读取 data/raw/*.parquet，拼成长表面板：
    列: code, trade_date, open/high/low/close/volume
    """
    log = logger or utils.setup_logger("panel")
    files = sorted(cfg.RAW_DIR.glob("*.parquet"))
    if not files:
        raise RuntimeError("无本地数据，请先运行 train.py / 先执行数据下载")
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if len(df) == 0:
                continue
            df["code"] = f.stem
            frames.append(df)
        except Exception as e:
            log.warning(f"读取 {f.name} 失败：{e}，跳过")
    panel = pd.concat(frames, ignore_index=True)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel.sort_values(["code", "trade_date"]).reset_index(drop=True)
    log.info(f"面板加载：{panel['code'].nunique()} 只，{len(panel)} 行，"
             f"{panel['trade_date'].min().date()} ~ {panel['trade_date'].max().date()}")
    return panel


# ---------------- 正股日线（正股联动 + 转股溢价率） ----------------
def _stock_path(stock_code: str):
    return cfg.STOCK_RAW_DIR / f"{stock_code}.parquet"


def _sina_symbol(stock_code: str) -> str:
    sc = str(stock_code).zfill(6)
    if sc.startswith(("60", "68", "90", "11")):
        return "sh" + sc
    return "sz" + sc


def _fetch_stock(stock_code: str, retries: int = 2):
    sym = _sina_symbol(stock_code)
    for attempt in range(retries + 1):
        try:
            df = ak.stock_zh_a_daily(symbol=sym, start_date="20100101",
                                     end_date="20991231", adjust="qfq")
            if df is None or len(df) == 0:
                return None
            df = df.rename(columns={
                "date": "trade_date", "open": "open", "close": "close",
                "high": "high", "low": "low", "volume": "volume", "amount": "amount",
            })
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df[["trade_date", "open", "close", "high", "low", "volume", "amount"]]
            df = df.dropna(subset=["close"]).sort_values("trade_date").reset_index(drop=True)
            return df
        except Exception:
            if attempt < retries:
                time.sleep(2 + attempt * 2)
            else:
                raise


def update_stocks(universe: pd.DataFrame, logger=None):
    log = logger or utils.setup_logger("stock")
    cfg.STOCK_RAW_DIR.mkdir(parents=True, exist_ok=True)
    stocks = universe["stock_code"].dropna().astype(str).str.zfill(6).unique().tolist()
    ok, skipped, failed = 0, 0, 0
    failed_codes = []
    log.info(f"正股数据更新：{len(stocks)} 只唯一正股")
    for sc in tqdm(stocks, desc="更新正股", ncols=100):
        path = _stock_path(sc)
        if path.exists():
            try:
                old = pd.read_parquet(path)
                if len(old) and pd.to_datetime(old["trade_date"]).max() >= (datetime.now() - timedelta(days=5)):
                    skipped += 1;
                    continue
            except Exception:
                pass
        try:
            time.sleep(cfg.DOWNLOAD_SLEEP_SEC)
            new = _fetch_stock(sc)
            if new is None or len(new) == 0:
                continue
            tmp = path.with_suffix(".parquet.tmp")
            new.to_parquet(tmp, index=False)
            utils.os_replace(tmp, path)
            ok += 1
        except Exception as e:
            failed += 1;
            failed_codes.append(sc)
            log.warning(f"正股 {sc} 下载失败：{e}（已隔离）")
    log.info(f"正股更新完成：成功 {ok}，跳过 {skipped}，失败 {failed}")
    return {"ok": ok, "skipped": skipped, "failed": failed, "failed_codes": failed_codes}


def load_stock_panel(logger=None):
    log = logger or utils.setup_logger("stock_panel")
    files = sorted(cfg.STOCK_RAW_DIR.glob("*.parquet"))
    if not files:
        log.warning("无正股缓存");
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            df["stock_code"] = f.stem
            frames.append(df)
        except Exception as e:
            log.warning(f"读取正股 {f.name} 失败：{e}")
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    log.info(f"正股面板：{panel['stock_code'].nunique()} 只，{len(panel)} 行")
    return panel
