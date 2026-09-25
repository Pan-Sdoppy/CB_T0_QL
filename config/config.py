# -*- coding: utf-8 -*-
"""全局配置：所有可调参数集中在此。"""
from pathlib import Path

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # 每只转债一只 parquet
STOCK_RAW_DIR = DATA_DIR / "stock_raw"  # 正股日线缓存
META_DIR = DATA_DIR / "meta"
UNIVERSE_FILE = DATA_DIR / "universe.csv"
META_FILE = META_DIR / "meta.json"  # 每只 code 的最新日期元数据

MODELS_DIR = ROOT / "models"
BACKTEST_MODEL_DIR = MODELS_DIR / "latest" / "backtest"
PREDICT_MODEL_DIR = MODELS_DIR / "latest" / "predict"

LOG_DIR = ROOT / "logs"
RESULTS_DIR = ROOT / "results"

# ---------- 股票池过滤 ----------
MIN_LISTED_DAYS = 60  # 次新股：上市不足 60 个自然日剔除
MIN_OUTSTANDING_YI = 1.0  # 发行规模 < 1 亿剔除（流动性过滤）
MIN_BOND_PRICE = 95.0  # 过滤深度低价/可能违约的券（可选保守线）
EXCLUDE_ST = True  # 剔除正股 ST（后续用名称判断）

# ---------- 数据下载 ----------
DOWNLOAD_SLEEP_SEC = 0.3  # 每只请求间隔，防新浪封 IP
WORKDAY_SKIP_WINDOW = 1  # 工作日 skip 窗口（自然日）
WEEKEND_SKIP_WINDOW = 3  # 周末 skip 窗口
MARKET_CLOSE_HOUR = 15  # 收盘时点
HIST_START_DATE = None  # None = 拉全量历史

# ---------- 特征 / 标签 ----------
LABEL_HORIZON = 1  # 预测未来 1 个交易日收益
MIN_HISTORY_BARS = 60  # 单券至少 60 根 K 线才进训练

# ---------- 模型 ----------
TOP_N = 1  # 每日选前 N 只等权持有
HOLDOUT_DAYS = 120  # fallback：数据不足时，最后 N 个交易日作为 holdout
# backtest 模型训练数据的固定截止日：此日及之前用于训练，之后全部作为样本外回测区间
BACKTEST_TRAIN_END = "2023-12-31"
RANDOM_SEED = 42

# LightGBM 参数（RTX 4070 Ti Super 16GB；device 自动探测 cuda/gpu/cpu）
LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": RANDOM_SEED,
    "num_threads": 8,
}
NUM_BOOST_ROUND = 500
EARLY_STOPPING = 50

# ---------- 每日选股过滤（回测与预测共用）----------
MIN_AVG_AMOUNT_YI = 0.5  # 过去20日日均成交额 >= 0.5亿（流动性门槛）
PREMIUM_LOW = -0.05  # 转股溢价率 >= -5%（剔除深度价内可能强赎）
PREMIUM_HIGH = 0.50  # 转股溢价率 <= 50%（剔除高溢价跟涨弱）

# ---------- 二次过滤：当天涨幅区间（网格最优 -4% ~ +5%）----------
RET1_LOW = -0.04  # 当天涨幅 >= -4%
RET1_HIGH = 0.05  # 当天涨幅 <= +5%

# ---------- 盘中止盈止损（T+1 开盘买入后挂条件单）----------
TAKE_PROFIT = 0.10  # 涨10%止盈
STOP_LOSS_INTRADAY = 0.05  # 跌5%止损

# ---------- 仓位/风控 ----------
STOP_LOSS_DAILY = 0.03  # 昨日组合亏损 > 3%，今日空仓
USE_TREND_FILTER = True  # 池等权指数 < MA20 时半仓
TREND_HALF_POS = 0.5  # 趋势过滤时的仓位比例

# ---------- 回测交易成本（可转债 T+0，佣金约万0.5，无印花税）----------
COMMISSION_RATE = 0.0002  # 单边万 2（含佣金，保守）
SLIPPAGE = 0.0002  # 单边滑点
