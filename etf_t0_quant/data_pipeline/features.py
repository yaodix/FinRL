"""Feature engineering and normalization."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

try:
    from etf_t0_quant.config import FeatureConfig
    from .constants import BARS_PER_DAY
except ImportError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from etf_t0_quant.config import FeatureConfig
    from etf_t0_quant.data_pipeline.constants import BARS_PER_DAY

def build_features(df: pd.DataFrame, cfg: FeatureConfig, interval: str) -> pd.DataFrame:
    """Compute all technical features. Input must have lowercase OHLCV columns."""
    data = df.copy()
    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]

    bars_per_day = BARS_PER_DAY.get(interval, 8)
    
    #--------第一层特征：基础统计---------
    
    data["returns"] = close.pct_change()
    data["log_ret"] = np.log(close / close.shift(1))
    data["amplitude"] = (high - low) / close.shift(1)

    data["volume_ma"] = volume.rolling(cfg.volume_ma_period).mean()
    data["volume_ratio"] = (volume / data["volume_ma"]).clip(0, 3)

    rng = high - low
    data["close_position"] = np.where(rng > 0, (close - low) / rng, 0.5)
    
    # --------第二层特征：技术指标---------
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(cfg.rsi_period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(cfg.rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    data["rsi"] = (100 - (100 / (1 + rs))).fillna(50)
    data["rsi_norm_raw"] = data["rsi"] / 100

    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    data["atr"] = tr.rolling(cfg.atr_period).mean()
    data["atr_ratio"] = (data["atr"] / close).clip(0, 0.1)

    ema_s = close.ewm(span=cfg.ema_short, adjust=False).mean()
    ema_l = close.ewm(span=cfg.ema_long, adjust=False).mean()
    data["ema_diff"] = ((ema_s - ema_l) / close).clip(-0.1, 0.1) # MACD核心特征

    ma = close.rolling(cfg.bb_period).mean()
    std = close.rolling(cfg.bb_period).std()
    bb_upper = ma + cfg.bb_std * std
    bb_lower = ma - cfg.bb_std * std
    data["bb_width"] = ((bb_upper - bb_lower) / ma).clip(0, 0.2)

    ann = 252 * bars_per_day
    data["vol20"] = data["log_ret"].rolling(20).std() * np.sqrt(ann) # 年化波动率

    if "trade_time" in data.columns:
        # 假设trade_time为datetime类型或字符串
        if not np.issubdtype(data["trade_time"].dtype, np.datetime64):
            data["trade_time"] = pd.to_datetime(data["trade_time"])
        hm = data["trade_time"].dt.hour * 60 + data["trade_time"].dt.minute
        # 以A股常见交易时间 9:30-15:00（570-900分钟）为例
        data["time_slot"] = (hm - 570) / (900 - 570)
        data["time_slot"] = data["time_slot"].clip(0, 1)
        # 收盘前2个bar
        bar_minutes = 30 if interval == "30m" else 15
        data["is_near_close"] = (hm >= (900 - 2 * bar_minutes)).astype(float)
    else:
        data["time_slot"] = 0.5
        data["is_near_close"] = 0.0
        
    data = data.dropna()

    return data


def normalize_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Rolling z-score normalization without future leakage."""
    result = df.copy()
    eps = 1e-8
    w = cfg.normalization_window
    mp = cfg.normalization_min_periods

    zscore_cols = ["returns", "amplitude", "atr_ratio", "ema_diff", "bb_width", "vol20"]
    for col in zscore_cols:
        if col not in df.columns:
            continue
        mu = df[col].rolling(w, min_periods=mp).mean().shift(1)
        sigma = df[col].rolling(w, min_periods=mp).std().shift(1)
        result[f"{col}_norm"] = ((df[col] - mu) / (sigma + eps)).clip(-3, 3)

    if "volume_ratio" in df.columns:
        med = df["volume_ratio"].rolling(w, min_periods=mp).median().shift(1)
        q75 = df["volume_ratio"].rolling(w, min_periods=mp).quantile(0.75).shift(1)
        q25 = df["volume_ratio"].rolling(w, min_periods=mp).quantile(0.25).shift(1)
        iqr = (q75 - q25).replace(0, np.nan)
        result["volume_ratio_norm"] = ((df["volume_ratio"] - med) / (iqr + eps)).clip(-3, 3)

    if "rsi_norm_raw" in df.columns:
        result["rsi_norm"] = df["rsi_norm_raw"] * 2 - 1

    for col in ("close_position", "time_slot", "is_near_close"):
        if col in df.columns:
            result[col] = df[col]
    # 保留归一化后的特征列和部分原始特征列，丢弃其他中间计算列
    available_cols = [c for c in result.columns if c.endswith("_norm") or c in ("trade_time","close_position", "time_slot", "is_near_close", "open")]
    result = result[available_cols]
    result = result.dropna()
    return result


if __name__ == "__main__":
    df = pd.read_csv("/home/yao/Work/myproject/FinRL/etf_t0_quant/workdata/159740_sync_30m/159740_sync_30m.csv")
    
    
    features = build_features(df, FeatureConfig(), "30m")
    # print(features.tail(20))
    features_norm = normalize_features(features, FeatureConfig())
    print(features_norm)
    print(features_norm.columns)