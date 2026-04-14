"""
Data pipeline: fetch → [resample] → validate → clean → feature engineering → save.

Public entry point::

    pipeline = DataPipeline(cfg)
    pipeline.run(mode="full")      # full historical fetch
    pipeline.run(mode="incremental")

Outputs written to::

    data/raw/{symbol}/{interval}/data.parquet
    data/cleaned/{symbol}/{interval}/data.parquet
    data/features/{symbol}/{interval}/data.parquet
    data/reports/{run_id}_quality.json
    data/metadata/{run_id}_meta.json
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import AppConfig, DataConfig, FeatureConfig
from .logger import get_logger, get_run_id

log = get_logger("data")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BARS_PER_DAY: Dict[str, int] = {"30m": 8, "15m": 16}

# A-share ETF trading time: 09:30–11:30 and 13:00–15:00
_TRADE_START = pd.Timestamp("09:30:00").time()
_TRADE_END = pd.Timestamp("15:00:00").time()


# ---------------------------------------------------------------------------
# 1-min → target interval resample helpers (ref: gen_30min_man.py)
# ---------------------------------------------------------------------------

def _30min_label(ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    t = ts.strftime("%H:%M")
    rules = [
        ("09:30", "10:00", 10, 0),
        ("10:00", "10:30", 10, 30),
        ("10:30", "11:00", 11, 0),
        ("11:00", "11:30", 11, 30),
        ("13:00", "13:30", 13, 30),
        ("13:30", "14:00", 14, 0),
        ("14:00", "14:30", 14, 30),
        ("14:30", "15:00", 15, 0),
    ]
    for start, end, h, m in rules:
        if start <= t <= end:
            return ts.replace(hour=h, minute=m, second=0, microsecond=0)
    return None


def _15min_label(ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    t = ts.strftime("%H:%M")
    rules = [
        ("09:30", "09:45", 9, 45),
        ("09:45", "10:00", 10, 0),
        ("10:00", "10:15", 10, 15),
        ("10:15", "10:30", 10, 30),
        ("10:30", "10:45", 10, 45),
        ("10:45", "11:00", 11, 0),
        ("11:00", "11:15", 11, 15),
        ("11:15", "11:30", 11, 30),
        ("13:00", "13:15", 13, 15),
        ("13:15", "13:30", 13, 30),
        ("13:30", "13:45", 13, 45),
        ("13:45", "14:00", 14, 0),
        ("14:00", "14:15", 14, 15),
        ("14:15", "14:30", 14, 30),
        ("14:30", "14:45", 14, 45),
        ("14:45", "15:00", 15, 0),
    ]
    for start, end, h, m in rules:
        if start <= t <= end:
            return ts.replace(hour=h, minute=m, second=0, microsecond=0)
    return None


def resample_1min_to_interval(df_1m: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Resample 1-minute OHLCV data to 15m or 30m bars.

    Input index must be DatetimeIndex.  Columns required: open high low close volume.
    Optional: amount.
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "amount" in df_1m.columns:
        agg["amount"] = "sum"

    label_fn = _30min_label if interval == "30m" else _15min_label

    df = df_1m.copy()
    df = df[(df.index.time >= _TRADE_START) & (df.index.time <= _TRADE_END)]
    df["_label"] = df.index.map(label_fn)
    df = df.dropna(subset=["_label"])
    result = df.groupby("_label").agg({k: v for k, v in agg.items() if k in df.columns})
    result.index.name = "timestamp"
    return result.dropna(how="all")


# ---------------------------------------------------------------------------
# Data source abstraction
# ---------------------------------------------------------------------------

class BaseDataFetcher(ABC):
    """Abstract fetcher; subclass for tickflow or local file."""

    @abstractmethod
    def fetch_full(self, symbol: str, interval: str) -> pd.DataFrame:
        """Return a DataFrame with columns [open, high, low, close, volume, (amount)]."""

    @abstractmethod
    def fetch_incremental(self, symbol: str, interval: str, since: datetime) -> pd.DataFrame:
        """Return bars since *since* (exclusive lower bound)."""


class LocalFileFetcher(BaseDataFetcher):
    """Read historical data from a local CSV or Parquet file."""

    def __init__(self, cfg: DataConfig) -> None:
        self._path = cfg.local_csv_path
        self._needs_resample = cfg.needs_resample

    def _load(self) -> pd.DataFrame:
        if not self._path:
            raise ValueError("data.local_csv_path must be set when source=local")
        p = Path(self._path)
        if not p.exists():
            raise FileNotFoundError(f"Local data file not found: {p}")
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
        # Normalise timestamp column
        ts_col = next(
            (c for c in df.columns if c.lower() in ("timestamp", "trade_time", "datetime", "date")),
            None,
        )
        if ts_col is None:
            raise ValueError("Cannot detect timestamp column in local file.")
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.rename(columns={ts_col: "timestamp"})
        df = df.set_index("timestamp")
        df.columns = df.columns.str.lower()
        return df

    def fetch_full(self, symbol: str, interval: str) -> pd.DataFrame:
        df = self._load()
        if self._needs_resample:
            df = resample_1min_to_interval(df, interval)
        return df

    def fetch_incremental(self, symbol: str, interval: str, since: datetime) -> pd.DataFrame:
        df = self.fetch_full(symbol, interval)
        since_ts = pd.Timestamp(since)
        return df[df.index > since_ts]


class TickflowFetcher(BaseDataFetcher):
    """Fetch data via the tickflow Python SDK (pip install 'tickflow[all]').

    SDK docs: https://docs.tickflow.org/zh-Hans/sdk/python-quickstart
    Auth: TickFlow(api_key=token)
    Kline: tf.klines.get(symbol, period="30m", count=10000, as_dataframe=True)
    """

    def __init__(self, cfg: DataConfig) -> None:
        token = cfg.tickflow_token or os.environ.get("ETF_TICKFLOW_TOKEN", "")
        if not token:
            raise ValueError(
                "tickflow token not set. "
                "Pass data.tickflow_token in config or set ETF_TICKFLOW_TOKEN env var."
            )
        try:
            from tickflow import TickFlow  # type: ignore
            self._tf = TickFlow(api_key=token)
        except ImportError as exc:
            raise ImportError(
                "tickflow package not installed. Run: pip install 'tickflow[all]'"
            ) from exc
        self._needs_resample = cfg.needs_resample

    @staticmethod
    def _tf_symbol(symbol: str) -> str:
        """Add market suffix if missing.  e.g. '159740' → '159740.SZ'."""
        if "." in symbol:
            return symbol
        code = symbol.strip()
        # Shanghai: 60xxxx (stocks), 51xxxx/50xxxx/58xxxx/56xxxx (ETF)
        if code[:2] in ("60", "68", "51", "50", "56", "58"):
            return f"{code}.SH"
        return f"{code}.SZ"

    def _norm(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Normalise tickflow DataFrame → DatetimeIndex OHLCV frame."""
        df = df.copy()
        df.columns = df.columns.str.lower()
        # trade_time is e.g. '2026-04-03 15:00:00' – use it as bar timestamp
        if "trade_time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["trade_time"])
        elif "timestamp" in df.columns:
            # SDK returns ms-epoch integer for daily bars
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        else:
            raise ValueError("tickflow response has no recognisable timestamp column")
        df = df.set_index("timestamp")
        df.index.name = "timestamp"
        keep = [c for c in ("open", "high", "low", "close", "volume", "amount") if c in df.columns]
        df = df[keep]
        if self._needs_resample:
            df = resample_1min_to_interval(df, interval)
        return df

    def fetch_full(self, symbol: str, interval: str) -> pd.DataFrame:
        raw_interval = "1m" if self._needs_resample else interval
        tf_sym = self._tf_symbol(symbol)
        df = self._tf.klines.get(tf_sym, period=raw_interval, count=10000, as_dataframe=True)
        return self._norm(df, interval)

    def fetch_incremental(self, symbol: str, interval: str, since: datetime) -> pd.DataFrame:
        raw_interval = "1m" if self._needs_resample else interval
        tf_sym = self._tf_symbol(symbol)
        since_ms = int(since.timestamp() * 1000)
        df = self._tf.klines.get(
            tf_sym, period=raw_interval, count=10000, start_time=since_ms, as_dataframe=True
        )
        return self._norm(df, interval)


def make_fetcher(cfg: DataConfig) -> BaseDataFetcher:
    if cfg.source == "tickflow":
        return TickflowFetcher(cfg)
    return LocalFileFetcher(cfg)


# ---------------------------------------------------------------------------
# Data validator
# ---------------------------------------------------------------------------

class DataQualityReport:
    def __init__(self) -> None:
        self.n_rows: int = 0
        self.time_range: Tuple[str, str] = ("", "")
        self.columns: List[str] = []
        self.duplicate_timestamps: int = 0
        self.missing_values: Dict[str, int] = {}
        self.anomalies: List[str] = []
        self.continuity_issues: int = 0
        self.new_rows: int = 0
        self.passed: bool = True
        self.fail_reasons: List[str] = []

    def to_dict(self) -> dict:
        return self.__dict__


def validate_ohlcv(df: pd.DataFrame, interval: str) -> DataQualityReport:
    """Run all quality checks and return a DataQualityReport."""
    report = DataQualityReport()
    report.n_rows = len(df)
    if len(df) == 0:
        report.passed = False
        report.fail_reasons.append("dataframe_empty")
        return report

    report.time_range = (str(df.index.min()), str(df.index.max()))
    report.columns = list(df.columns)

    # Duplicate timestamps
    report.duplicate_timestamps = df.index.duplicated().sum()
    if report.duplicate_timestamps > 0:
        report.anomalies.append(f"duplicate_timestamps={report.duplicate_timestamps}")

    # Missing values
    report.missing_values = df.isnull().sum().to_dict()
    total_nans = sum(report.missing_values.values())
    if total_nans > 0:
        report.anomalies.append(f"total_nan={total_nans}")

    # OHLC sanity
    if "high" in df.columns and "low" in df.columns:
        bad_hl = (df["high"] < df["low"]).sum()
        if bad_hl:
            report.anomalies.append(f"high_lt_low={bad_hl}")
    if "volume" in df.columns:
        neg_vol = (df["volume"] < 0).sum()
        if neg_vol:
            report.anomalies.append(f"negative_volume={neg_vol}")
    if all(c in df.columns for c in ("open", "high", "low", "close")):
        ohlc_bad = (
            (df["open"] > df["high"]) | (df["open"] < df["low"]) |
            (df["close"] > df["high"]) | (df["close"] < df["low"])
        ).sum()
        if ohlc_bad:
            report.anomalies.append(f"ohlc_inconsistent={ohlc_bad}")

    # Time continuity (business days, skip gaps between sessions)
    expected_bars = BARS_PER_DAY.get(interval, 8)
    dates = df.index.normalize().unique()
    gaps = 0
    for date in dates:
        day_bars = df[df.index.normalize() == date]
        # Allow up to 1 missing bar per day before flagging
        if len(day_bars) < expected_bars - 1:
            gaps += 1
    report.continuity_issues = gaps
    if gaps > 0:
        report.anomalies.append(f"incomplete_trading_days={gaps}")

    return report


# ---------------------------------------------------------------------------
# Data cleaner
# ---------------------------------------------------------------------------

def clean_ohlcv(df: pd.DataFrame, timezone: str = "Asia/Shanghai") -> pd.DataFrame:
    """Apply cleaning rules per doc 02_数据工程."""
    df = df.copy()

    # 1. Enforce timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone)
    else:
        df.index = df.index.tz_convert(timezone)

    # 2. Remove duplicates (keep first = trusted source order)
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    # 3. Remove rows where high < low (clearly erroneous)
    if "high" in df.columns and "low" in df.columns:
        bad = df["high"] < df["low"]
        if bad.any():
            log.warning(f"Dropping {bad.sum()} rows with high < low")
            df = df[~bad]

    # 4. Remove rows with negative volume
    if "volume" in df.columns:
        bad = df["volume"] < 0
        if bad.any():
            log.warning(f"Dropping {bad.sum()} rows with negative volume")
            df = df[~bad]

    # 5. Forward-fill small within-day gaps only
    #    Group by date and forward-fill within each day
    def _ffill_day(g: pd.DataFrame) -> pd.DataFrame:
        return g.ffill(limit=2)

    df = df.groupby(df.index.date, group_keys=False).apply(_ffill_day)

    return df


# ---------------------------------------------------------------------------
# Feature engineering  (ref: etf_t0_quant/ref/features.py)
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, cfg: FeatureConfig, interval: str) -> pd.DataFrame:
    """Compute all technical features.  Input must have lowercase OHLCV columns."""
    data = df.copy()
    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]

    bars_per_day = BARS_PER_DAY.get(interval, 8)

    # --- Returns ---
    data["returns"] = close.pct_change()
    data["log_ret"] = np.log(close / close.shift(1))
    data["amplitude"] = (high - low) / close.shift(1)

    # --- Volume z-score ---
    data["volume_ma"] = volume.rolling(cfg.volume_ma_period).mean()
    data["volume_ratio"] = (volume / data["volume_ma"]).clip(0, 3)

    # --- Close position within bar ---
    rng = high - low
    data["close_position"] = np.where(rng > 0, (close - low) / rng, 0.5)

    # --- RSI ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(cfg.rsi_period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(cfg.rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    data["rsi"] = (100 - (100 / (1 + rs))).fillna(50)
    data["rsi_norm_raw"] = data["rsi"] / 100  # 0–1, no future leakage

    # --- ATR ---
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    data["atr"] = tr.rolling(cfg.atr_period).mean()
    data["atr_ratio"] = (data["atr"] / close).clip(0, 0.1)

    # --- EMA diff (MACD core) ---
    ema_s = close.ewm(span=cfg.ema_short, adjust=False).mean()
    ema_l = close.ewm(span=cfg.ema_long, adjust=False).mean()
    data["ema_diff"] = ((ema_s - ema_l) / close).clip(-0.1, 0.1)

    # --- Bollinger Band width ---
    ma = close.rolling(cfg.bb_period).mean()
    std = close.rolling(cfg.bb_period).std()
    bb_upper = ma + cfg.bb_std * std
    bb_lower = ma - cfg.bb_std * std
    data["bb_width"] = ((bb_upper - bb_lower) / ma).clip(0, 0.2)

    # --- Annualised volatility (20-bar rolling) ---
    ann = 252 * bars_per_day
    data["vol20"] = data["log_ret"].rolling(20).std() * np.sqrt(ann)

    # --- Time features ---
    if hasattr(data.index, "hour"):
        hm = data.index.hour * 60 + data.index.minute
        # Map to 0-1 within session (09:30=570, 15:00=900)
        data["time_slot"] = (hm - 570) / (900 - 570)
        data["time_slot"] = data["time_slot"].clip(0, 1)
        # Is near close (last 2 bars)
        data["is_near_close"] = (hm >= (900 - 2 * (30 if interval == "30m" else 15))).astype(float)
    else:
        data["time_slot"] = 0.5
        data["is_near_close"] = 0.0

    return data


def normalize_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Rolling z-score normalization (no future leakage: shift(1) on stats).

    Ref: etf_t0_quant/ref/features.py smart_feature_normalization
    """
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

    # Volume ratio – robust (IQR-based)
    if "volume_ratio" in df.columns:
        med = df["volume_ratio"].rolling(w, min_periods=mp).median().shift(1)
        q75 = df["volume_ratio"].rolling(w, min_periods=mp).quantile(0.75).shift(1)
        q25 = df["volume_ratio"].rolling(w, min_periods=mp).quantile(0.25).shift(1)
        iqr = (q75 - q25).replace(0, np.nan)
        result["volume_ratio_norm"] = ((df["volume_ratio"] - med) / (iqr + eps)).clip(-3, 3)

    # RSI already 0-1; pass through zero-centred
    if "rsi_norm_raw" in df.columns:
        result["rsi_norm"] = df["rsi_norm_raw"] * 2 - 1  # → [-1, 1]

    # close_position, time_slot, is_near_close already bounded
    for col in ("close_position", "time_slot", "is_near_close"):
        if col in df.columns:
            result[col] = df[col]

    return result


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _compute_hash(df: pd.DataFrame) -> str:
    import hashlib
    h = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()


def build_metadata(
    df_raw: pd.DataFrame,
    df_feat: pd.DataFrame,
    cfg: AppConfig,
    run_id: str,
    source: str,
) -> dict:
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "symbol": cfg.base.symbol,
        "interval": cfg.base.interval,
        "source": source,
        "raw_rows": len(df_raw),
        "feature_rows": len(df_feat),
        "raw_time_range": [str(df_raw.index.min()), str(df_raw.index.max())],
        "feature_time_range": [str(df_feat.index.min()), str(df_feat.index.max())],
        "feature_columns": list(df_feat.columns),
        "data_hash": _compute_hash(df_raw),
        "adjust_note": "no_adjust_mv1",
    }


# ---------------------------------------------------------------------------
# DataPipeline – orchestrator
# ---------------------------------------------------------------------------

class DataPipeline:
    """Orchestrates the full data engineering pipeline."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.fetcher = make_fetcher(cfg.data)
        self._symbol = cfg.base.symbol
        self._interval = cfg.base.interval

        # Ensure output directories exist
        base = cfg.base
        for d in (
            base.data_dir / "raw" / self._symbol / self._interval,
            base.data_dir / "cleaned" / self._symbol / self._interval,
            base.data_dir / "features" / self._symbol / self._interval,
            base.reports_dir,
            base.metadata_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def run(self, mode: str = "full") -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Run the pipeline.

        Returns:
            df_features: normalised feature dataframe
            df_prices:   clean OHLCV (for env.py price replay)
            metadata:    dict with provenance info
        """
        run_id = get_run_id()
        log.info(f"DataPipeline.run mode={mode} symbol={self._symbol} interval={self._interval}")

        # ── 1. Fetch ──────────────────────────────────────────────────
        df_raw = self._fetch(mode)
        log.info(f"Fetched {len(df_raw)} raw bars")

        # ── 2. Save raw ───────────────────────────────────────────────
        raw_path = (
            self.cfg.base.data_dir / "raw" / self._symbol / self._interval / "data.parquet"
        )
        df_raw.to_parquet(raw_path)

        # ── 3. Validate ───────────────────────────────────────────────
        report = validate_ohlcv(df_raw, self._interval)
        report.new_rows = len(df_raw)
        report_path = self.cfg.base.reports_dir / f"{run_id}_quality.json"
        report_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))

        if not report.passed:
            msg = f"Data quality FAILED: {report.fail_reasons}"
            log.error(msg)
            raise RuntimeError(msg)

        log.info(f"Data quality passed. Anomalies: {report.anomalies}")

        # ── 4. Clean ──────────────────────────────────────────────────
        df_clean = clean_ohlcv(df_raw, self.cfg.base.timezone)
        clean_path = (
            self.cfg.base.data_dir / "cleaned" / self._symbol / self._interval / "data.parquet"
        )
        df_clean.to_parquet(clean_path)
        log.info(f"Cleaned data: {len(df_clean)} rows")

        # ── 5. Feature engineering ────────────────────────────────────
        df_feat_raw = build_features(df_clean, self.cfg.feature, self._interval)
        df_feat_norm = normalize_features(df_feat_raw, self.cfg.feature)

        # Drop NaN prefix rows (from rolling windows)
        feat_cols = [c for c in self.cfg.feature.feature_columns if c in df_feat_norm.columns]
        df_feat_norm = df_feat_norm.dropna(subset=feat_cols)
        # Align prices to features index
        df_prices_aligned = df_clean.loc[df_feat_norm.index]

        feat_path = (
            self.cfg.base.data_dir / "features" / self._symbol / self._interval / "data.parquet"
        )
        df_feat_norm.to_parquet(feat_path)
        log.info(f"Feature data: {len(df_feat_norm)} rows, {len(feat_cols)} feature columns")

        # ── 6. Metadata ───────────────────────────────────────────────
        meta = build_metadata(df_raw, df_feat_norm, self.cfg, run_id, self.cfg.data.source)
        meta_path = self.cfg.base.metadata_dir / f"{run_id}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

        return df_feat_norm[feat_cols], df_prices_aligned, meta

    # ------------------------------------------------------------------
    def load_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load pre-computed feature and price data from disk."""
        feat_path = (
            self.cfg.base.data_dir / "features" / self._symbol / self._interval / "data.parquet"
        )
        clean_path = (
            self.cfg.base.data_dir / "cleaned" / self._symbol / self._interval / "data.parquet"
        )
        if not feat_path.exists():
            raise FileNotFoundError(
                f"Feature file not found: {feat_path}. Run pipeline first."
            )
        df_feat = pd.read_parquet(feat_path)
        df_prices = pd.read_parquet(clean_path)
        feat_cols = [c for c in self.cfg.feature.feature_columns if c in df_feat.columns]
        df_prices = df_prices.loc[df_feat.index]
        return df_feat[feat_cols], df_prices

    # ------------------------------------------------------------------
    def _fetch(self, mode: str) -> pd.DataFrame:
        if mode == "full":
            return self.fetcher.fetch_full(self._symbol, self._interval)

        # Incremental: find latest timestamp in existing raw file
        raw_path = (
            self.cfg.base.data_dir / "raw" / self._symbol / self._interval / "data.parquet"
        )
        if raw_path.exists():
            existing = pd.read_parquet(raw_path)
            since = existing.index.max().to_pydatetime()
        else:
            since = datetime.now() - timedelta(days=self.cfg.data.incremental_window_days)

        new_df = self.fetcher.fetch_incremental(self._symbol, self._interval, since)
        if raw_path.exists() and len(new_df) > 0:
            existing = pd.read_parquet(raw_path)
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            return combined
        return new_df


# ---------------------------------------------------------------------------
# Convenience: generate synthetic data for testing
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    n_days: int = 300,
    interval: str = "30m",
    symbol: str = "TEST",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV bars for testing (no real market data needed)."""
    rng = np.random.default_rng(seed)
    bars_per_day = BARS_PER_DAY[interval]
    minutes = 30 if interval == "30m" else 15

    timestamps = []
    base = pd.Timestamp("2023-01-03 10:00:00", tz="Asia/Shanghai")
    day = 0
    while len(timestamps) < n_days * bars_per_day:
        ts = base + pd.Timedelta(days=day)
        if ts.weekday() >= 5:  # skip weekends
            day += 1
            continue
        for i in range(bars_per_day):
            timestamps.append(ts + pd.Timedelta(minutes=i * minutes))
        day += 1

    n = len(timestamps)
    log_returns = rng.normal(0.0002, 0.005, n)
    close = 1.0 * np.exp(np.cumsum(log_returns)) * 2.5  # start ~2.5

    open_ = close * (1 + rng.normal(0, 0.001, n))
    high = np.maximum(close, open_) * (1 + rng.uniform(0, 0.005, n))
    low = np.minimum(close, open_) * (1 - rng.uniform(0, 0.005, n))
    volume = rng.integers(1_000_000, 50_000_000, n).astype(float)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(timestamps[:n], name="timestamp"),
    )
    return df
