"""Data source abstractions and concrete fetchers."""
from __future__ import annotations

import os
import zipfile
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd

from ..config import DataConfig
from .resample import resample_1min_to_interval

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
    """Fetch data via the tickflow Python SDK."""

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
        """Add market suffix if missing. e.g. '159740' -> '159740.SZ'."""
        if "." in symbol:
            return symbol
        code = symbol.strip()
        if code[:2] in ("60", "68", "51", "50", "56", "58"):
            return f"{code}.SH"
        return f"{code}.SZ"

    def _norm(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Normalize tickflow DataFrame to DatetimeIndex OHLCV frame."""
        df = df.copy()
        df.columns = df.columns.str.lower()
        if "trade_time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["trade_time"])
        elif "timestamp" in df.columns:
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
            tf_sym,
            period=raw_interval,
            count=10000,
            start_time=since_ms,
            as_dataframe=True,
        )
        return self._norm(df, interval)


def make_fetcher(cfg: DataConfig) -> BaseDataFetcher:
    if cfg.source == "tickflow":
        return TickflowFetcher(cfg)
    return LocalFileFetcher(cfg)
