"""Resample 1-minute bars to target intervals."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .constants import _TRADE_END, _TRADE_START


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
    """Resample 1-minute OHLCV data to 15m or 30m bars."""
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
