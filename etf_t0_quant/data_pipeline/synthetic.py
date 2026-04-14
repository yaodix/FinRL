"""Synthetic data generator for tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import BARS_PER_DAY


def generate_synthetic_data(
    n_days: int = 300,
    interval: str = "30m",
    symbol: str = "TEST",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV bars for testing (no real market data needed)."""
    del symbol
    rng = np.random.default_rng(seed)
    bars_per_day = BARS_PER_DAY[interval]
    minutes = 30 if interval == "30m" else 15

    timestamps = []
    base = pd.Timestamp("2023-01-03 10:00:00", tz="Asia/Shanghai")
    day = 0
    while len(timestamps) < n_days * bars_per_day:
        ts = base + pd.Timedelta(days=day)
        if ts.weekday() >= 5:
            day += 1
            continue
        for i in range(bars_per_day):
            timestamps.append(ts + pd.Timedelta(minutes=i * minutes))
        day += 1

    n = len(timestamps)
    log_returns = rng.normal(0.0002, 0.005, n)
    close = 1.0 * np.exp(np.cumsum(log_returns)) * 2.5

    open_ = close * (1 + rng.normal(0, 0.001, n))
    high = np.maximum(close, open_) * (1 + rng.uniform(0, 0.005, n))
    low = np.minimum(close, open_) * (1 - rng.uniform(0, 0.005, n))
    volume = rng.integers(1_000_000, 50_000_000, n).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(timestamps[:n], name="timestamp"),
    )
