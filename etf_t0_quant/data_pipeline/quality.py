"""Data quality report and validators."""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from .constants import BARS_PER_DAY


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

    report.duplicate_timestamps = df.index.duplicated().sum()
    if report.duplicate_timestamps > 0:
        report.anomalies.append(f"duplicate_timestamps={report.duplicate_timestamps}")

    report.missing_values = df.isnull().sum().to_dict()
    total_nans = sum(report.missing_values.values())
    if total_nans > 0:
        report.anomalies.append(f"total_nan={total_nans}")

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
            (df["open"] > df["high"])
            | (df["open"] < df["low"])
            | (df["close"] > df["high"])
            | (df["close"] < df["low"])
        ).sum()
        if ohlc_bad:
            report.anomalies.append(f"ohlc_inconsistent={ohlc_bad}")

    expected_bars = BARS_PER_DAY.get(interval, 8)
    dates = df.index.normalize().unique()
    gaps = 0
    for date in dates:
        day_bars = df[df.index.normalize() == date]
        if len(day_bars) < expected_bars - 1:
            gaps += 1
    report.continuity_issues = gaps
    if gaps > 0:
        report.anomalies.append(f"incomplete_trading_days={gaps}")

    return report
