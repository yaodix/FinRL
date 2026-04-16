"""Data cleaning utilities for OHLCV bars."""
from __future__ import annotations

import pandas as pd

from ..logger import get_logger

log = get_logger("data")


def clean_ohlcv(df_in: pd.DataFrame, timezone: str = "Asia/Shanghai") -> pd.DataFrame:
    """Apply cleaning rules for OHLCV data."""
    df = df_in.copy()

    # if df.index.tz is None:
    #     df.index = df.index.tz_localize(timezone)
    # else:
    #     df.index = df.index.tz_convert(timezone)

    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    if "high" in df.columns and "low" in df.columns:
        bad = df["high"] < df["low"]
        if bad.any():
            log.warning(f"Dropping {bad.sum()} rows with high < low")
            df = df[~bad]

    if "volume" in df.columns:
        bad = df["volume"] < 0
        if bad.any():
            log.warning(f"Dropping {bad.sum()} rows with negative volume")
            df = df[~bad]

    def _ffill_day(g: pd.DataFrame) -> pd.DataFrame:
        '''
        对每一天的数据进行前向填充，最多填充2行，超过则认为是连续性问题
        '''        
        return g.ffill(limit=2)
    
    # df = df.groupby(df.index.date, group_keys=False).apply(_ffill_day)
    
    # 对比clean前后数据，统计填充的行数和连续性问题的行数
    

    return df
