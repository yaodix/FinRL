"""Metadata helpers for data pipeline outputs."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from ..config import AppConfig


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
