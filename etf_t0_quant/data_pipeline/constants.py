"""Shared constants for data pipeline."""
from __future__ import annotations

from typing import Dict

import pandas as pd

BARS_PER_DAY: Dict[str, int] = {"30m": 8, "15m": 16}

# A-share ETF trading time: 09:30-11:30 and 13:00-15:00
_TRADE_START = pd.Timestamp("09:30:00").time()
_TRADE_END = pd.Timestamp("15:00:00").time()
