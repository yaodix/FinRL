"""
Signal Service (doc 07_实盘qmt交易 – signal_service).

Responsibilities:
  - Read the currently published model.
  - Collect the latest closed bar's features from the data pipeline.
  - Run model inference to produce a trading signal.
  - Emit the signal as a structured dict.

The signal is NOT executed here; it is passed to the risk_gateway for validation.

Usage::

    svc = SignalService(cfg)
    signal = svc.generate_signal(current_position_shares=0, current_cash=100000)
    print(signal)
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from etf_t0_quant.config import AppConfig
from etf_t0_quant.env import ACTION_NAMES, ETFTradingEnv
from etf_t0_quant.logger import get_logger, get_run_id

log = get_logger("live")


class SignalService:
    """Generate trading signals from a published model."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._model_version: str = ""
        self._data_version: str = ""

    # ------------------------------------------------------------------

    def load_published_model(self) -> None:
        """Load the currently published model from disk."""
        from etf_t0_quant.trainer import ETFTrainer

        pub_file = self.cfg.base.models_dir / "published_model.json"
        if not pub_file.exists():
            raise FileNotFoundError(
                "No published model found. Run: python -m etf_t0_quant publish <dir>"
            )
        pub = json.loads(pub_file.read_text())
        model_path = Path(pub["path"]) / "model"
        self._model = ETFTrainer.load_model(model_path)
        self._model_version = pub.get("version", "unknown")
        log.info(f"Loaded published model version={self._model_version}")

    def generate_signal(
        self,
        current_position_shares: float = 0.0,
        current_cash: Optional[float] = None,
        df_features: Optional[pd.DataFrame] = None,
        df_prices: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """Generate a trading signal for the next bar.

        Args:
            current_position_shares: currently held shares
            current_cash: current cash balance (defaults to env initial_cash)
            df_features: pre-loaded feature DataFrame (optional; loads from disk if None)
            df_prices:   aligned price DataFrame (optional)

        Returns:
            Signal dict conforming to the protocol in doc 07.
        """
        if self._model is None:
            self.load_published_model()

        if df_features is None or df_prices is None:
            from etf_t0_quant.data_pipeline import DataPipeline
            pipeline = DataPipeline(self.cfg)
            df_features, df_prices = pipeline.load_features()
            self._data_version = "latest"

        # Build env with recent window
        lw = self.cfg.feature.lookback_window
        df_f = df_features.iloc[-(lw + 10):]
        df_p = df_prices.iloc[-(lw + 10):]

        env = ETFTradingEnv(df_f, df_p, self.cfg.env, self.cfg.feature)
        # Manually set position state to match real account
        if current_cash is not None:
            env._cash = current_cash
        env._shares = current_position_shares

        obs, _ = env.reset()
        # Fast-forward to penultimate bar (last closed bar)
        while env._idx < env._end_idx - 1:
            action_ff, _ = self._model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action_ff))
            if terminated or truncated:
                break

        action, _ = self._model.predict(obs, deterministic=True)
        action = int(action)

        signal: Dict = {
            "timestamp": datetime.now().isoformat(),
            "symbol": self.cfg.base.symbol,
            "interval": self.cfg.base.interval,
            "action": ACTION_NAMES.get(action, str(action)),
            "action_code": action,
            "model_version": self._model_version,
            "data_version": self._data_version,
            "run_id": get_run_id(),
            "reason_code": "model_inference",
        }
        log.info(f"Signal generated: {signal['action']} for {signal['symbol']}")
        return signal
