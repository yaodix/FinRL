"""
Configuration module for ETF T0 Quant System.

All parameters are validated by Pydantic and loaded from YAML.
Three usage patterns:
    1. load_config()               – loads etf_t0_quant/configs/default.yaml
    2. load_config("path/x.yaml") – loads a custom YAML file
    3. AppConfig()                 – pure in-code defaults (good for tests)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, Optional
from  dotenv import load_dotenv
import yaml
from pydantic import BaseModel, Field

load_dotenv()  # Load .env file for environment variables (e.g. API keys)

# ─────────────────────────── sub-configs ─────────────────────────────

class BaseConfig(BaseModel):
    symbol: str = "159740.SZ"
    interval: Literal["15m", "30m"] = "30m"
    timezone: str = "Asia/Shanghai"
    base_dir: Path = Path(__file__).resolve().parent.parent / "etf_t0_quant"  # 改为工程目录+/etf_t0_quant

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "workdata"

    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"

    @property
    def backtests_dir(self) -> Path:
        return self.base_dir / "backtests"

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    @property
    def reports_dir(self) -> Path:
        return self.base_dir / "workdata" / "reports"

    @property
    def metadata_dir(self) -> Path:
        return self.base_dir / "workdata" / "metadata"


class DataConfig(BaseModel):
    source: Literal["tickflow", "local"] = "local"
    tickflow_API_KEY: Optional[str] = os.getenv("TICKFLOW_TOKEN")
    # Path to a local CSV/Parquet file or a directory of files (source=local)
    local_csv_path: Optional[str|Path] = None
    # Optional directory for persisting raw TickFlow responses as parquet
    raw_cache_dir: Optional[str|Path] = None


class FeatureConfig(BaseModel):
    lookback_window: int = 30
    rsi_period: int = 14
    atr_period: int = 14
    ema_short: int = 12
    ema_long: int = 26
    ma_period: int = 20
    bb_period: int = 20
    bb_std: float = 2.0
    volume_ma_period: int = 20
    # Rolling normalization window (bars). Default ≈ 22 days × 8 bars/day for 30m.
    normalization_window: int = 176
    normalization_min_periods: int = 30
    # Ordered feature column names (after normalization)
    feature_columns: List[str] = Field(default_factory=lambda: [
        "returns_norm",
        "amplitude_norm",
        "volume_ratio_norm",
        "close_position",
        "rsi_norm",
        "atr_ratio_norm",
        "ema_diff_norm",
        "bb_width_norm",
        "vol20_norm",
        "time_slot",
        "is_near_close",
    ])


class EnvConfig(BaseModel):
    initial_cash: float = 100_000.0
    fee_rate_buy: float = 0.00005   # 万 0.5 双边佣金
    fee_rate_sell: float = 0.00005
    min_fee: float = 5.0            # 最低佣金 5 元
    lot_size: int = 100             # 最小交易单位 100 份
    reward_mode: Literal["nav_log", "trade_pnl"] = "nav_log"
    allow_overnight: bool = True
    # Explicit turnover penalty subtracted from reward on every trade (buy or sell).
    # Discourages churning caused by noisy 1-bar signals.
    # Set to ~3x the expected single-bar fee cost so the agent prefers HOLD.
    turnover_penalty: float = 0.0003


class TrainConfig(BaseModel):
    model_type: Literal["dqn", "double_dqn"] = "dqn"
    total_timesteps: int = 200_000
    learning_rate: float = 1e-4
    batch_size: int = 64
    gamma: float = 0.99
    replay_buffer_size: int = 50_000
    target_update_interval: int = 1_000
    exploration_fraction: float = 0.3
    exploration_final_eps: float = 0.05
    net_arch: List[int] = Field(default_factory=lambda: [256, 256])
    seed: int = 42
    # Walk-Forward windows (in trading days)
    train_days: int = 120
    val_days: int = 20
    test_days: int = 20
    step_days: int = 20
    # Optuna
    n_trials: int = 20
    optuna_timeout: int = 3_600   # seconds


class BacktestConfig(BaseModel):
    slippage_rate: float = 0.0
    # Minimum-pass thresholds for a window to be considered "passing"
    min_trades: int = 30
    sharpe_threshold: float = 1.0
    max_drawdown_threshold: float = 0.15
    profit_factor_threshold: float = 1.10
    # Fraction of WF windows that must pass to publish model
    wf_pass_ratio: float = 2 / 3


class RiskConfig(BaseModel):
    max_position_ratio: float = 1.0   # No leverage
    gap_threshold: float = 0.02       # 2% gap triggers market risk
    max_daily_gap_hits: int = 2       # After N gaps → protect mode
    data_delay_bars: int = 2          # Bars allowed before declaring stale
    max_consecutive_data_failures: int = 3
    # Intraday NAV drop threshold → halt (live / final backtest)
    nav_drawdown_halt: float = 0.05
    # Intraday NAV drop threshold used during HPO / training evaluation.
    # Set high so early random exploration doesn't trigger halt and mask valid signals.
    nav_drawdown_halt_train: float = 0.20
    # Action on downward gap: "hold" or "sell_all"
    gap_action_down: Literal["hold", "sell_all"] = "hold"


class NotifierConfig(BaseModel):
    enabled: bool = False
    feishu_webhook: Optional[str] = None


class AppConfig(BaseModel):
    """Root configuration object."""
    base: BaseConfig = Field(default_factory=BaseConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    feature: FeatureConfig = Field(default_factory=FeatureConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    notifier: NotifierConfig = Field(default_factory=NotifierConfig)


# ─────────────────────────── loader ──────────────────────────────────

def load_config(path: str | Path | None = None) -> AppConfig:
    """Load AppConfig from a YAML file.

    Falls back to etf_t0_quant/configs/default.yaml if *path* is None.
    Returns pure defaults if no file is found.
    """
    if path is None:
        default_path = Path(__file__).parent / "configs" / "default.yaml"
        if default_path.exists():
            path = default_path
        else:
            return AppConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return AppConfig.model_validate(raw)
