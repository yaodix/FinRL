"""
Smoke tests – verify every module can be imported and instantiated.
These tests do NOT require real data or a GPU.
"""
import pytest
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Import checks
# ---------------------------------------------------------------------------

def test_config_import():
    from etf_t0_quant.config import AppConfig, load_config
    cfg = AppConfig()
    assert cfg.base.symbol == "159740"


def test_logger_import():
    from etf_t0_quant.logger import get_logger, new_run_id, bind_run_id
    run_id = new_run_id()
    bind_run_id(run_id)
    log = get_logger("data")
    log.info("smoke test")


def test_notifier_import():
    from etf_t0_quant.config import NotifierConfig
    from etf_t0_quant.notifier import FeishuNotifier
    n = FeishuNotifier(NotifierConfig(enabled=False))
    assert n.send("test", "hello") is False


def test_data_pipeline_import():
    from etf_t0_quant.data_pipeline import (
        generate_synthetic_data,
        validate_ohlcv,
        clean_ohlcv,
        build_features,
        normalize_features,
    )
    from etf_t0_quant.config import FeatureConfig
    df = generate_synthetic_data(n_days=30, interval="30m")
    assert len(df) > 0
    report = validate_ohlcv(df, "30m")
    assert report.n_rows == len(df)


def test_env_import():
    from etf_t0_quant.env import ETFTradingEnv, BUY_ALL, SELL_ALL, HOLD


def test_risk_import():
    from etf_t0_quant.config import RiskConfig
    from etf_t0_quant.risk import RiskManager
    rm = RiskManager(RiskConfig())
    result = rm.check_pre_trade(
        action=1, position_shares=0, cash=100000, nav=100000, price=2.5
    )
    assert result.allowed


def test_backtester_import():
    from etf_t0_quant.backtester import (
        compute_metrics,
        buy_and_hold_nav,
        Backtester,
        WalkForwardBacktester,
    )


def test_live_imports():
    from etf_t0_quant.live.signal_service.signal_service import SignalService
    from etf_t0_quant.live.risk_gateway.risk_gateway import RiskGateway
    from etf_t0_quant.live.qmt_gateway.qmt_gateway import QMTGateway


# ---------------------------------------------------------------------------
# Minimal functional checks
# ---------------------------------------------------------------------------

def test_synthetic_data_shape():
    from etf_t0_quant.data_pipeline import generate_synthetic_data
    df = generate_synthetic_data(n_days=50, interval="30m", seed=0)
    assert set(df.columns) == {"open", "high", "low", "close", "volume"}
    assert len(df) == 50 * 8


def test_feature_pipeline_no_nan():
    from etf_t0_quant.data_pipeline import (
        generate_synthetic_data,
        build_features,
        normalize_features,
        clean_ohlcv,
    )
    from etf_t0_quant.config import FeatureConfig

    cfg = FeatureConfig()
    df = generate_synthetic_data(n_days=60, interval="30m", seed=1)
    df_clean = clean_ohlcv(df)
    df_feat = build_features(df_clean, cfg, "30m")
    df_norm = normalize_features(df_feat, cfg)

    feat_cols = [c for c in cfg.feature_columns if c in df_norm.columns]
    df_valid = df_norm.dropna(subset=feat_cols)
    assert len(df_valid) > 0, "All rows dropped after normalization"
    assert not df_valid[feat_cols].isnull().any().any(), "NaN in features after dropna"


def test_env_episode_runs():
    from etf_t0_quant.data_pipeline import (
        generate_synthetic_data,
        build_features,
        normalize_features,
        clean_ohlcv,
    )
    from etf_t0_quant.env import ETFTradingEnv, BUY_ALL, SELL_ALL, HOLD
    from etf_t0_quant.config import AppConfig

    cfg = AppConfig()
    df = generate_synthetic_data(n_days=60, interval="30m", seed=2)
    df_clean = clean_ohlcv(df)
    df_feat_raw = build_features(df_clean, cfg.feature, "30m")
    df_norm = normalize_features(df_feat_raw, cfg.feature)
    feat_cols = [c for c in cfg.feature.feature_columns if c in df_norm.columns]
    df_norm = df_norm.dropna(subset=feat_cols)
    df_clean = df_clean.loc[df_norm.index]

    env = ETFTradingEnv(df_norm[feat_cols], df_clean, cfg.env, cfg.feature)
    obs, info = env.reset()
    assert obs.shape[0] == cfg.feature.lookback_window * len(feat_cols) + 4

    total_reward = 0.0
    done = False
    steps = 0
    while not done and steps < 50:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1

    assert steps > 0
    assert "nav" in info


def test_risk_gap_logic():
    from etf_t0_quant.config import RiskConfig
    from etf_t0_quant.risk import RiskManager
    from etf_t0_quant.env import BUY_ALL, HOLD

    rc = RiskConfig(gap_threshold=0.02, gap_action_down="hold")
    rm = RiskManager(rc)

    # Downward gap > 2%
    result = rm.check_market_conditions(
        action=BUY_ALL,
        next_open=98.0,
        prev_close=100.2,
        is_halted=False,
    )
    assert result.modified_action == HOLD


def test_backtest_metrics():
    from etf_t0_quant.backtester import compute_metrics

    nav = np.linspace(100000, 110000, 200).astype(np.float64)
    trade_log = pd.DataFrame({"action": ["buy_all", "sell_all"], "pnl": [1000.0, 500.0]})
    metrics = compute_metrics(nav, trade_log, bars_per_day=8)
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert metrics["cumulative_return"] == pytest.approx(0.1, abs=0.01)
