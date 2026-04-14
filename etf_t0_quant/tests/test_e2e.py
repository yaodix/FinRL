"""
End-to-end integration test: synthetic data → feature pipeline →
environment episode → backtest metrics.

This test does NOT train a real model; it uses a random policy to verify
the full data ↔ env ↔ backtester chain works without crashes and that
outputs are structurally correct.
"""
import numpy as np
import pandas as pd
import pytest


class _RandomModel:
    """Minimal stub that imitates stable-baselines3 model.predict()."""

    def __init__(self, n_actions: int = 3, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._n = n_actions

    def predict(self, obs, deterministic=False):
        return int(self._rng.integers(0, self._n)), None


@pytest.fixture(scope="module")
def synth_data():
    from etf_t0_quant.data_pipeline import (
        generate_synthetic_data,
        build_features,
        normalize_features,
        clean_ohlcv,
    )
    from etf_t0_quant.config import AppConfig

    cfg = AppConfig()
    df_raw = generate_synthetic_data(n_days=200, interval="30m", seed=42)
    df_clean = clean_ohlcv(df_raw)
    df_feat_raw = build_features(df_clean, cfg.feature, "30m")
    df_norm = normalize_features(df_feat_raw, cfg.feature)
    feat_cols = [c for c in cfg.feature.feature_columns if c in df_norm.columns]
    df_norm = df_norm.dropna(subset=feat_cols)
    df_clean = df_clean.loc[df_norm.index]
    return cfg, df_norm[feat_cols], df_clean


def test_e2e_env_full_episode(synth_data):
    """Full episode using random policy – verifies env correctness."""
    from etf_t0_quant.env import ETFTradingEnv

    cfg, df_feat, df_prices = synth_data
    model = _RandomModel(seed=42)

    env = ETFTradingEnv(df_feat, df_prices, cfg.env, cfg.feature)
    obs, info = env.reset()

    nav_list = [info["nav"]]
    done = False
    n_steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        nav_list.append(info["nav"])
        done = terminated or truncated
        n_steps += 1

    # Sanity checks
    assert n_steps > 0, "Zero steps taken"
    assert all(v > 0 for v in nav_list), "NAV went negative"
    trade_log = env.get_trade_log()
    assert isinstance(trade_log, pd.DataFrame)


def test_e2e_backtester_random_policy(synth_data):
    """Backtester with random policy produces valid BacktestResult."""
    from etf_t0_quant.backtester import Backtester, BacktestResult

    cfg, df_feat, df_prices = synth_data
    model = _RandomModel(seed=7)

    n = len(df_feat)
    split = int(n * 0.8)
    df_f = df_feat.iloc[split:]
    df_p = df_prices.iloc[split:]

    backtester = Backtester(cfg)
    result = backtester.run(model, df_f, df_p, window_label="e2e_random")

    assert isinstance(result, BacktestResult)
    assert len(result.nav_series) > 1
    assert "sharpe_ratio" in result.metrics
    assert "max_drawdown" in result.metrics
    assert result.metrics["max_drawdown"] >= 0
    assert result.metrics["max_drawdown"] <= 1


def test_e2e_walk_forward_random_policy(synth_data, tmp_path):
    """Walk-Forward with tiny window sizes to test the loop logic."""
    from etf_t0_quant.backtester import WalkForwardBacktester
    from etf_t0_quant.config import AppConfig, TrainConfig

    cfg, df_feat, df_prices = synth_data

    # Use tiny WF windows so tests run fast
    import copy
    cfg2 = AppConfig(
        base=cfg.base,
        feature=cfg.feature,
        env=cfg.env,
        backtest=cfg.backtest,
        risk=cfg.risk,
        train=TrainConfig(
            train_days=10,
            val_days=5,
            test_days=5,
            step_days=5,
            total_timesteps=100,
        ),
    )
    cfg2.base = cfg.base

    model = _RandomModel(seed=99)

    def trainer_fn(df_f, df_p):
        return model  # just return the same random model

    wf = WalkForwardBacktester(cfg2)
    results = wf.run(trainer_fn, df_feat, df_prices, out_dir=tmp_path / "wf")

    assert results.n_total > 0, "No WF windows completed"
    assert isinstance(results.publish_ok, bool)


def test_e2e_risk_gateway_integration(synth_data):
    """Risk gateway should allow a clean BUY signal and block an invalid one."""
    from etf_t0_quant.live.risk_gateway.risk_gateway import RiskGateway
    from etf_t0_quant.env import BUY_ALL, HOLD

    cfg, df_feat, df_prices = synth_data
    gw = RiskGateway(cfg)

    valid_signal = {
        "timestamp": "2025-01-01T10:00:00",
        "symbol": cfg.base.symbol,
        "interval": cfg.base.interval,
        "action": "buy_all",
        "action_code": BUY_ALL,
        "model_version": "v1",
        "run_id": "abc123",
    }
    account = {"position_shares": 0, "cash": 100000, "nav": 100000, "current_price": 2.5}
    result = gw.validate(valid_signal, account)
    assert result.allowed, f"Expected allowed, got: {result.reason}"

    # Missing required field → should fail
    bad_signal = dict(valid_signal)
    del bad_signal["model_version"]
    result2 = gw.validate(bad_signal, account)
    assert not result2.allowed


def test_e2e_qmt_simulation(synth_data):
    """QMT gateway in simulation mode should not raise."""
    from etf_t0_quant.live.qmt_gateway.qmt_gateway import QMTGateway
    from etf_t0_quant.env import BUY_ALL

    cfg, _, _ = synth_data
    gw = QMTGateway(cfg, simulation=True)
    gw.connect()

    state = gw.get_account_state()
    assert state.cash == cfg.env.initial_cash

    signal = {
        "symbol": cfg.base.symbol,
        "action": "buy_all",
        "action_code": BUY_ALL,
    }
    order = gw.place_order(signal)
    assert order.status.startswith("sim")

    gw.disconnect()
