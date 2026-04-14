"""
Unit tests for the ETFTradingEnv.
Tests: state shape, action execution, fee calculation, reward, terminal condition.
"""
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_env():
    """Minimal env for deterministic tests."""
    from etf_t0_quant.data_pipeline import (
        generate_synthetic_data, build_features, normalize_features, clean_ohlcv,
    )
    from etf_t0_quant.env import ETFTradingEnv
    from etf_t0_quant.config import AppConfig

    cfg = AppConfig()
    df_raw = generate_synthetic_data(n_days=80, interval="30m", seed=99)
    df_c = clean_ohlcv(df_raw)
    df_feat_raw = build_features(df_c, cfg.feature, "30m")
    df_norm = normalize_features(df_feat_raw, cfg.feature)
    feat_cols = [c for c in cfg.feature.feature_columns if c in df_norm.columns]
    df_norm = df_norm.dropna(subset=feat_cols)
    df_c = df_c.loc[df_norm.index]

    env = ETFTradingEnv(df_norm[feat_cols], df_c, cfg.env, cfg.feature)
    return env, cfg


# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------

def test_obs_shape(small_env):
    env, cfg = small_env
    obs, _ = env.reset()
    expected_size = cfg.feature.lookback_window * env._n_feat + 4
    assert obs.shape == (expected_size,)


def test_obs_dtype(small_env):
    env, _ = small_env
    obs, _ = env.reset()
    assert obs.dtype == np.float32


def test_obs_no_nan_after_reset(small_env):
    env, _ = small_env
    obs, _ = env.reset()
    assert not np.isnan(obs).any(), "NaN in initial observation"


# ---------------------------------------------------------------------------
# Action execution
# ---------------------------------------------------------------------------

def test_hold_action_no_state_change(small_env):
    from etf_t0_quant.env import HOLD
    env, _ = small_env
    env.reset()
    cash_before = env._cash
    shares_before = env._shares
    obs, reward, terminated, truncated, info = env.step(HOLD)
    # Cash and shares remain the same after hold
    assert env._cash == pytest.approx(cash_before)
    assert env._shares == pytest.approx(shares_before)


def test_buy_then_sell_fees_are_charged(small_env):
    """Fees must be deducted: verify trade log records positive fee values."""
    from etf_t0_quant.env import BUY_ALL, SELL_ALL
    env, cfg = small_env
    env.reset()

    env.step(BUY_ALL)
    env.step(SELL_ALL)

    log = env.get_trade_log()
    assert len(log) >= 2, "Expected at least buy and sell entries"
    fees = log["fee"].tolist()
    assert all(f >= cfg.env.min_fee for f in fees), "All fees must be >= min_fee"


def test_buy_with_zero_cash():
    """If cash is 0, buying should be a no-op (hold)."""
    from etf_t0_quant.data_pipeline import generate_synthetic_data, build_features, normalize_features, clean_ohlcv
    from etf_t0_quant.env import ETFTradingEnv, BUY_ALL
    from etf_t0_quant.config import AppConfig, EnvConfig

    cfg = AppConfig()
    df_raw = generate_synthetic_data(n_days=50, interval="30m", seed=5)
    df_c = clean_ohlcv(df_raw)
    df_f = build_features(df_c, cfg.feature, "30m")
    df_n = normalize_features(df_f, cfg.feature)
    feat_cols = [c for c in cfg.feature.feature_columns if c in df_n.columns]
    df_n = df_n.dropna(subset=feat_cols)
    df_c = df_c.loc[df_n.index]

    env_cfg = EnvConfig(initial_cash=0.0)
    env = ETFTradingEnv(df_n[feat_cols], df_c, env_cfg, cfg.feature)
    env.reset()
    shares_before = env._shares
    env.step(BUY_ALL)
    assert env._shares == pytest.approx(shares_before), "Should not buy with zero cash"


def test_sell_with_no_position(small_env):
    """Selling when empty should simply hold."""
    from etf_t0_quant.env import SELL_ALL
    env, _ = small_env
    env.reset()
    assert env._shares == 0
    cash_before = env._cash
    env.step(SELL_ALL)
    assert env._cash == pytest.approx(cash_before)


def test_lot_size_constraint(small_env):
    """Bought shares must be a multiple of lot_size."""
    from etf_t0_quant.env import BUY_ALL
    env, cfg = small_env
    env.reset()
    env.step(BUY_ALL)
    lot = cfg.env.lot_size
    assert env._shares % lot == 0, f"Shares {env._shares} not multiple of lot_size {lot}"


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def test_reward_is_log_nav_change(small_env):
    from etf_t0_quant.env import HOLD
    env, cfg = small_env
    env.reset()
    nav_before = env._nav()
    _, reward, _, _, info = env.step(HOLD)
    nav_after = info["nav"]
    expected = float(np.log(nav_after / nav_before)) if nav_before > 0 and nav_after > 0 else 0.0
    assert reward == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# Terminal condition
# ---------------------------------------------------------------------------

def test_episode_terminates(small_env):
    from etf_t0_quant.env import HOLD
    env, _ = small_env
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, terminated, truncated, _ = env.step(HOLD)
        done = terminated or truncated
        steps += 1
        if steps > 10_000:
            pytest.fail("Episode did not terminate")
    assert done


# ---------------------------------------------------------------------------
# Trade log
# ---------------------------------------------------------------------------

def test_trade_log_structure(small_env):
    from etf_t0_quant.env import BUY_ALL, SELL_ALL, HOLD
    env, _ = small_env
    env.reset()
    env.step(BUY_ALL)
    env.step(HOLD)
    env.step(SELL_ALL)
    log = env.get_trade_log()
    assert isinstance(log, pd.DataFrame)
    assert len(log) >= 2  # at least buy + sell
    assert "action" in log.columns
    assert "price" in log.columns
    assert "fee" in log.columns


# ---------------------------------------------------------------------------
# Fee calculation
# ---------------------------------------------------------------------------

def test_minimum_fee_applied():
    """When trade is very small, minimum fee (5 CNY) should be used."""
    from etf_t0_quant.data_pipeline import generate_synthetic_data, build_features, normalize_features, clean_ohlcv
    from etf_t0_quant.env import ETFTradingEnv, BUY_ALL
    from etf_t0_quant.config import AppConfig, EnvConfig

    cfg = AppConfig()
    df_raw = generate_synthetic_data(n_days=40, interval="30m", seed=55)
    df_c = clean_ohlcv(df_raw)
    df_f = build_features(df_c, cfg.feature, "30m")
    df_n = normalize_features(df_f, cfg.feature)
    feat_cols = [c for c in cfg.feature.feature_columns if c in df_n.columns]
    df_n = df_n.dropna(subset=feat_cols)
    df_c = df_c.loc[df_n.index]

    # Very small cash so fee hits minimum
    env_cfg = EnvConfig(initial_cash=1000.0, min_fee=5.0, fee_rate_buy=0.00005)
    env = ETFTradingEnv(df_n[feat_cols], df_c, env_cfg, cfg.feature)
    env.reset()
    env.step(BUY_ALL)

    if len(env._trade_log) > 0:
        fee = env._trade_log[0]["fee"]
        assert fee >= env_cfg.min_fee, f"Fee {fee} < min_fee {env_cfg.min_fee}"
