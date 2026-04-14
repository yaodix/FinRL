"""
Unit tests for the RiskManager.
Tests all 5 risk layers: data, model, pre-trade, market, runtime.
"""
import pytest
from datetime import datetime, date


@pytest.fixture
def rm():
    from etf_t0_quant.config import RiskConfig
    from etf_t0_quant.risk import RiskManager
    return RiskManager(
        RiskConfig(
            gap_threshold=0.02,
            max_daily_gap_hits=2,
            gap_action_down="hold",
            nav_drawdown_halt=0.05,
            max_consecutive_data_failures=3,
        )
    )


# ---------------------------------------------------------------------------
# Layer 1 – Data freshness
# ---------------------------------------------------------------------------

def test_data_fresh(rm):
    from datetime import timedelta
    now = datetime(2025, 1, 2, 10, 30)
    last_bar = datetime(2025, 1, 2, 10, 29)
    result = rm.check_data_freshness(last_bar, now)
    assert result.allowed


def test_data_stale(rm):
    from datetime import timedelta
    now = datetime(2025, 1, 2, 10, 30)
    last_bar = datetime(2025, 1, 2, 9, 0)  # 90 min stale
    result = rm.check_data_freshness(last_bar, now)
    assert not result.allowed
    assert result.level in ("WARNING", "ERROR", "CRITICAL")


def test_consecutive_data_failures(rm):
    rm._data_fail_streak = 0
    rm._halt_mode = False
    rm.register_data_failure()
    rm.register_data_failure()
    halted = rm.register_data_failure()
    assert halted
    assert rm._halt_mode


def test_data_success_resets_streak(rm):
    rm._data_fail_streak = 2
    rm.register_data_success()
    assert rm._data_fail_streak == 0


# ---------------------------------------------------------------------------
# Layer 2 – Model risk
# ---------------------------------------------------------------------------

def test_model_metadata_ok(rm):
    meta = {
        "run_id": "abc",
        "model_version": "v1",
        "data_hash": "xyz",
        "backtest_sharpe": 1.5,
    }
    result = rm.check_model_metadata(meta)
    assert result.allowed


def test_model_metadata_missing(rm):
    meta = {"run_id": "abc"}
    result = rm.check_model_metadata(meta)
    assert not result.allowed


def test_model_backtest_threshold_pass(rm):
    result = rm.check_model_backtest_threshold(
        sharpe=1.5, max_dd=0.10, profit_factor=1.5, n_trades=50
    )
    assert result.allowed


def test_model_backtest_threshold_fail_sharpe(rm):
    result = rm.check_model_backtest_threshold(
        sharpe=0.5, max_dd=0.10, profit_factor=1.5, n_trades=50
    )
    assert not result.allowed
    assert "sharpe" in result.reason.lower()


def test_model_backtest_threshold_fail_drawdown(rm):
    result = rm.check_model_backtest_threshold(
        sharpe=1.5, max_dd=0.20, profit_factor=1.5, n_trades=50
    )
    assert not result.allowed
    assert "dd" in result.reason.lower()


def test_model_backtest_threshold_fail_trades(rm):
    result = rm.check_model_backtest_threshold(
        sharpe=1.5, max_dd=0.05, profit_factor=1.5, n_trades=5
    )
    assert not result.allowed


# ---------------------------------------------------------------------------
# Layer 3 – Pre-trade
# ---------------------------------------------------------------------------

def test_pretrade_buy_ok(rm):
    from etf_t0_quant.env import BUY_ALL
    result = rm.check_pre_trade(
        action=BUY_ALL, position_shares=0, cash=100000, nav=100000, price=2.5
    )
    assert result.allowed


def test_pretrade_buy_no_cash(rm):
    from etf_t0_quant.env import BUY_ALL, HOLD
    result = rm.check_pre_trade(
        action=BUY_ALL, position_shares=0, cash=0, nav=0, price=2.5
    )
    assert not result.allowed or result.modified_action == HOLD


def test_pretrade_sell_no_position(rm):
    from etf_t0_quant.env import SELL_ALL, HOLD
    result = rm.check_pre_trade(
        action=SELL_ALL, position_shares=0, cash=100000, nav=100000
    )
    # Should allow but modify to HOLD
    assert result.modified_action == HOLD


def test_pretrade_illegal_action(rm):
    result = rm.check_pre_trade(
        action=99, position_shares=0, cash=100000, nav=100000
    )
    assert not result.allowed


def test_pretrade_halt_mode_blocks_all(rm):
    from etf_t0_quant.env import BUY_ALL, HOLD
    rm.enter_halt("test")
    result = rm.check_pre_trade(
        action=BUY_ALL, position_shares=0, cash=100000, nav=100000
    )
    assert not result.allowed
    assert result.level == "CRITICAL"
    rm.reset_halt()


# ---------------------------------------------------------------------------
# Layer 4 – Market conditions
# ---------------------------------------------------------------------------

def test_market_ok(rm):
    from etf_t0_quant.env import HOLD
    result = rm.check_market_conditions(
        action=HOLD, next_open=100.0, prev_close=100.0
    )
    assert result.allowed


def test_market_halted(rm):
    from etf_t0_quant.env import BUY_ALL, HOLD
    result = rm.check_market_conditions(
        action=BUY_ALL, next_open=100.0, prev_close=100.0, is_halted=True
    )
    assert not result.allowed


def test_downward_gap_buy_downgraded(rm):
    from etf_t0_quant.env import BUY_ALL, HOLD
    rm._today = None  # reset counters
    rm._gap_hits_today = 0
    result = rm.check_market_conditions(
        action=BUY_ALL,
        next_open=96.0,   # -4% gap
        prev_close=100.0,
    )
    assert result.allowed  # still allowed (just first hit)
    assert result.modified_action == HOLD


def test_downward_gap_repeated_triggers_protect(rm):
    from etf_t0_quant.env import BUY_ALL, HOLD
    rm._today = date.today()
    rm._gap_hits_today = 1  # already one hit
    rm._protect_mode = False

    result = rm.check_market_conditions(
        action=BUY_ALL,
        next_open=95.0,   # another -5% gap
        prev_close=100.0,
        current_date=date.today(),
    )
    assert rm._protect_mode is True


def test_upward_gap_allowed(rm):
    from etf_t0_quant.env import BUY_ALL
    rm._protect_mode = False
    rm._halt_mode = False
    result = rm.check_market_conditions(
        action=BUY_ALL,
        next_open=103.0,  # +3% gap
        prev_close=100.0,
    )
    assert result.allowed
    assert result.modified_action == BUY_ALL


# ---------------------------------------------------------------------------
# Layer 5 – NAV drawdown halt
# ---------------------------------------------------------------------------

def test_nav_drawdown_halt(rm):
    rm.reset_halt()
    rm._session_high_nav = 100000
    rm._halt_mode = False
    # Drop 6% should trigger
    result = rm.update_nav(94000)
    assert result is not None
    assert not result.allowed
    assert rm._halt_mode
    rm.reset_halt()


def test_nav_small_drawdown_ok(rm):
    rm.reset_halt()
    rm._session_high_nav = 100000
    result = rm.update_nav(98000)  # 2% drawdown, below 5% threshold
    assert result is None  # no halt


# ---------------------------------------------------------------------------
# Daily reset
# ---------------------------------------------------------------------------

def test_gap_hits_reset_on_new_day(rm):
    rm._today = date(2025, 1, 1)
    rm._gap_hits_today = 5
    rm._protect_mode = True

    rm._refresh_day(date(2025, 1, 2))
    assert rm._gap_hits_today == 0
    assert rm._protect_mode is False
