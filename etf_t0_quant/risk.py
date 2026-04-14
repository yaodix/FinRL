"""
Risk management module (doc 06_风控).

Five layers:
  1. Data risk     – freshness, field validation
  2. Model risk    – NaN loss, metadata completeness, backtest threshold
  3. Pre-trade     – position limits, cash check, action legality
  4. Market risk   – halt detection, gap control
  5. Runtime risk  – dependency health, file-write failures

Usage (inference / backtest)::

    rm = RiskManager(cfg.risk, symbol="159740", interval="30m")
    result = rm.check_pre_trade(action=1, position=0.0, cash=100000, nav=100000)
    if not result.allowed:
        action = result.modified_action
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

from .config import RiskConfig
from .logger import get_logger

log = get_logger("risk")

HOLD = 0
BUY_ALL = 1
SELL_ALL = 2
LEGAL_ACTIONS = {HOLD, BUY_ALL, SELL_ALL}

_GAP_ACTION_MAP = {"hold": HOLD, "sell_all": SELL_ALL}


@dataclass
class RiskCheckResult:
    allowed: bool
    modified_action: int  # may be same as original if allowed=True
    reason: str = ""
    level: str = "INFO"   # INFO / WARNING / ERROR / CRITICAL


class RiskManager:
    """Stateful risk manager.  One instance per trading session (or backtest run)."""

    def __init__(
        self,
        cfg: RiskConfig,
        symbol: str = "",
        interval: str = "30m",
        training_mode: bool = False,
    ) -> None:
        self.cfg = cfg
        self.symbol = symbol
        self.interval = interval
        # When True, use nav_drawdown_halt_train (higher) instead of nav_drawdown_halt.
        # Set during HPO / training evaluation to avoid early-stop masking the signal.
        self._training_mode = training_mode

        # Intraday counters (reset each day)
        self._today: Optional[date] = None
        self._gap_hits_today: int = 0
        self._data_fail_streak: int = 0
        self._protect_mode: bool = False
        self._halt_mode: bool = False

        # NAV watermarks
        self._session_high_nav: float = 0.0
        self._initial_nav: float = 0.0

    # ------------------------------------------------------------------
    # Layer 1 – Data risk
    # ------------------------------------------------------------------

    def check_data_freshness(
        self, last_bar_time: datetime, now: Optional[datetime] = None
    ) -> RiskCheckResult:
        """Check whether the latest bar is fresh enough."""
        if now is None:
            now = datetime.now()
        mins = {"30m": 30, "15m": 15}.get(self.interval, 30)
        delay_limit = mins * self.cfg.data_delay_bars
        diff_minutes = (now - last_bar_time).total_seconds() / 60
        if diff_minutes > delay_limit:
            msg = f"Data stale: last_bar={last_bar_time}, diff={diff_minutes:.1f}min > {delay_limit}min"
            log.warning(msg)
            return RiskCheckResult(False, HOLD, msg, "WARNING")
        return RiskCheckResult(True, HOLD, "data_fresh", "INFO")

    def register_data_failure(self) -> bool:
        """Call when a data fetch fails. Returns True if hard stop triggered."""
        self._data_fail_streak += 1
        if self._data_fail_streak >= self.cfg.max_consecutive_data_failures:
            msg = f"Consecutive data failures={self._data_fail_streak} >= threshold"
            log.critical(msg)
            self._halt_mode = True
            return True
        log.warning(f"Data failure #{self._data_fail_streak}")
        return False

    def register_data_success(self) -> None:
        self._data_fail_streak = 0

    # ------------------------------------------------------------------
    # Layer 2 – Model risk
    # ------------------------------------------------------------------

    def check_model_metadata(self, metadata: dict) -> RiskCheckResult:
        """Validate that a model has the required provenance fields."""
        required = {"run_id", "model_version", "data_hash", "backtest_sharpe"}
        missing = required - set(metadata.keys())
        if missing:
            msg = f"Model metadata missing fields: {missing}"
            log.error(msg)
            return RiskCheckResult(False, HOLD, msg, "ERROR")
        return RiskCheckResult(True, HOLD, "model_metadata_ok", "INFO")

    def check_model_backtest_threshold(
        self,
        sharpe: float,
        max_dd: float,
        profit_factor: float,
        n_trades: int,
        sharpe_thr: float = 1.0,
        dd_thr: float = 0.15,
        pf_thr: float = 1.10,
        min_trades: int = 30,
    ) -> RiskCheckResult:
        fails = []
        if sharpe < sharpe_thr:
            fails.append(f"sharpe={sharpe:.3f}<{sharpe_thr}")
        if max_dd > dd_thr:
            fails.append(f"max_dd={max_dd:.3f}>{dd_thr}")
        if profit_factor < pf_thr:
            fails.append(f"pf={profit_factor:.3f}<{pf_thr}")
        if n_trades < min_trades:
            fails.append(f"n_trades={n_trades}<{min_trades}")
        if fails:
            msg = "Model does not meet backtest threshold: " + ", ".join(fails)
            log.error(msg)
            return RiskCheckResult(False, HOLD, msg, "ERROR")
        log.info("Model passed backtest threshold")
        return RiskCheckResult(True, HOLD, "threshold_ok", "INFO")

    # ------------------------------------------------------------------
    # Layer 3 – Pre-trade
    # ------------------------------------------------------------------

    def check_pre_trade(
        self,
        action: int,
        position_shares: float,
        cash: float,
        nav: float,
        price: float = 0.0,
        lot_size: int = 100,
    ) -> RiskCheckResult:
        """Validate a proposed action before execution."""
        if self._halt_mode:
            return RiskCheckResult(False, HOLD, "halt_mode_active", "CRITICAL")
        if self._protect_mode:
            return RiskCheckResult(True, HOLD, "protect_mode_signal_only", "WARNING")

        # Illegal action
        if action not in LEGAL_ACTIONS:
            msg = f"Illegal action={action}"
            log.error(msg)
            return RiskCheckResult(False, HOLD, msg, "ERROR")

        # Cash not sufficient for buy
        if action == BUY_ALL:
            if cash <= 0:
                return RiskCheckResult(False, HOLD, "cash_insufficient", "WARNING")
            if price > 0 and cash < price * lot_size:
                return RiskCheckResult(False, HOLD, "cash_lt_one_lot", "WARNING")

        # Can't sell if no position
        if action == SELL_ALL and position_shares <= 0:
            return RiskCheckResult(True, HOLD, "no_position_to_sell", "INFO")

        # No leveraged position allowed
        if nav > 0 and position_shares * (price if price > 0 else 1) / nav > self.cfg.max_position_ratio + 0.01:
            msg = f"Position ratio exceeds limit={self.cfg.max_position_ratio}"
            log.warning(msg)
            return RiskCheckResult(False, HOLD, msg, "WARNING")

        return RiskCheckResult(True, action, "pre_trade_ok", "INFO")

    # ------------------------------------------------------------------
    # Layer 4 – Market
    # ------------------------------------------------------------------

    def check_market_conditions(
        self,
        action: int,
        next_open: float,
        prev_close: float,
        is_halted: bool = False,
        current_date: Optional[date] = None,
    ) -> RiskCheckResult:
        """Apply market risk rules before execution."""
        self._refresh_day(current_date)

        if is_halted:
            msg = f"{self.symbol} is halted – skipping order"
            log.warning(msg)
            return RiskCheckResult(False, HOLD, msg, "WARNING")

        if self._protect_mode:
            return RiskCheckResult(True, HOLD, "protect_mode", "WARNING")

        if prev_close <= 0:
            return RiskCheckResult(True, action, "prev_close_invalid_skip_gap", "WARNING")

        gap = (next_open - prev_close) / prev_close

        # Downward gap
        if gap <= -self.cfg.gap_threshold:
            self._gap_hits_today += 1
            log.warning(
                f"Downward gap {gap:.2%} on {self.symbol}. "
                f"hits_today={self._gap_hits_today}"
            )
            if self._gap_hits_today >= self.cfg.max_daily_gap_hits:
                self._protect_mode = True
                log.critical(
                    f"Entering protect mode after {self._gap_hits_today} gap hits today."
                )
                return RiskCheckResult(True, HOLD, "protect_mode_activated", "CRITICAL")

            _down_action = _GAP_ACTION_MAP.get(self.cfg.gap_action_down, HOLD)
            new_action = _down_action if action == BUY_ALL else HOLD
            msg = f"Downward gap: action downgraded to {new_action}"
            return RiskCheckResult(True, new_action, msg, "WARNING")

        # Upward gap – allow but warn
        if gap >= self.cfg.gap_threshold:
            log.warning(f"Upward gap {gap:.2%} on {self.symbol} – flagged but allowed")
            return RiskCheckResult(True, action, f"upward_gap_{gap:.2%}_allowed", "WARNING")

        return RiskCheckResult(True, action, "market_ok", "INFO")

    # ------------------------------------------------------------------
    # Layer 5 – Runtime / NAV drawdown halt
    # ------------------------------------------------------------------

    def update_nav(self, nav: float, timestamp: Optional[str] = None) -> Optional[RiskCheckResult]:
        """Call on every bar.  Returns a result if drawdown halt is triggered."""
        if self._initial_nav <= 0:
            self._initial_nav = nav
        self._session_high_nav = max(self._session_high_nav, nav)

        drawdown = (self._session_high_nav - nav) / (self._session_high_nav + 1e-8)
        halt_thr = (
            self.cfg.nav_drawdown_halt_train
            if self._training_mode
            else self.cfg.nav_drawdown_halt
        )
        if drawdown >= halt_thr:
            msg = (
                f"Intraday NAV drawdown {drawdown:.2%} >= halt threshold "
                f"{halt_thr:.2%}{' [train]' if self._training_mode else ''}"
            )
            log.critical(msg)
            self._halt_mode = True
            return RiskCheckResult(False, HOLD, msg, "CRITICAL")
        return None

    def enter_halt(self, reason: str = "manual") -> None:
        log.critical(f"Entering halt mode: {reason}")
        self._halt_mode = True

    def reset_halt(self) -> None:
        log.info("Halt mode reset by operator")
        self._halt_mode = False
        self._protect_mode = False

    def reset_session(self, initial_nav: float = 0.0) -> None:
        """Call at session start (e.g., start of trading day)."""
        today = date.today()
        self._refresh_day(today)
        if initial_nav > 0:
            self._initial_nav = initial_nav
            self._session_high_nav = initial_nav

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh_day(self, current_date: Optional[date]) -> None:
        today = current_date or date.today()
        if today != self._today:
            self._today = today
            self._gap_hits_today = 0
            # protect_mode resets each new day
            if self._protect_mode:
                log.info("New trading day – protect mode cleared")
                self._protect_mode = False
