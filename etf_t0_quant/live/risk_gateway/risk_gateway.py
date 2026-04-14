"""
Risk Gateway (doc 07_实盘qmt交易 – risk_gateway).

The risk gateway sits between signal_service and qmt_gateway.
It has the final authority to intercept, modify, or allow a trading signal.

Validates:
  - Signal completeness (required fields present)
  - Account state consistency
  - Market conditions (gap, halt)
  - Pre-trade checks (position limits, cash, action legality)
  - Model metadata completeness

Usage::

    gw = RiskGateway(cfg)
    result = gw.validate(signal, account_state, market_state)
    if result.allowed:
        qmt.place_order(result.final_signal)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from etf_t0_quant.config import AppConfig
from etf_t0_quant.env import BUY_ALL, HOLD, SELL_ALL
from etf_t0_quant.logger import get_logger
from etf_t0_quant.risk import RiskManager

log = get_logger("risk")

_REQUIRED_SIGNAL_FIELDS = {
    "timestamp", "symbol", "interval", "action", "action_code",
    "model_version", "run_id",
}
_ACTION_MAP = {"hold": HOLD, "buy_all": BUY_ALL, "sell_all": SELL_ALL}


@dataclass
class GatewayResult:
    allowed: bool
    final_signal: Optional[Dict]
    reason: str
    level: str  # INFO / WARNING / ERROR / CRITICAL


class RiskGateway:
    """Validate and potentially modify a trading signal before execution."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._rm = RiskManager(cfg.risk, cfg.base.symbol, cfg.base.interval)

    # ------------------------------------------------------------------

    def validate(
        self,
        signal: Dict,
        account_state: Dict,
        market_state: Optional[Dict] = None,
    ) -> GatewayResult:
        """
        Validate *signal* against account and market state.

        account_state keys: position_shares, cash, nav, current_price
        market_state  keys: next_open, prev_close, is_halted (all optional)
        """
        # 1. Signal completeness
        missing = _REQUIRED_SIGNAL_FIELDS - set(signal.keys())
        if missing:
            msg = f"Signal missing fields: {missing}"
            log.error(msg)
            return GatewayResult(False, None, msg, "ERROR")

        action_code = _ACTION_MAP.get(signal.get("action", ""), HOLD)

        # 2. Pre-trade
        position = float(account_state.get("position_shares", 0))
        cash = float(account_state.get("cash", 0))
        nav = float(account_state.get("nav", cash))
        price = float(account_state.get("current_price", 0))

        pt = self._rm.check_pre_trade(
            action=action_code,
            position_shares=position,
            cash=cash,
            nav=nav,
            price=price,
            lot_size=self.cfg.env.lot_size,
        )
        if not pt.allowed:
            return GatewayResult(False, None, pt.reason, pt.level)
        action_code = pt.modified_action

        # 3. Market conditions
        if market_state:
            now_date = None
            ts_str = signal.get("timestamp", "")
            if ts_str:
                try:
                    now_date = datetime.fromisoformat(ts_str).date()
                except ValueError:
                    pass
            mc = self._rm.check_market_conditions(
                action=action_code,
                next_open=float(market_state.get("next_open", 0)),
                prev_close=float(market_state.get("prev_close", 1)),
                is_halted=bool(market_state.get("is_halted", False)),
                current_date=now_date,
            )
            if not mc.allowed:
                return GatewayResult(False, None, mc.reason, mc.level)
            action_code = mc.modified_action

        # 4. Emit final signal
        from etf_t0_quant.env import ACTION_NAMES
        final = dict(signal)
        final["action_code"] = action_code
        final["action"] = ACTION_NAMES.get(action_code, str(action_code))
        final["gateway_ts"] = datetime.now().isoformat()

        log.info(
            f"Gateway approved: {final['action']} {final['symbol']} "
            f"(model={final.get('model_version', '')})"
        )
        return GatewayResult(True, final, "approved", "INFO")

    def reset_session(self, initial_nav: float = 0.0) -> None:
        self._rm.reset_session(initial_nav)

    def update_nav(self, nav: float) -> None:
        result = self._rm.update_nav(nav)
        if result and not result.allowed:
            log.critical(f"NAV halt triggered: {result.reason}")
