"""
Single-ETF trading environment (Gymnasium style).

Design (doc 03_交易环境设计):
  - Actions: 0=hold, 1=buy_all, 2=sell_all
  - Execution: next-bar open price (t+1)
  - Fee: double-sided, default 万0.5 (0.00005), min 5 CNY
  - Minimum trade unit: lot_size shares (default 100)
  - Reward: log(nav_t / nav_{t-1})  or  trade P&L on close
  - State: flat feature matrix window + [position_ratio, cash_ratio, cost_norm, is_holding]
  - No look-ahead: action at bar t uses only info up to close of bar t,
    and is executed at open of bar t+1.

Usage::

    env = ETFTradingEnv(df_features, df_prices, cfg_env, cfg_feat)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(1)  # buy_all

df_features index must be aligned with df_prices index (same timestamps).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .config import EnvConfig, FeatureConfig

# Action constants
HOLD = 0
BUY_ALL = 1
SELL_ALL = 2

ACTION_NAMES = {HOLD: "hold", BUY_ALL: "buy_all", SELL_ALL: "sell_all"}


class ETFTradingEnv(gym.Env):
    """Single-ETF discrete-action trading environment."""

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        df_features: pd.DataFrame,
        df_prices: pd.DataFrame,
        cfg_env: EnvConfig,
        cfg_feat: FeatureConfig,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self._feat = df_features.values.astype(np.float32)      # (T, n_feat)
        self._open = df_prices["open"].values.astype(np.float32)
        self._close = df_prices["close"].values.astype(np.float32)
        self._timestamps = df_prices.index

        self._n_feat = self._feat.shape[1]
        self._lookback = cfg_feat.lookback_window
        self._cfg = cfg_env

        # The first valid step index (need lookback-1 prior bars)
        self._start_idx = self._lookback - 1
        self._end_idx = len(self._feat) - 2   # need t+1 for execution

        if self._end_idx <= self._start_idx:
            raise ValueError("Not enough data bars for the given lookback_window.")

        # State dim: lookback * n_feat + 4 account variables
        flat_size = self._lookback * self._n_feat + 4
        high_obs = np.full(flat_size, np.inf, dtype=np.float32)
        low_obs = np.full(flat_size, -np.inf, dtype=np.float32)

        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.render_mode = render_mode

        # Episode state (set in reset)
        self._idx: int = self._start_idx
        self._cash: float = cfg_env.initial_cash
        self._shares: float = 0.0
        self._cost_basis: float = 0.0   # avg buy price
        self._nav0: float = cfg_env.initial_cash
        self._nav_prev: float = cfg_env.initial_cash
        self._trade_log: List[dict] = []
        self._risk_blocked_today: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._idx = self._start_idx
        self._cash = self._cfg.initial_cash
        self._shares = 0.0
        self._cost_basis = 0.0
        self._nav0 = self._cfg.initial_cash
        self._nav_prev = self._cfg.initial_cash
        self._trade_log = []
        self._risk_blocked_today = 0
        obs = self._get_obs()
        return obs, self._get_info()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        exec_price = float(self._open[self._idx + 1])   # t+1 open
        prev_nav = self._nav()
        prev_shares = self._shares  # track position change for turnover penalty

        # Execute action
        self._execute(action, exec_price)

        # Advance time
        self._idx += 1

        nav_now = self._nav()
        traded = self._shares != prev_shares
        reward = self._compute_reward(prev_nav, nav_now, traded=traded)

        terminated = self._idx >= self._end_idx
        truncated = False

        obs = self._get_obs()
        info = self._get_info(
            action=action,
            executed_price=exec_price,
            prev_nav=prev_nav,
            nav=nav_now,
        )
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        nav = self._nav()
        ret = nav / self._nav0 - 1
        msg = (
            f"t={self._idx:5d} | NAV={nav:10.2f} | ret={ret:+.2%} | "
            f"cash={self._cash:10.2f} | shares={self._shares:.0f} | "
            f"cost={self._cost_basis:.4f}"
        )
        if self.render_mode == "human":
            print(msg)
        return msg

    def get_trade_log(self) -> pd.DataFrame:
        if not self._trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self._trade_log)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        start = self._idx - self._lookback + 1
        feat_window = self._feat[start : self._idx + 1].flatten()  # (lookback * n_feat,)

        last_price = float(self._close[self._idx])
        nav = self._nav()

        position_ratio = (self._shares * last_price) / (nav + 1e-8)
        cash_ratio = self._cash / (nav + 1e-8)
        cost_norm = (self._cost_basis / (last_price + 1e-8)) - 1.0 if self._shares > 0 else 0.0
        is_holding = 1.0 if self._shares > 0 else 0.0

        account = np.array(
            [position_ratio, cash_ratio, cost_norm, is_holding], dtype=np.float32
        )
        return np.concatenate([feat_window.astype(np.float32), account])

    def _get_info(
        self,
        action: int = HOLD,
        executed_price: float = 0.0,
        prev_nav: float = 0.0,
        nav: Optional[float] = None,
    ) -> dict:
        if nav is None:
            nav = self._nav()
        ts = str(self._timestamps[self._idx]) if self._idx < len(self._timestamps) else ""
        return {
            "timestamp": ts,
            "action": ACTION_NAMES.get(action, str(action)),
            "executed_price": executed_price,
            "position": float(self._shares),
            "cash": float(self._cash),
            "nav": nav,
            "cost_basis": float(self._cost_basis),
            "nav_return": nav / (self._nav0 if self._nav0 != 0 else 1.0) - 1,
        }

    def _nav(self) -> float:
        return self._cash + self._shares * float(self._close[self._idx])

    def _compute_reward(self, prev_nav: float, cur_nav: float, traded: bool = False) -> float:
        penalty = self._cfg.turnover_penalty if traded else 0.0
        if self._cfg.reward_mode == "nav_log":
            if prev_nav <= 0 or cur_nav <= 0:
                return -penalty
            return float(np.log(cur_nav / prev_nav)) - penalty
        # trade_pnl: reward only on trade events (handled externally for now)
        return float(cur_nav - prev_nav) / (self._cfg.initial_cash + 1e-8) - penalty

    def _execute(self, action: int, price: float) -> None:
        cfg = self._cfg

        if action == BUY_ALL and self._cash > 0:
            max_shares = (self._cash / price) // cfg.lot_size * cfg.lot_size
            if max_shares <= 0:
                return
            gross = max_shares * price
            fee = max(gross * cfg.fee_rate_buy, cfg.min_fee)
            total_cost = gross + fee
            if total_cost > self._cash:
                # Re-compute affordable shares after fee
                affordable = ((self._cash - cfg.min_fee) / (price * (1 + cfg.fee_rate_buy)))
                max_shares = affordable // cfg.lot_size * cfg.lot_size
                if max_shares <= 0:
                    return
                gross = max_shares * price
                fee = max(gross * cfg.fee_rate_buy, cfg.min_fee)
                total_cost = gross + fee
            self._cash -= total_cost
            # Update avg cost basis
            prev_val = self._shares * self._cost_basis
            self._shares += max_shares
            self._cost_basis = (prev_val + gross) / self._shares
            self._trade_log.append({
                "timestamp": str(self._timestamps[self._idx + 1])
                    if self._idx + 1 < len(self._timestamps) else "",
                "action": "buy_all",
                "price": price,
                "shares": max_shares,
                "fee": fee,
                "cash_after": self._cash,
                "nav": self._nav(),
            })

        elif action == SELL_ALL and self._shares > 0:
            gross = self._shares * price
            fee = max(gross * cfg.fee_rate_sell, cfg.min_fee)
            proceeds = gross - fee
            self._cash += proceeds
            cost = self._shares * self._cost_basis
            pnl = proceeds - cost
            self._trade_log.append({
                "timestamp": str(self._timestamps[self._idx + 1])
                    if self._idx + 1 < len(self._timestamps) else "",
                "action": "sell_all",
                "price": price,
                "shares": self._shares,
                "fee": fee,
                "pnl": pnl,
                "cash_after": self._cash,
                "nav": self._nav(),
            })
            self._shares = 0.0
            self._cost_basis = 0.0
