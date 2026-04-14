"""
Backtester and Walk-Forward evaluation module (doc 05_回测).

Usage::

    backtester = Backtester(cfg, risk_manager)
    result = backtester.run(model, df_features, df_prices)
    print(result.metrics)
    backtester.save_report(result, out_dir)

    # Walk-Forward
    wf = WalkForwardBacktester(cfg)
    wf_results = wf.run(trainer_fn, df_features, df_prices)
    publish_ok = wf.evaluate_publish_decision(wf_results)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import AppConfig, BacktestConfig, EnvConfig, FeatureConfig, RiskConfig, TrainConfig
from .env import BUY_ALL, HOLD, SELL_ALL, ETFTradingEnv
from .logger import get_logger, get_run_id
from .risk import RiskManager

log = get_logger("backtest")

BARS_PER_DAY: Dict[str, int] = {"30m": 8, "15m": 16}


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _annualised_return(nav_series: np.ndarray, bars_per_day: int) -> float:
    if len(nav_series) < 2:
        return 0.0
    total = nav_series[-1] / nav_series[0] - 1
    years = len(nav_series) / (bars_per_day * 252)
    if years <= 0:
        return 0.0
    return float((1 + total) ** (1 / years) - 1)


def _max_drawdown(nav_series: np.ndarray) -> float:
    peak = np.maximum.accumulate(nav_series)
    dd = (nav_series - peak) / peak
    return float(-dd.min()) if len(dd) > 0 else 0.0


def _sharpe(returns: np.ndarray, bars_per_year: int) -> float:
    if len(returns) < 2:
        return 0.0
    mu = returns.mean()
    sigma = returns.std()
    if sigma <= 0:
        return 0.0
    return float(mu / sigma * np.sqrt(bars_per_year))


def _sortino(returns: np.ndarray, bars_per_year: int) -> float:
    if len(returns) < 2:
        return 0.0
    mu = returns.mean()
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float("inf")
    sigma_d = downside.std()
    if sigma_d <= 0:
        return 0.0
    return float(mu / sigma_d * np.sqrt(bars_per_year))


def _profit_factor(pnl_list: List[float]) -> float:
    gains = sum(p for p in pnl_list if p > 0)
    losses = -sum(p for p in pnl_list if p < 0)
    if losses <= 0:
        return float("inf") if gains > 0 else 1.0
    return float(gains / losses)


def compute_metrics(
    nav_series: np.ndarray,
    trade_log: pd.DataFrame,
    bars_per_day: int,
) -> Dict[str, float]:
    """Compute all performance metrics required by doc 05."""
    bars_per_year = bars_per_day * 252
    log_rets = np.diff(np.log(nav_series + 1e-10))
    pnl_list = trade_log["pnl"].tolist() if "pnl" in trade_log.columns else []

    metrics: Dict[str, float] = {
        "cumulative_return": float(nav_series[-1] / nav_series[0] - 1) if len(nav_series) > 1 else 0.0,
        "annualised_return": _annualised_return(nav_series, bars_per_day),
        "sharpe_ratio": _sharpe(log_rets, bars_per_year),
        "sortino_ratio": _sortino(log_rets, bars_per_year),
        "max_drawdown": _max_drawdown(nav_series),
        "n_trades": int(len(trade_log)),
        "profit_factor": _profit_factor(pnl_list),
        "win_rate": float(sum(1 for p in pnl_list if p > 0) / len(pnl_list)) if pnl_list else 0.0,
        "avg_holding_bars": 0.0,
    }

    if "shares" in trade_log.columns and len(trade_log) > 1:
        # Naive avg holding: (sell_t - buy_t) bars
        buys = trade_log[trade_log["action"] == "buy_all"].index.tolist()
        sells = trade_log[trade_log["action"] == "sell_all"].index.tolist()
        holdings = [s - b for b, s in zip(buys, sells) if s > b]
        metrics["avg_holding_bars"] = float(np.mean(holdings)) if holdings else 0.0

    return metrics


def buy_and_hold_nav(df_prices: pd.DataFrame, initial_cash: float) -> np.ndarray:
    """Return NAV series for a simple buy-and-hold strategy."""
    opens = df_prices["open"].values
    closes = df_prices["close"].values
    shares = initial_cash // opens[0]
    remaining_cash = initial_cash - shares * opens[0]
    nav = closes * shares + remaining_cash
    return nav.astype(np.float64)


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    metrics: Dict[str, float] = field(default_factory=dict)
    nav_series: np.ndarray = field(default_factory=lambda: np.array([]))
    trade_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    position_series: np.ndarray = field(default_factory=lambda: np.array([]))
    bah_nav: np.ndarray = field(default_factory=lambda: np.array([]))
    risk_stats: Dict[str, Any] = field(default_factory=dict)
    passed_threshold: bool = False
    run_id: str = ""
    window_label: str = ""

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "window_label": self.window_label,
            "metrics": self.metrics,
            "passed_threshold": self.passed_threshold,
            "risk_stats": self.risk_stats,
            "n_nav_points": len(self.nav_series),
            "n_trades": len(self.trade_log),
        }


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """Run a trained model on a dataset and collect performance metrics."""

    def __init__(
        self,
        cfg: AppConfig,
        risk_manager: Optional[RiskManager] = None,
    ) -> None:
        self.cfg = cfg
        self._rm = risk_manager or RiskManager(cfg.risk, cfg.base.symbol, cfg.base.interval)
        self._bars_per_day = BARS_PER_DAY.get(cfg.base.interval, 8)

    def run(
        self,
        model,  # stable-baselines3 DQN or any object with .predict(obs)
        df_features: pd.DataFrame,
        df_prices: pd.DataFrame,
        window_label: str = "test",
    ) -> BacktestResult:
        """Replay model on data and return BacktestResult."""
        env = ETFTradingEnv(
            df_features,
            df_prices,
            self.cfg.env,
            self.cfg.feature,
        )
        obs, _ = env.reset()

        nav_list: List[float] = [self.cfg.env.initial_cash]
        position_list: List[float] = [0.0]
        risk_intercepts: List[Dict] = []

        prev_close = float(df_prices["close"].iloc[0])
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            # Market risk check
            current_idx = env._idx
            if current_idx + 1 < len(df_prices):
                next_open = float(df_prices["open"].iloc[current_idx + 1])
                rm_result = self._rm.check_market_conditions(
                    action=action,
                    next_open=next_open,
                    prev_close=prev_close,
                    is_halted=False,
                    current_date=df_prices.index[current_idx].date()
                    if hasattr(df_prices.index[current_idx], "date") else None,
                )
                if rm_result.modified_action != action:
                    if rm_result.reason:
                        risk_intercepts.append({"step": current_idx, "reason": rm_result.reason})
                    action = rm_result.modified_action

            # Pre-trade risk
            info_pre = env._get_info()
            pt_result = self._rm.check_pre_trade(
                action=action,
                position_shares=info_pre["position"],
                cash=info_pre["cash"],
                nav=info_pre["nav"],
                price=next_open if current_idx + 1 < len(df_prices) else 0.0,
                lot_size=self.cfg.env.lot_size,
            )
            if not pt_result.allowed:
                risk_intercepts.append({"step": current_idx, "reason": pt_result.reason})
                action = HOLD

            obs, reward, terminated, truncated, info = env.step(action)
            prev_close = float(df_prices["close"].iloc[env._idx])
            nav_list.append(info["nav"])
            position_list.append(info["position"])

            # NAV halt check
            halt = self._rm.update_nav(info["nav"])
            if halt:
                log.warning(f"NAV halt triggered at step {env._idx}")
                break

            done = terminated or truncated

        nav_arr = np.array(nav_list, dtype=np.float64)
        pos_arr = np.array(position_list, dtype=np.float64)
        trade_log = env.get_trade_log()
        bah = buy_and_hold_nav(df_prices, self.cfg.env.initial_cash)

        metrics = compute_metrics(nav_arr, trade_log, self._bars_per_day)

        # Risk stats
        risk_stats = {
            "total_intercepts": len(risk_intercepts),
            "intercept_reasons": {},
        }
        for item in risk_intercepts:
            r = item["reason"]
            risk_stats["intercept_reasons"][r] = risk_stats["intercept_reasons"].get(r, 0) + 1

        # Check threshold
        bc = self.cfg.backtest
        passed = (
            metrics["sharpe_ratio"] >= bc.sharpe_threshold
            and metrics["max_drawdown"] <= bc.max_drawdown_threshold
            and metrics["profit_factor"] >= bc.profit_factor_threshold
            and metrics["n_trades"] >= bc.min_trades
        )

        return BacktestResult(
            metrics=metrics,
            nav_series=nav_arr,
            trade_log=trade_log,
            position_series=pos_arr,
            bah_nav=bah,
            risk_stats=risk_stats,
            passed_threshold=passed,
            run_id=get_run_id(),
            window_label=window_label,
        )

    def save_report(
        self,
        result: BacktestResult,
        out_dir: Path,
        prefix: str = "",
    ) -> Path:
        """Save JSON summary + PNG charts to *out_dir*."""
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{prefix}_{result.window_label}" if prefix else result.window_label

        # JSON
        report_path = out_dir / f"{tag}_report.json"
        report_path.write_text(json.dumps(result.to_dict(), indent=2, default=str))

        # Charts
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(12, 14))
            fig.suptitle(f"Backtest: {tag}", fontsize=14)

            # NAV curve
            axes[0].plot(result.nav_series, label="Strategy", linewidth=1.5)
            if len(result.bah_nav) > 0:
                n = min(len(result.nav_series), len(result.bah_nav))
                axes[0].plot(result.bah_nav[:n], label="Buy&Hold", linewidth=1, linestyle="--", alpha=0.8)
            axes[0].set_title("NAV Curve")
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            # Drawdown
            peak = np.maximum.accumulate(result.nav_series)
            dd = (result.nav_series - peak) / (peak + 1e-8)
            axes[1].fill_between(range(len(dd)), dd, 0, alpha=0.4, color="red")
            axes[1].set_title(f"Drawdown  (max={result.metrics['max_drawdown']:.2%})")
            axes[1].grid(alpha=0.3)

            # Position
            axes[2].step(range(len(result.position_series)), result.position_series, linewidth=1)
            axes[2].set_title("Position (shares)")
            axes[2].grid(alpha=0.3)

            plt.tight_layout()
            chart_path = out_dir / f"{tag}_chart.png"
            fig.savefig(chart_path, dpi=120)
            plt.close(fig)

        except Exception as exc:  # noqa: BLE001
            log.warning(f"Chart generation failed: {exc}")

        log.info(f"Report saved to {out_dir / tag}")
        return report_path


# ---------------------------------------------------------------------------
# Walk-Forward Backtester
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    windows: List[BacktestResult] = field(default_factory=list)
    n_passed: int = 0
    n_total: int = 0
    publish_ok: bool = False
    summary_metrics: Dict[str, float] = field(default_factory=dict)


def _slice_window(
    df_feat: pd.DataFrame,
    df_prices: pd.DataFrame,
    start: int,
    end: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return df_feat.iloc[start:end], df_prices.iloc[start:end]


class WalkForwardBacktester:
    """Walk-Forward training + evaluation loop."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._bars_per_day = BARS_PER_DAY.get(cfg.base.interval, 8)

    def run(
        self,
        trainer_fn: Callable[[pd.DataFrame, pd.DataFrame], Any],
        df_features: pd.DataFrame,
        df_prices: pd.DataFrame,
        out_dir: Optional[Path] = None,
    ) -> WalkForwardResult:
        """Run walk-forward loop.

        Args:
            trainer_fn: callable(df_feat_train, df_prices_train) → trained model
            df_features: full normalised feature dataframe
            df_prices:   aligned OHLCV dataframe

        Returns:
            WalkForwardResult with per-window BacktestResult objects
        """
        tc = self.cfg.train
        bpd = self._bars_per_day

        train_bars = tc.train_days * bpd
        val_bars = tc.val_days * bpd
        test_bars = tc.test_days * bpd
        step_bars = tc.step_days * bpd
        window_bars = train_bars + val_bars + test_bars

        n = len(df_features)
        if n < window_bars:
            raise ValueError(
                f"Not enough bars ({n}) for one WF window ({window_bars}). "
                "Reduce train/val/test days or provide more data."
            )

        wf_result = WalkForwardResult()
        out_dir = out_dir or (self.cfg.base.backtests_dir / "walk_forward")
        backtester = Backtester(self.cfg)

        cursor = 0
        win_idx = 0
        while cursor + window_bars <= n:
            train_start = cursor
            train_end = cursor + train_bars
            val_end = train_end + val_bars
            test_end = val_end + test_bars

            log.info(
                f"WF window {win_idx}: train[{train_start}:{train_end}] "
                f"val[{train_end}:{val_end}] test[{val_end}:{test_end}]"
            )

            # Slice
            df_ft, df_pt = _slice_window(df_features, df_prices, train_start, train_end)
            df_fv, df_pv = _slice_window(df_features, df_prices, train_end, val_end)
            df_ftest, df_ptest = _slice_window(df_features, df_prices, val_end, test_end)

            # Train
            try:
                model = trainer_fn(df_ft, df_pt)
            except Exception as exc:
                log.error(f"Training failed for window {win_idx}: {exc}")
                cursor += step_bars
                win_idx += 1
                continue

            # Val (for selecting best model in Optuna, not used directly here)
            val_result = backtester.run(model, df_fv, df_pv, window_label=f"wf{win_idx}_val")

            # Test
            test_result = backtester.run(
                model, df_ftest, df_ptest, window_label=f"wf{win_idx}_test"
            )
            wf_result.windows.append(test_result)

            if out_dir:
                backtester.save_report(test_result, out_dir, prefix=get_run_id())

            cursor += step_bars
            win_idx += 1

        wf_result.n_total = len(wf_result.windows)
        wf_result.n_passed = sum(1 for w in wf_result.windows if w.passed_threshold)
        required = max(1, round(self.cfg.backtest.wf_pass_ratio * wf_result.n_total))
        wf_result.publish_ok = wf_result.n_passed >= required

        # Aggregate metrics
        if wf_result.windows:
            for key in wf_result.windows[0].metrics:
                vals = [w.metrics[key] for w in wf_result.windows]
                wf_result.summary_metrics[f"{key}_mean"] = float(np.mean(vals))
                wf_result.summary_metrics[f"{key}_median"] = float(np.median(vals))

        log.info(
            f"WF done: {wf_result.n_passed}/{wf_result.n_total} windows passed. "
            f"publish_ok={wf_result.publish_ok}"
        )
        return wf_result
