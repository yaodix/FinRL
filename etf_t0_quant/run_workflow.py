#!/usr/bin/env python3
"""
End-to-end one-shot workflow:
  tickflow data fetch → features → Optuna HPO → single DQN train → backtest → report

Usage
-----
    # Token via CLI
    python etf_t0_quant/run_workflow.py --token YOUR_TOKEN

    # Token via environment variable (recommended)
    ETF_TICKFLOW_TOKEN=xxx python etf_t0_quant/run_workflow.py

    # Custom options
    python etf_t0_quant/run_workflow.py \\
        --token YOUR_TOKEN \\
        --symbol 159740 \\
        --interval 30m \\
        --n_trials 10 \\
        --total_timesteps 50000

Notes
-----
- 159740.SZ is the Hang Seng Tech ETF (恒生科技ETF) listed on Shenzhen.
- Requires a tickflow full-service API key for 30m minute candles.
  Free tier only provides daily data.  Get a key at https://tickflow.org/
- Install SDK first: pip install "tickflow[all]"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is importable when run as a script
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from etf_t0_quant.backtester import Backtester
from etf_t0_quant.config import AppConfig, load_config
from etf_t0_quant.data_pipeline import DataPipeline
from etf_t0_quant.logger import get_run_id, setup_logging
from etf_t0_quant.trainer import ETFTrainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ETF T0 – full workflow from tickflow data to backtest report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--token",
        default=os.environ.get("ETF_TICKFLOW_TOKEN", ""),
        help="Tickflow API key (or set ETF_TICKFLOW_TOKEN env var)",
    )
    p.add_argument("--symbol", default="159740", help="ETF code (no market suffix)")
    p.add_argument("--interval", default="30m", choices=["30m", "15m"], help="Bar interval")
    p.add_argument("--n_trials", type=int, default=10, help="Optuna HPO trial count")
    p.add_argument(
        "--total_timesteps",
        type=int,
        default=50_000,
        help="DQN timesteps for the final training run",
    )
    p.add_argument("--train_frac", type=float, default=0.70, help="Train split fraction")
    p.add_argument("--val_frac", type=float, default=0.15, help="Validation split fraction")
    p.add_argument("--config", default=None, help="Optional path to a YAML config override")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print(f"{'=' * 62}")


def _split(df_feat, df_prices, train_frac: float, val_frac: float):
    """Chronological 3-way split, returns (train, val, test) tuples."""
    n = len(df_feat)
    n_tr = int(n * train_frac)
    n_va = int(n * val_frac)

    def _s(df, a, b):
        return df.iloc[a:b]

    tr = (_s(df_feat, 0, n_tr), _s(df_prices, 0, n_tr))
    va = (_s(df_feat, n_tr, n_tr + n_va), _s(df_prices, n_tr, n_tr + n_va))
    te = (_s(df_feat, n_tr + n_va, n), _s(df_prices, n_tr + n_va, n))
    return tr, va, te


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.token:
        print(
            "ERROR: Tickflow API key required.\n"
            "  Pass --token YOUR_KEY  or  export ETF_TICKFLOW_TOKEN=YOUR_KEY",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── 1. Config ────────────────────────────────────────────────────────────
    _banner("1 / 7  Configuration")
    cfg: AppConfig = load_config(args.config)
    cfg.base.symbol = args.symbol
    cfg.base.interval = args.interval
    cfg.data.source = "tickflow"
    cfg.data.tickflow_token = args.token
    cfg.train.n_trials = args.n_trials
    cfg.train.total_timesteps = args.total_timesteps

    setup_logging(cfg.base.logs_dir)

    print(f"  Symbol          : {args.symbol}.SZ  (恒生科技ETF)")
    print(f"  Interval        : {args.interval}")
    print(f"  Optuna trials   : {args.n_trials}")
    print(f"  Final train TS  : {args.total_timesteps:,}")
    print(f"  Split           : {args.train_frac:.0%} / {args.val_frac:.0%} / "
          f"{1 - args.train_frac - args.val_frac:.0%}")

    # ── 2. Data pipeline ─────────────────────────────────────────────────────
    _banner("2 / 7  Fetching & Building Features")
    t0 = time.perf_counter()
    pipeline = DataPipeline(cfg)
    df_feat, df_prices, meta = pipeline.run("full")
    elapsed = time.perf_counter() - t0

    n_bars = len(df_feat)
    print(f"  Bars fetched    : {n_bars}  ({elapsed:.1f}s)")
    print(f"  Date range      : {df_feat.index[0]}  →  {df_feat.index[-1]}")
    print(f"  Feature columns : {df_feat.shape[1]}")

    if n_bars < 300:
        print(
            "WARNING: fewer than 300 bars available – model quality will be poor.",
            file=sys.stderr,
        )

    # ── 3. Train / Val / Test split ──────────────────────────────────────────
    _banner("3 / 7  Chronological Split")
    (tr_feat, tr_prices), (va_feat, va_prices), (te_feat, te_prices) = _split(
        df_feat, df_prices, args.train_frac, args.val_frac
    )
    for label, feat in (("Train", tr_feat), ("Val  ", va_feat), ("Test ", te_feat)):
        d0, d1 = feat.index[0].date(), feat.index[-1].date()
        print(f"  {label}: {len(feat):>5} bars   {d0} → {d1}")

    # ── 4. Optuna HPO ────────────────────────────────────────────────────────
    _banner(f"4 / 7  Optuna Hyperparameter Search  ({args.n_trials} trials)")
    # Use a reduced timestep budget per trial so HPO completes quickly
    hpo_ts = max(5_000, args.total_timesteps // 5)
    cfg.train.total_timesteps = hpo_ts
    print(f"  Timesteps/trial : {hpo_ts:,}  (1/5 of final budget)")
    trainer_hpo = ETFTrainer(cfg)
    best_params = trainer_hpo.run_optuna_search(tr_feat, tr_prices, va_feat, va_prices)
    print(f"  Best params     : {best_params}")

    # ── 5. Final training ────────────────────────────────────────────────────
    _banner("5 / 7  Final DQN Training")
    cfg.train.total_timesteps = args.total_timesteps
    trainer = ETFTrainer(cfg)
    run_id = get_run_id()
    t0 = time.perf_counter()
    model = trainer.train_single(tr_feat, tr_prices, overrides=best_params or {}, run_id=run_id)
    elapsed = time.perf_counter() - t0
    model_path = trainer.save_model(model, run_id=run_id)
    print(f"  Training time   : {elapsed:.1f}s")
    print(f"  Model saved to  : {model_path}")

    # ── 6. Backtest ──────────────────────────────────────────────────────────
    _banner("6 / 7  Backtest on Test Set")
    backtester = Backtester(cfg)
    result = backtester.run(
        model, te_feat, te_prices, window_label=f"test_{run_id[:8]}"
    )

    m = result.metrics
    print(f"\n  {'Metric':<28} {'Value':>12}")
    print(f"  {'-' * 42}")
    for k, v in m.items():
        if isinstance(v, float):
            if "rate" in k or "return" in k or "drawdown" in k:
                print(f"  {k:<28} {v:>11.2%}")
            else:
                print(f"  {k:<28} {v:>12.4f}")
        else:
            print(f"  {k:<28} {str(v):>12}")

    # ── 7. Save report ───────────────────────────────────────────────────────
    _banner("7 / 7  Saving Report")
    out_dir = cfg.base.backtests_dir / f"run_{run_id[:12]}"
    report_json = backtester.save_report(result, out_dir=out_dir, prefix="workflow")
    print(f"  Report JSON     : {report_json}")
    print(f"  Chart PNG       : {out_dir}")

    summary = {
        "run_id": run_id,
        "symbol": args.symbol,
        "interval": args.interval,
        "total_bars": n_bars,
        "date_range": [str(df_feat.index[0]), str(df_feat.index[-1])],
        "split": {
            "train_bars": len(tr_feat),
            "val_bars": len(va_feat),
            "test_bars": len(te_feat),
        },
        "hpo_timesteps_per_trial": hpo_ts,
        "best_hpo_params": best_params,
        "final_timesteps": args.total_timesteps,
        "model_path": str(model_path),
        "test_metrics": m,
        "report_dir": str(out_dir),
    }
    summary_path = out_dir / "workflow_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  Workflow JSON   : {summary_path}")

    # ── Final summary ────────────────────────────────────────────────────────
    _banner("Done")
    sharpe = m.get("sharpe_ratio", float("nan"))
    max_dd = m.get("max_drawdown", float("nan"))
    cum_ret = m.get("cumulative_return", float("nan"))
    n_trades = m.get("n_trades", 0)
    print(f"  Sharpe Ratio    : {sharpe:.4f}")
    print(f"  Max Drawdown    : {max_dd:.2%}")
    print(f"  Cumulative Ret  : {cum_ret:.2%}")
    print(f"  # Trades        : {n_trades}")
    print()


if __name__ == "__main__":
    main()
