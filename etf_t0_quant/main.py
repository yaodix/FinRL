"""
CLI entry point for ETF T0 Quant system.

Usage examples::

    # Run full pipeline: fetch data → features → train → backtest
    python -m etf_t0_quant pipeline --config etf_t0_quant/configs/default.yaml

    # Individual stages
    python -m etf_t0_quant data          # fetch & process data only
    python -m etf_t0_quant data --mode incremental
    python -m etf_t0_quant train         # walk-forward training
    python -m etf_t0_quant backtest      # replay best model on test set
    python -m etf_t0_quant publish <model_dir>   # mark model as published
    python -m etf_t0_quant infer         # one-shot inference (latest bar)
    python -m etf_t0_quant dashboard     # launch Streamlit dashboard
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="etf_t0_quant",
        description="ETF T0 Quant – single-ETF 15m/30m timing strategy system",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (default: etf_t0_quant/configs/default.yaml)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Explicit run ID (default: auto-generated UUID)",
    )

    sub = parser.add_subparsers(dest="command")

    # data
    p_data = sub.add_parser("data", help="Fetch and process data")
    p_data.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="full",
        help="full: re-fetch all history; incremental: append new bars",
    )

    # train
    p_train = sub.add_parser("train", help="Run walk-forward training")
    p_train.add_argument(
        "--no-optuna",
        action="store_true",
        help="Skip Optuna HPO and use config defaults",
    )
    p_train.add_argument(
        "--single",
        action="store_true",
        help="Single train/val/test split instead of walk-forward",
    )
    p_train.add_argument(
        "--split",
        default="0.75,0.875",
        help="Train/val split ratio for --single mode (default: 0.75,0.875)",
    )

    # backtest
    p_bt = sub.add_parser("backtest", help="Run backtest with a given model")
    p_bt.add_argument("--model-dir", default=None, help="Path to model directory")
    p_bt.add_argument("--split", default="0.875", help="Start fraction for test set")

    # publish
    p_pub = sub.add_parser("publish", help="Manually publish a model")
    p_pub.add_argument("model_dir", help="Path to model directory to publish")
    p_pub.add_argument("--force", action="store_true", help="Skip threshold check")

    # infer
    sub.add_parser("infer", help="Run single-step inference on latest data")

    # pipeline
    p_pipe = sub.add_parser("pipeline", help="Full pipeline: data + train + backtest")
    p_pipe.add_argument("--mode", choices=["full", "incremental"], default="full")

    # dashboard
    sub.add_parser("dashboard", help="Launch Streamlit dashboard")

    return parser


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _load(config_path):
    from etf_t0_quant.config import load_config
    from etf_t0_quant.logger import setup_logging, bind_run_id
    cfg = load_config(config_path)
    setup_logging(cfg.base.logs_dir)
    return cfg


def cmd_data(args, cfg):
    from etf_t0_quant.data_pipeline import DataPipeline
    from etf_t0_quant.logger import get_logger
    from etf_t0_quant.notifier import FeishuNotifier

    log = get_logger("data")
    notifier = FeishuNotifier(cfg.notifier)
    pipeline = DataPipeline(cfg)

    try:
        df_feat, df_prices, meta = pipeline.run(mode=args.mode)
        log.info(f"Data pipeline complete: {meta['feature_rows']} feature rows")
        notifier.send(
            "data_ok",
            f"数据更新成功 {meta['feature_rows']} bars",
            run_id=meta["run_id"],
            symbol=cfg.base.symbol,
            interval=cfg.base.interval,
        )
    except Exception as exc:
        log.error(f"Data pipeline failed: {exc}")
        notifier.send(
            "data_fail",
            f"数据更新失败: {exc}",
            symbol=cfg.base.symbol,
            interval=cfg.base.interval,
        )
        raise


def cmd_train(args, cfg, run_id: str):
    from etf_t0_quant.data_pipeline import DataPipeline
    from etf_t0_quant.trainer import ETFTrainer
    from etf_t0_quant.backtester import WalkForwardBacktester, Backtester
    from etf_t0_quant.logger import get_logger
    from etf_t0_quant.notifier import FeishuNotifier

    log = get_logger("train")
    notifier = FeishuNotifier(cfg.notifier)

    # Load data
    pipeline = DataPipeline(cfg)
    df_feat, df_prices, _ = pipeline.load_features()

    trainer = ETFTrainer(cfg)

    if getattr(args, "single", False):
        ratios = [float(x) for x in args.split.split(",")]
        n = len(df_feat)
        t_end = int(n * ratios[0])
        v_end = int(n * ratios[1]) if len(ratios) > 1 else t_end + int(n * 0.125)

        model = trainer.train_single(df_feat.iloc[:t_end], df_prices.iloc[:t_end])
        trainer.save_model(model, run_id)
        notifier.send(
            "train_complete",
            "单次划分训练完成",
            run_id=run_id,
            symbol=cfg.base.symbol,
        )
        return

    # Optuna HPO on first window
    if not getattr(args, "no_optuna", False):
        bpd = 8 if cfg.base.interval == "30m" else 16
        train_bars = cfg.train.train_days * bpd
        val_bars = cfg.train.val_days * bpd
        df_ft = df_feat.iloc[:train_bars]
        df_pt = df_prices.iloc[:train_bars]
        df_fv = df_feat.iloc[train_bars : train_bars + val_bars]
        df_pv = df_prices.iloc[train_bars : train_bars + val_bars]
        log.info("Running Optuna HPO on first window…")
        best_params = trainer.run_optuna_search(df_ft, df_pt, df_fv, df_pv)
    else:
        best_params = {}

    # Walk-Forward
    wf = WalkForwardBacktester(cfg)
    trainer_fn = trainer.make_trainer_fn(best_params)
    out_dir = cfg.base.backtests_dir / "walk_forward"
    results = wf.run(trainer_fn, df_feat, df_prices, out_dir=out_dir)

    log.info(
        f"WF complete: {results.n_passed}/{results.n_total} windows passed, "
        f"publish_ok={results.publish_ok}"
    )

    # Train final model on all data and save
    final_model = trainer.train_single(df_feat, df_prices, best_params)
    model_path = trainer.save_model(final_model, run_id)

    notifier.send(
        "train_complete" if results.publish_ok else "risk_warning",
        f"Walk-Forward 训练完成 {results.n_passed}/{results.n_total} 达标",
        run_id=run_id,
        symbol=cfg.base.symbol,
        interval=cfg.base.interval,
        extra={"model_dir": str(model_path), "publish_ok": results.publish_ok},
    )


def cmd_backtest(args, cfg, run_id: str):
    from etf_t0_quant.data_pipeline import DataPipeline
    from etf_t0_quant.trainer import ETFTrainer
    from etf_t0_quant.backtester import Backtester
    from etf_t0_quant.risk import RiskManager
    from etf_t0_quant.logger import get_logger
    from etf_t0_quant.notifier import FeishuNotifier

    log = get_logger("backtest")
    notifier = FeishuNotifier(cfg.notifier)

    pipeline = DataPipeline(cfg)
    df_feat, df_prices, _ = pipeline.load_features()

    model_dir_arg = getattr(args, "model_dir", None)
    if model_dir_arg:
        model_path = Path(model_dir_arg) / "model"
    else:
        # Find latest model
        import glob
        candidates = sorted(
            glob.glob(str(cfg.base.models_dir / cfg.base.symbol / cfg.base.interval / "*" / "model.zip")),
            reverse=True,
        )
        if not candidates:
            log.error("No model found. Run train first.")
            sys.exit(1)
        model_path = Path(candidates[0]).parent / "model"

    model = ETFTrainer.load_model(model_path)

    split = float(getattr(args, "split", "0.875"))
    n = len(df_feat)
    test_start = int(n * split)
    df_ftest = df_feat.iloc[test_start:]
    df_ptest = df_prices.iloc[test_start:]

    rm = RiskManager(cfg.risk, cfg.base.symbol, cfg.base.interval)
    backtester = Backtester(cfg, rm)
    result = backtester.run(model, df_ftest, df_ptest, window_label="final_test")

    out_dir = cfg.base.backtests_dir / "final"
    backtester.save_report(result, out_dir, prefix=run_id)

    log.info(f"Backtest metrics: {result.metrics}")
    notifier.send(
        "backtest_complete",
        f"回测完成 Sharpe={result.metrics.get('sharpe_ratio', 0):.3f} "
        f"MDD={result.metrics.get('max_drawdown', 0):.2%}",
        run_id=run_id,
        symbol=cfg.base.symbol,
        interval=cfg.base.interval,
        extra={"passed": result.passed_threshold},
    )


def cmd_publish(args, cfg):
    from etf_t0_quant.logger import get_logger

    log = get_logger("train")
    model_dir = Path(args.model_dir)
    pub_file = cfg.base.models_dir / "published_model.json"

    payload = {
        "version": model_dir.name,
        "path": str(model_dir),
        "published_at": __import__("datetime").datetime.now().isoformat(),
        "symbol": cfg.base.symbol,
        "interval": cfg.base.interval,
        "force": args.force,
    }
    pub_file.write_text(json.dumps(payload, indent=2))
    log.info(f"Published model: {model_dir.name}")
    print(f"Published: {model_dir}")


def cmd_infer(args, cfg):
    from etf_t0_quant.data_pipeline import DataPipeline
    from etf_t0_quant.trainer import ETFTrainer
    from etf_t0_quant.risk import RiskManager
    from etf_t0_quant.logger import get_logger

    log = get_logger("live")

    # Load published model
    pub_file = cfg.base.models_dir / "published_model.json"
    if not pub_file.exists():
        log.error("No published model. Run: python -m etf_t0_quant publish <dir>")
        sys.exit(1)

    pub = json.loads(pub_file.read_text())
    model = ETFTrainer.load_model(Path(pub["path"]) / "model")

    pipeline = DataPipeline(cfg)
    df_feat, df_prices, _ = pipeline.load_features()

    # Use last lookback_window bars
    lw = cfg.feature.lookback_window
    df_f = df_feat.iloc[-(lw + 10):]
    df_p = df_prices.iloc[-(lw + 10):]

    from etf_t0_quant.env import ETFTradingEnv
    env = ETFTradingEnv(df_f, df_p, cfg.env, cfg.feature)
    obs, _ = env.reset()

    # Fast-forward to last state
    while env._idx < env._end_idx - 1:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            break

    action, _ = model.predict(obs, deterministic=True)
    action = int(action)
    from etf_t0_quant.env import ACTION_NAMES
    log.info(f"Infer signal: {ACTION_NAMES.get(action, action)} at {df_p.index[-1]}")
    print(json.dumps({
        "timestamp": str(df_p.index[-1]),
        "symbol": cfg.base.symbol,
        "interval": cfg.base.interval,
        "action": ACTION_NAMES.get(action, str(action)),
        "model_version": pub["version"],
    }, ensure_ascii=False, indent=2))


def cmd_pipeline(args, cfg, run_id: str):
    cmd_data(args, cfg)
    cmd_train(args, cfg, run_id)


def cmd_dashboard(_args, cfg):
    import subprocess
    script = Path(__file__).parent / "dashboard.py"
    cfg_str = str(Path(__file__).parent / "configs" / "default.yaml")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(script), "--", "--config", cfg_str],
        check=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return

    from etf_t0_quant.logger import bind_run_id, new_run_id
    run_id = args.run_id or new_run_id()
    bind_run_id(run_id)

    cfg = _load(args.config)

    dispatch = {
        "data": lambda: cmd_data(args, cfg),
        "train": lambda: cmd_train(args, cfg, run_id),
        "backtest": lambda: cmd_backtest(args, cfg, run_id),
        "publish": lambda: cmd_publish(args, cfg),
        "infer": lambda: cmd_infer(args, cfg),
        "pipeline": lambda: cmd_pipeline(args, cfg, run_id),
        "dashboard": lambda: cmd_dashboard(args, cfg),
    }

    fn = dispatch.get(args.command)
    if fn:
        fn()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
