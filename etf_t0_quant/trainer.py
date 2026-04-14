"""
Model training module: DQN / Double-DQN via stable-baselines3.

Features (doc 04_模型和训练设置):
  - DQN with MLP Q-network
  - Optuna hyperparameter search (primary)
  - Walk-Forward training loop
  - MLflow experiment tracking
  - Reproducible (seed, config snapshot)
  - Model versioning: saved under models/{symbol}/{interval}/{run_id}/

Usage::

    trainer = ETFTrainer(cfg)
    model = trainer.train_single(df_feat_train, df_prices_train)
    trainer.save_model(model, run_id="abc123")

    # With Optuna search
    best_params = trainer.run_optuna_search(df_feat_train, df_prices_train,
                                             df_feat_val, df_prices_val)

    # Full Walk-Forward (calls trainer internally)
    from etf_t0_quant.backtester import WalkForwardBacktester
    wf = WalkForwardBacktester(cfg)
    results = wf.run(trainer.make_trainer_fn(), df_features, df_prices)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import AppConfig, TrainConfig
from .env import ETFTradingEnv
from .logger import get_logger, get_run_id

log = get_logger("train")

try:
    from stable_baselines3 import DQN  # type: ignore
    from stable_baselines3.common.callbacks import EvalCallback  # type: ignore
    from stable_baselines3.common.monitor import Monitor  # type: ignore
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False
    log.warning("stable-baselines3 not installed. Training will fail.")

try:
    import mlflow  # type: ignore
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

try:
    import optuna  # type: ignore
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model version helpers
# ---------------------------------------------------------------------------

def model_version_tag(run_id: str) -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{run_id[:8]}"


def model_dir(cfg: AppConfig, run_id: str) -> Path:
    d = (
        cfg.base.models_dir
        / cfg.base.symbol
        / cfg.base.interval
        / model_version_tag(run_id)
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Build environment from data slices
# ---------------------------------------------------------------------------

def make_env(
    df_features: pd.DataFrame,
    df_prices: pd.DataFrame,
    cfg: AppConfig,
    seed: int = 42,
) -> ETFTradingEnv:
    env = ETFTradingEnv(df_features, df_prices, cfg.env, cfg.feature)
    return env


# ---------------------------------------------------------------------------
# ETFTrainer
# ---------------------------------------------------------------------------

class ETFTrainer:
    """Train DQN models and manage experiments."""

    def __init__(self, cfg: AppConfig) -> None:
        if not _SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required. pip install stable-baselines3")
        self.cfg = cfg
        self._run_id = get_run_id()

    # ------------------------------------------------------------------
    # Single training run
    # ------------------------------------------------------------------

    def train_single(
        self,
        df_feat_train: pd.DataFrame,
        df_prices_train: pd.DataFrame,
        overrides: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ):
        """Train DQN and return the trained model."""
        tc = self.cfg.train
        rid = run_id or get_run_id()
        params = self._build_params(tc, overrides or {})

        log.info(f"Training start run_id={rid} params={params}")

        env = make_env(df_feat_train, df_prices_train, self.cfg, seed=params["seed"])

        policy_kwargs = {"net_arch": list(tc.net_arch)}

        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
            gamma=params["gamma"],
            buffer_size=params["replay_buffer_size"],
            target_update_interval=params["target_update_interval"],
            exploration_fraction=params["exploration_fraction"],
            exploration_final_eps=params["exploration_final_eps"],
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=params["seed"],
            # double_q removed in SB3 v2.x – Double DQN is the default
        )

        if _MLFLOW_AVAILABLE:
            mlflow.set_experiment(
                f"etf_t0_{self.cfg.base.symbol}_{self.cfg.base.interval}"
            )
            with mlflow.start_run(run_name=rid):
                mlflow.log_params(params)
                model.learn(total_timesteps=params["total_timesteps"])
                train_nav = self._eval_nav(model, env)
                mlflow.log_metric("train_final_nav_return", train_nav)
                log.info(f"MLflow logged for run {rid}")
        else:
            model.learn(total_timesteps=params["total_timesteps"])

        log.info(f"Training done run_id={rid}")
        return model

    # ------------------------------------------------------------------
    # Optuna hyperparameter search
    # ------------------------------------------------------------------

    def run_optuna_search(
        self,
        df_feat_train: pd.DataFrame,
        df_prices_train: pd.DataFrame,
        df_feat_val: pd.DataFrame,
        df_prices_val: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Search hyperparameters using Optuna.  Returns best params dict."""
        if not _OPTUNA_AVAILABLE:
            log.warning("Optuna not installed – skipping hyperparameter search")
            return {}

        from .backtester import Backtester, compute_metrics

        study = optuna.create_study(
            direction="maximize",
            study_name=f"etf_t0_{self.cfg.base.symbol}",
            sampler=optuna.samplers.TPESampler(seed=self.cfg.train.seed),
        )

        # During HPO use a relaxed RiskManager so early-exploration losses
        # don't trigger the 5 % halt and return non-informative sharpe=0.
        from .risk import RiskManager as _RM
        _train_rm = _RM(
            self.cfg.risk,
            symbol=self.cfg.base.symbol,
            interval=self.cfg.base.interval,
            training_mode=True,
        )
        backtester = Backtester(self.cfg, risk_manager=_train_rm)

        def objective(trial: "optuna.Trial") -> float:
            overrides = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "gamma": trial.suggest_categorical("gamma", [0.95, 0.99, 0.995]),
                "replay_buffer_size": trial.suggest_categorical(
                    "replay_buffer_size", [20_000, 50_000, 100_000]
                ),
                "target_update_interval": trial.suggest_categorical(
                    "target_update_interval", [500, 1000, 2000]
                ),
                "exploration_fraction": trial.suggest_float("exploration_fraction", 0.1, 0.5),
            }
            try:
                model = self.train_single(df_feat_train, df_prices_train, overrides)
                result = backtester.run(model, df_feat_val, df_prices_val, window_label="optuna_val")
                sharpe = result.metrics.get("sharpe_ratio", -999.0)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"Trial {trial.number} failed: {exc}")
                return -999.0
            return sharpe

        tc = self.cfg.train
        study.optimize(
            objective,
            n_trials=tc.n_trials,
            timeout=tc.optuna_timeout,
            show_progress_bar=False,
        )

        best = study.best_params
        log.info(f"Optuna best params: {best} | best_value={study.best_value:.4f}")
        return best

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def save_model(self, model, run_id: Optional[str] = None) -> Path:
        rid = run_id or get_run_id()
        d = model_dir(self.cfg, rid)
        model_path = d / "model"
        model.save(str(model_path))

        # Save config snapshot
        cfg_snap = self.cfg.model_dump() if hasattr(self.cfg, "model_dump") else {}
        (d / "config.json").write_text(json.dumps(cfg_snap, indent=2, default=str))

        log.info(f"Model saved to {d}")
        return d

    @staticmethod
    def load_model(model_path: str | Path):
        if not _SB3_AVAILABLE:
            raise ImportError("stable-baselines3 required")
        return DQN.load(str(model_path))

    # ------------------------------------------------------------------
    # Factory for WalkForwardBacktester
    # ------------------------------------------------------------------

    def make_trainer_fn(
        self, best_params: Optional[Dict[str, Any]] = None
    ) -> Callable[[pd.DataFrame, pd.DataFrame], Any]:
        """Return a trainer_fn callable compatible with WalkForwardBacktester."""
        overrides = best_params or {}

        def _train(df_feat: pd.DataFrame, df_prices: pd.DataFrame):
            return self.train_single(df_feat, df_prices, overrides=overrides)

        return _train

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_params(tc: TrainConfig, overrides: Dict[str, Any]) -> Dict[str, Any]:
        base = {
            "learning_rate": tc.learning_rate,
            "batch_size": tc.batch_size,
            "gamma": tc.gamma,
            "replay_buffer_size": tc.replay_buffer_size,
            "target_update_interval": tc.target_update_interval,
            "exploration_fraction": tc.exploration_fraction,
            "exploration_final_eps": tc.exploration_final_eps,
            "total_timesteps": tc.total_timesteps,
            "seed": tc.seed,
        }
        base.update(overrides)
        return base

    @staticmethod
    def _eval_nav(model, env: ETFTradingEnv) -> float:
        """Quick evaluation: return cumulative NAV return on env."""
        obs, _ = env.reset()
        done = False
        nav0 = env.cfg.initial_cash if hasattr(env, "cfg") else env._cfg.initial_cash
        nav_final = nav0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            nav_final = info.get("nav", nav_final)
            done = terminated or truncated
        return nav_final / nav0 - 1
