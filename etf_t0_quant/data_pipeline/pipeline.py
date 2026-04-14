"""Pipeline orchestrator for data fetch/clean/feature/save."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Tuple

import pandas as pd

from ..config import AppConfig
from ..logger import get_logger, get_run_id
from .cleaning import clean_ohlcv
from .features import build_features, normalize_features
from .fetchers import make_fetcher
from .metadata import build_metadata
from .quality import validate_ohlcv

log = get_logger("data")


class DataPipeline:
    """Orchestrates the full data engineering pipeline."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.fetcher = make_fetcher(cfg.data)
        self._symbol = cfg.base.symbol
        self._interval = cfg.base.interval

        base = cfg.base
        for d in (
            base.data_dir / "raw" / self._symbol / self._interval,
            base.data_dir / "cleaned" / self._symbol / self._interval,
            base.data_dir / "features" / self._symbol / self._interval,
            base.reports_dir,
            base.metadata_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def run(self, mode: str = "full") -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Run the pipeline and return features, aligned prices, and metadata."""
        run_id = get_run_id()
        log.info(f"DataPipeline.run mode={mode} symbol={self._symbol} interval={self._interval}")

        df_raw = self._fetch(mode)
        log.info(f"Fetched {len(df_raw)} raw bars")

        raw_path = self.cfg.base.data_dir / "raw" / self._symbol / self._interval / "data.parquet"
        df_raw.to_parquet(raw_path)

        report = validate_ohlcv(df_raw, self._interval)
        report.new_rows = len(df_raw)
        report_path = self.cfg.base.reports_dir / f"{run_id}_quality.json"
        report_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))

        if not report.passed:
            msg = f"Data quality FAILED: {report.fail_reasons}"
            log.error(msg)
            raise RuntimeError(msg)

        log.info(f"Data quality passed. Anomalies: {report.anomalies}")

        df_clean = clean_ohlcv(df_raw, self.cfg.base.timezone)
        clean_path = self.cfg.base.data_dir / "cleaned" / self._symbol / self._interval / "data.parquet"
        df_clean.to_parquet(clean_path)
        log.info(f"Cleaned data: {len(df_clean)} rows")

        df_feat_raw = build_features(df_clean, self.cfg.feature, self._interval)
        df_feat_norm = normalize_features(df_feat_raw, self.cfg.feature)

        feat_cols = [c for c in self.cfg.feature.feature_columns if c in df_feat_norm.columns]
        df_feat_norm = df_feat_norm.dropna(subset=feat_cols)
        df_prices_aligned = df_clean.loc[df_feat_norm.index]

        feat_path = self.cfg.base.data_dir / "features" / self._symbol / self._interval / "data.parquet"
        df_feat_norm.to_parquet(feat_path)
        log.info(f"Feature data: {len(df_feat_norm)} rows, {len(feat_cols)} feature columns")

        meta = build_metadata(df_raw, df_feat_norm, self.cfg, run_id, self.cfg.data.source)
        meta_path = self.cfg.base.metadata_dir / f"{run_id}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

        return df_feat_norm[feat_cols], df_prices_aligned, meta

    def load_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load pre-computed feature and price data from disk."""
        feat_path = self.cfg.base.data_dir / "features" / self._symbol / self._interval / "data.parquet"
        clean_path = self.cfg.base.data_dir / "cleaned" / self._symbol / self._interval / "data.parquet"
        if not feat_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feat_path}. Run pipeline first.")

        df_feat = pd.read_parquet(feat_path)
        df_prices = pd.read_parquet(clean_path)
        feat_cols = [c for c in self.cfg.feature.feature_columns if c in df_feat.columns]
        df_prices = df_prices.loc[df_feat.index]
        return df_feat[feat_cols], df_prices

    def _fetch(self, mode: str) -> pd.DataFrame:
        if mode == "full":
            return self.fetcher.fetch_full(self._symbol, self._interval)

        raw_path = self.cfg.base.data_dir / "raw" / self._symbol / self._interval / "data.parquet"
        if raw_path.exists():
            existing = pd.read_parquet(raw_path)
            since = existing.index.max().to_pydatetime()
        else:
            since = datetime.now() - timedelta(days=self.cfg.data.incremental_window_days)

        new_df = self.fetcher.fetch_incremental(self._symbol, self._interval, since)
        if raw_path.exists() and len(new_df) > 0:
            existing = pd.read_parquet(raw_path)
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            return combined
        return new_df
