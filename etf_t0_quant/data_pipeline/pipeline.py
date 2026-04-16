"""Pipeline orchestrator for CSV-based data fetch/check/clean/feature/save."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

if __package__ in {None, ""}:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from etf_t0_quant.config import AppConfig
    from etf_t0_quant.data_pipeline.cleaning import clean_ohlcv
    from etf_t0_quant.data_pipeline.features import build_features, normalize_features
    from etf_t0_quant.data_pipeline.fetchers import LocalFileFetcher, TickflowFetcher
    from etf_t0_quant.data_pipeline.metadata import build_metadata
    from etf_t0_quant.data_pipeline.quality import validate_ohlcv
    from etf_t0_quant.logger import get_logger, get_run_id
else:
    from ..config import AppConfig
    from ..logger import get_logger, get_run_id
    from .cleaning import clean_ohlcv
    from .features import build_features, normalize_features
    from .fetchers import LocalFileFetcher, TickflowFetcher
    from .metadata import build_metadata
    from .quality import validate_ohlcv

log = get_logger("data")

_COLUMN_ALIASES = {
    "timestamp": "trade_time",
    "time": "trade_time",
    "datetime": "trade_time",
    "date": "trade_time",
    "代码": "symbol",
    "code": "symbol",
    "ticker": "symbol",
    "名称": "name",
    "name": "name",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "成交量": "volume",
    "vol": "volume",
    "成交额": "amount",
}
_DATA_COLUMNS = ["symbol", "name", "open", "high", "low", "close", "volume", "amount"]


class DataPipeline:
    """Orchestrates the full CSV-based data engineering pipeline."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._symbol = cfg.base.symbol
        self._interval = cfg.base.interval

        base = cfg.base
        dirs = [
            self._sync_dir,
            self._tickflow_cache_dir,
            self._clean_dir,
            self._feature_dir,
            base.reports_dir,
            base.metadata_dir,
        ]
        for d in dirs:
            target = d.parent if d.suffix else d
            target.mkdir(parents=True, exist_ok=True)

    @property
    def _sync_dir(self) -> Path:
        return self.cfg.base.data_dir / "sync_data" / f"{self._symbol}_sync_{self._interval}.csv"

    @property
    def _tickflow_cache_dir(self) -> Path:
        return self.cfg.base.data_dir / "tickflow_cache" 

    @property
    def _clean_dir(self) -> Path:
        return self.cfg.base.data_dir / "cleaned" 

    @property
    def _feature_dir(self) -> Path:
        return self.cfg.base.data_dir / "features" 



    @property
    def _clean_path(self) -> Path:
        return self._clean_dir / "clean_data.csv"

    @property
    def _feature_path(self) -> Path:
        return self._feature_dir / "feature_data.csv"

    def run(self, mode: str = "local") -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Run the pipeline and return features, aligned prices, and metadata."""
        if mode not in {"local", "incremental"}:
            raise ValueError(f"Unsupported pipeline mode: {mode}")

        run_id = get_run_id()
        log.info(f"DataPipeline.run mode={mode} symbol={self._symbol} interval={self._interval}")

        df_raw, new_rows = self._fetch(mode)
        log.info(f"Loaded {len(df_raw)} raw bars ({new_rows} new)")
        
        if mode == "incremental" and new_rows != 0:
            self._write_csv(df_raw, self._sync_dir)

        report = validate_ohlcv(df_raw, self._interval)
        report.new_rows = new_rows
        report_path = self.cfg.base.reports_dir / f"{run_id}_quality.json"
        report_path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")

        if not report.passed:
            msg = f"Data quality FAILED: {report.fail_reasons}"
            log.error(msg)
            raise RuntimeError(msg)

        log.info(f"Data quality passed. Anomalies: {report.anomalies}")

        # TODO： clean前后的信息对比输出
        df_clean = clean_ohlcv(df_raw, self.cfg.base.timezone)
        self._write_csv(df_clean, self._clean_path)
        log.info(f"Cleaned data: {len(df_clean)} rows")

        df_for_features = df_clean.copy()
        # df_for_features["trade_time"] = df_for_features.index.tz_localize(None) 
        df_feat_raw = build_features(df_for_features, self.cfg.feature, self._interval)
        df_feat_norm = normalize_features(df_feat_raw, self.cfg.feature)

        feat_cols = [c for c in self.cfg.feature.feature_columns if c in df_feat_norm.columns]
        if not feat_cols:
            raise RuntimeError("No configured feature columns were produced by the pipeline.")

        df_feat_norm = df_feat_norm.dropna(subset=feat_cols)
        df_prices_aligned = df_clean.loc[df_feat_norm.index]

        self._write_csv(df_feat_norm, self._feature_path)
        log.info(f"Feature data: {len(df_feat_norm)} rows, {len(feat_cols)} feature columns")

        meta = build_metadata(df_raw, df_feat_norm, self.cfg, run_id, self.cfg.data.source)
        meta_path = self.cfg.base.metadata_dir / f"{run_id}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

        return df_feat_norm[feat_cols], df_prices_aligned, meta

    def load_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load pre-computed feature and price data from disk."""
        if not self._feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {self._feature_path}. Run pipeline first.")
        if not self._clean_path.exists():
            raise FileNotFoundError(f"Clean data file not found: {self._clean_path}. Run pipeline first.")

        df_feat = self._read_pipeline_csv(self._feature_path)
        df_prices = self._read_pipeline_csv(self._clean_path)
        feat_cols = [c for c in self.cfg.feature.feature_columns if c in df_feat.columns]
        df_prices = df_prices.loc[df_feat.index]
        return df_feat[feat_cols], df_prices

    def _fetch(self, mode: str) -> Tuple[pd.DataFrame, int]:
        '''
        
        '''
        len_df = 0
        if mode == "local":
            fetcher = LocalFileFetcher(self._sync_dir)
            df = fetcher.fetch()
            len_df = len(df)
            return df, 0
        elif mode == "incremental":
            fetcher = TickflowFetcher(
                api_key = self.cfg.data.tickflow_API_KEY,
                local_data_path = self._sync_dir,
                raw_cache_dir = self._tickflow_cache_dir,
            )
            df = fetcher.fetch_incremental(self._symbol, self._interval, None, None)
            return df, len(df) - len_df
        else:
            raise ValueError(f"Unsupported fetch mode: {mode}")
    

    def _load_source_frame(self) -> pd.DataFrame:
        local_csv_path = self.cfg.data.local_csv_path
        if not local_csv_path:
            raise ValueError("cfg.data.local_csv_path is required for the CSV-only data pipeline.")

        source_path = Path(local_csv_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source CSV path not found: {source_path}")

        frames = [self._read_source_csv(path) for path in self._iter_csv_files(source_path)]
        if not frames:
            raise ValueError(f"No CSV files found under source path: {source_path}")

        df = pd.concat(frames, axis=0)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df

    def _iter_csv_files(self, source_path: Path) -> Iterable[Path]:
        if source_path.is_file():
            if source_path.suffix.lower() != ".csv":
                raise ValueError(f"CSV-only pipeline does not support non-CSV source: {source_path}")
            return [source_path]
        return sorted(p for p in source_path.glob("*.csv") if p.is_file())

    def _read_source_csv(self, csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        return self._normalize_source_frame(df, csv_path)

    def _normalize_source_frame(self, df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=_DATA_COLUMNS)

        renamed = df.rename(columns={col: _COLUMN_ALIASES.get(col, col) for col in df.columns})
        if "trade_time" not in renamed.columns:
            raise ValueError(f"Missing trade_time column in source CSV: {source_path}")

        renamed["trade_time"] = pd.to_datetime(renamed["trade_time"], errors="coerce")
        renamed = renamed.dropna(subset=["trade_time"])
        if renamed.empty:
            raise ValueError(f"No valid trade_time rows in source CSV: {source_path}")

        if "symbol" in renamed.columns:
            mask = renamed["symbol"].astype(str).str.startswith(self._symbol)
            if mask.any():
                renamed = renamed.loc[mask]
        else:
            renamed["symbol"] = self._symbol

        for column in ("open", "high", "low", "close", "volume", "amount"):
            if column in renamed.columns:
                renamed[column] = pd.to_numeric(renamed[column], errors="coerce")

        required = ["open", "high", "low", "close"]
        missing = [column for column in required if column not in renamed.columns]
        if missing:
            raise ValueError(f"Missing required OHLC columns {missing} in source CSV: {source_path}")

        columns = [column for column in _DATA_COLUMNS if column in renamed.columns]
        normalized = renamed[["trade_time", *columns]].dropna(subset=required)
        normalized = normalized.set_index("trade_time")
        normalized.index.name = "trade_time"
        return normalized.sort_index()

    def _read_pipeline_csv(self, csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path, parse_dates=["trade_time"])
        if "trade_time" not in df.columns:
            raise ValueError(f"Pipeline CSV missing trade_time column: {csv_path}")
        df = df.set_index("trade_time")
        df.index.name = "trade_time"
        return df.sort_index()

    def _write_csv(self, df: pd.DataFrame, csv_path: Path) -> None:
        out = df.copy()
        if "trade_time" in out.columns:
            out = out.drop(columns=["trade_time"])
        out.index.name = "trade_time"
        out = out.reset_index()
        if pd.api.types.is_datetime64tz_dtype(out["trade_time"]):
            out["trade_time"] = out["trade_time"].dt.tz_localize(None)
        out.to_csv(csv_path, index=False)


if __name__ == "__main__":
    cfg = AppConfig()
    pipeline = DataPipeline(cfg)
    features, prices, meta = pipeline.run(mode="local")
    print("Features:")
    print(features.head())
    print("\nAligned Prices:")
    print(prices.head())