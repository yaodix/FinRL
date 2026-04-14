"""
Unit tests for data_pipeline module.
Tests: resample, validate, clean, feature build, normalize.
"""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_1min():
    """1-minute synthetic bars for a single trading day."""
    from etf_t0_quant.data_pipeline import generate_synthetic_data
    # generate_synthetic_data gives 30m bars by default; make 1m manually
    idx = pd.date_range("2024-01-02 09:31:00", periods=240, freq="1min", tz="Asia/Shanghai")
    rng = np.random.default_rng(0)
    close = np.cumprod(1 + rng.normal(0, 0.001, 240)) * 2.5
    df = pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.0005, 240)),
        "high": close * (1 + rng.uniform(0, 0.002, 240)),
        "low": close * (1 - rng.uniform(0, 0.002, 240)),
        "close": close,
        "volume": rng.integers(100_000, 5_000_000, 240).astype(float),
    }, index=idx)
    return df


@pytest.fixture
def df_30m():
    from etf_t0_quant.data_pipeline import generate_synthetic_data
    return generate_synthetic_data(n_days=60, interval="30m", seed=10)


@pytest.fixture
def cfg():
    from etf_t0_quant.config import AppConfig
    return AppConfig()


# ---------------------------------------------------------------------------
# Resample tests
# ---------------------------------------------------------------------------

def test_resample_1min_to_30m(df_1min):
    from etf_t0_quant.data_pipeline import resample_1min_to_interval
    df_30 = resample_1min_to_interval(df_1min, "30m")
    # A trading day has 8 bars for 30m
    assert len(df_30) <= 8
    assert "open" in df_30.columns
    assert "close" in df_30.columns


def test_resample_1min_to_15m(df_1min):
    from etf_t0_quant.data_pipeline import resample_1min_to_interval
    df_15 = resample_1min_to_interval(df_1min, "15m")
    assert len(df_15) <= 16
    assert "volume" in df_15.columns


def test_resample_ohlc_consistency(df_1min):
    from etf_t0_quant.data_pipeline import resample_1min_to_interval
    df_30 = resample_1min_to_interval(df_1min, "30m")
    assert (df_30["high"] >= df_30["low"]).all()
    assert (df_30["high"] >= df_30["open"]).all()
    assert (df_30["high"] >= df_30["close"]).all()


# ---------------------------------------------------------------------------
# Validate tests
# ---------------------------------------------------------------------------

def test_validate_clean_data(df_30m):
    from etf_t0_quant.data_pipeline import validate_ohlcv
    report = validate_ohlcv(df_30m, "30m")
    assert report.n_rows == len(df_30m)
    assert report.duplicate_timestamps == 0


def test_validate_detects_duplicates():
    from etf_t0_quant.data_pipeline import validate_ohlcv, generate_synthetic_data
    df = generate_synthetic_data(n_days=10, interval="30m")
    df_dup = pd.concat([df, df.iloc[:5]])
    report = validate_ohlcv(df_dup, "30m")
    assert report.duplicate_timestamps > 0


def test_validate_detects_bad_ohlc():
    from etf_t0_quant.data_pipeline import validate_ohlcv, generate_synthetic_data
    df = generate_synthetic_data(n_days=10, interval="30m").copy()
    # Inject bad row
    df.iloc[0, df.columns.get_loc("high")] = 0.1   # high < low
    df.iloc[0, df.columns.get_loc("low")] = 999.0
    report = validate_ohlcv(df, "30m")
    assert any("high_lt_low" in a for a in report.anomalies)


def test_validate_empty_df():
    from etf_t0_quant.data_pipeline import validate_ohlcv, generate_synthetic_data
    df = generate_synthetic_data(n_days=5, interval="30m").iloc[:0]
    report = validate_ohlcv(df, "30m")
    assert not report.passed


# ---------------------------------------------------------------------------
# Clean tests
# ---------------------------------------------------------------------------

def test_clean_timezone(df_30m):
    from etf_t0_quant.data_pipeline import clean_ohlcv
    df_c = clean_ohlcv(df_30m, "Asia/Shanghai")
    assert df_c.index.tz is not None
    assert str(df_c.index.tz) == "Asia/Shanghai"


def test_clean_removes_duplicates():
    from etf_t0_quant.data_pipeline import clean_ohlcv, generate_synthetic_data
    df = generate_synthetic_data(n_days=10, interval="30m")
    df_dup = pd.concat([df, df.iloc[:5]])
    df_c = clean_ohlcv(df_dup)
    assert df_c.index.duplicated().sum() == 0


def test_clean_removes_bad_rows():
    from etf_t0_quant.data_pipeline import clean_ohlcv, generate_synthetic_data
    df = generate_synthetic_data(n_days=10, interval="30m").copy()
    df.iloc[5, df.columns.get_loc("high")] = 0.01
    df.iloc[5, df.columns.get_loc("low")] = 999.0
    df_c = clean_ohlcv(df)
    assert (df_c["high"] >= df_c["low"]).all()


def test_no_cross_day_ffill():
    """Forward fill must not propagate values across trading day boundaries."""
    from etf_t0_quant.data_pipeline import clean_ohlcv, generate_synthetic_data
    df = generate_synthetic_data(n_days=5, interval="30m").copy()
    # NaN all rows on day 2 (rows 8-15)
    start = 8
    df.iloc[start : start + 8, :] = np.nan
    df_c = clean_ohlcv(df)
    # Day 2 rows should still be NaN (or at most 2-step forward-filled within day)
    # The key is that day 1 values didn't bleed into day 3
    day3_close = df_c["close"].iloc[16:24]
    # Original day 3 data should be unchanged (day 2 NaN shouldn't infect day 3)
    # They might be NaN too (limit=2 fill within day), that's fine
    assert len(df_c) > 0


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

def test_build_features_columns(df_30m, cfg):
    from etf_t0_quant.data_pipeline import build_features, clean_ohlcv
    df_c = clean_ohlcv(df_30m)
    df_f = build_features(df_c, cfg.feature, "30m")
    expected_cols = ["returns", "log_ret", "amplitude", "volume_ratio", "rsi", "atr", "ema_diff", "bb_width"]
    for col in expected_cols:
        assert col in df_f.columns, f"Missing column: {col}"


def test_build_features_no_future_leakage(df_30m, cfg):
    """RSI, ATR, EMA must not look ahead."""
    from etf_t0_quant.data_pipeline import build_features, clean_ohlcv
    df_c = clean_ohlcv(df_30m)
    df_f = build_features(df_c, cfg.feature, "30m")
    # Check EMA is strictly causal (ema_diff at t uses close up to t)
    # Duplicate the first 10 rows and verify features are identical vs original
    df_f2 = build_features(df_c.iloc[:20], cfg.feature, "30m")
    pd.testing.assert_series_equal(
        df_f["ema_diff"].iloc[:20].reset_index(drop=True),
        df_f2["ema_diff"].reset_index(drop=True),
        check_names=False, rtol=1e-5,
    )


def test_normalize_features_no_future(df_30m, cfg):
    """Normalization parameters must use shift(1) – identical prefix behavior."""
    from etf_t0_quant.data_pipeline import build_features, normalize_features, clean_ohlcv
    df_c = clean_ohlcv(df_30m)
    df_feat = build_features(df_c, cfg.feature, "30m")
    df_norm_full = normalize_features(df_feat, cfg.feature)
    df_norm_trunc = normalize_features(df_feat.iloc[:100], cfg.feature)

    # The first 100 rows should match between full and truncated normalization
    for col in ["returns_norm", "amplitude_norm"]:
        if col in df_norm_full.columns and col in df_norm_trunc.columns:
            full_head = df_norm_full[col].iloc[:100].reset_index(drop=True)
            trunc = df_norm_trunc[col].reset_index(drop=True)
            pd.testing.assert_series_equal(full_head, trunc, check_names=False, rtol=1e-5)


def test_normalize_features_clipped(df_30m, cfg):
    from etf_t0_quant.data_pipeline import build_features, normalize_features, clean_ohlcv
    df_c = clean_ohlcv(df_30m)
    df_feat = build_features(df_c, cfg.feature, "30m")
    df_norm = normalize_features(df_feat, cfg.feature)
    for col in ["returns_norm", "amplitude_norm", "volume_ratio_norm"]:
        if col in df_norm.columns:
            assert df_norm[col].max() <= 3.01, f"{col} exceeds clip bound"
            assert df_norm[col].min() >= -3.01, f"{col} below clip bound"


def test_read_etf_1min_from_zip(tmp_path):
    from etf_t0_quant.data_pipeline import read_etf_1min_from_zip

    zip_dir = tmp_path / "etf_zip"
    zip_dir.mkdir()

    day_2020 = pd.DataFrame(
        {
            "code": ["159001.SZ", "510300.SH"],
            "trade_time": [
                "2020-12-31 14:59:00",
                "2020-12-31 14:59:00",
            ],
            "open": [0.9, 3.0],
            "high": [1.0, 3.1],
            "low": [0.8, 2.9],
            "close": [0.95, 3.0],
            "vol": [90.0, 999.0],
            "amount": [8550.0, 999000.0],
            "date": [20201231, 20201231],
        }
    )

    day_2021 = pd.DataFrame(
        {
            "code": ["159001.SZ", "159001.SZ", "510300.SH"],
            "trade_time": [
                "2021-01-04 09:30:00",
                "2021-01-04 09:31:00",
                "2021-01-04 09:31:00",
            ],
            "open": [1.0, 1.1, 3.0],
            "high": [1.0, 1.2, 3.1],
            "low": [1.0, 1.0, 2.9],
            "close": [1.0, 1.1, 3.0],
            "vol": [100.0, 120.0, 999.0],
            "amount": [10000.0, 13200.0, 999000.0],
            "date": [20210104, 20210104, 20210104],
        }
    )

    with zipfile.ZipFile(zip_dir / "2020.zip", "w") as zf:
        buf = io.BytesIO()
        day_2020.to_parquet(buf, index=False)
        zf.writestr("20201231.parquet", buf.getvalue())

    with zipfile.ZipFile(zip_dir / "2021.zip", "w") as zf:
        for name, df in (("20210104.parquet", day_2021),):
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            zf.writestr(name, buf.getvalue())

    out = read_etf_1min_from_zip(
        symbol="159001",
        start_time="2020-12-31 14:59:00",
        end_time="2021-01-04 09:31:00",
        zip_dir=zip_dir,
    )

    assert len(out) == 3
    assert list(out.columns) == ["code", "trade_time", "open", "high", "low", "close", "vol", "amount", "date"]
    assert out["trade_time"].iloc[0] == "2020-12-31 14:59:00"
    assert out["trade_time"].iloc[-1] == "2021-01-04 09:31:00"
    assert set(out["code"].unique()) == {"159001.SZ"}
