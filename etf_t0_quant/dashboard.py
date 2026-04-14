"""
Streamlit dashboard for ETF T0 Quant (doc 09_可视化和推送模块).

Run with::

    streamlit run etf_t0_quant/dashboard.py -- --config etf_t0_quant/configs/default.yaml

Pages:
  1. 数据概览   – latest data update, sample counts, anomalies
  2. 训练概览   – recent training runs, key params, best model
  3. 回测结果   – NAV curve, drawdown, trade log, metrics table
  4. 发布状态   – published model, threshold status
  5. 运行监控   – recent signals, risk events, system health
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

# Add repo root to sys.path when run directly
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Missing dependency: {e}. Install: pip install streamlit matplotlib pandas")
    sys.exit(1)

from etf_t0_quant.config import load_config

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

@st.cache_data
def _load_cfg():
    args = sys.argv
    cfg_path = None
    for i, a in enumerate(args):
        if a == "--config" and i + 1 < len(args):
            cfg_path = args[i + 1]
            break
    try:
        return load_config(cfg_path)
    except Exception:
        return load_config()


cfg = _load_cfg()
BASE_DIR = Path(cfg.base.base_dir)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json_files(pattern: str) -> list:
    files = sorted(glob.glob(pattern), reverse=True)
    results = []
    for f in files[:20]:
        try:
            results.append(json.loads(Path(f).read_text()))
        except Exception:
            pass
    return results


def _load_features() -> pd.DataFrame | None:
    p = (
        BASE_DIR / "data" / "features"
        / cfg.base.symbol / cfg.base.interval / "data.parquet"
    )
    if p.exists():
        return pd.read_parquet(p)
    return None


def _load_backtest_reports() -> list:
    pattern = str(BASE_DIR / "backtests" / "walk_forward" / "*_report.json")
    return _read_json_files(pattern)


def _load_quality_reports() -> list:
    pattern = str(BASE_DIR / "data" / "reports" / "*_quality.json")
    return _read_json_files(pattern)


def _load_metadata() -> list:
    pattern = str(BASE_DIR / "data" / "metadata" / "*_meta.json")
    return _read_json_files(pattern)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_data_overview():
    st.title("📊 数据概览")

    reports = _load_quality_reports()
    metas = _load_metadata()

    col1, col2 = st.columns(2)
    if metas:
        latest = metas[0]
        col1.metric("原始行数", latest.get("raw_rows", "–"))
        col1.metric("特征行数", latest.get("feature_rows", "–"))
        col2.metric("数据来源", latest.get("source", "–"))
        col2.metric("更新时间", latest.get("created_at", "–")[:19] if latest.get("created_at") else "–")
        st.markdown(f"**时间范围**: {latest.get('raw_time_range', ['–', '–'])}")
    else:
        st.info("暂无元数据，请先运行数据管道。")

    if reports:
        st.subheader("最新数据质量报告")
        r = reports[0]
        st.json({k: v for k, v in r.items() if k not in ("columns",)})
    else:
        st.info("暂无质量报告。")

    df = _load_features()
    if df is not None:
        st.subheader("特征数据预览")
        st.dataframe(df.tail(20), use_container_width=True)


def page_train_overview():
    st.title("🏋️ 训练概览")

    model_dirs = sorted(
        glob.glob(str(BASE_DIR / "models" / cfg.base.symbol / cfg.base.interval / "*" / "config.json")),
        reverse=True,
    )
    if not model_dirs:
        st.info("暂无训练记录。请先运行: python -m etf_t0_quant train")
        return

    rows = []
    for p in model_dirs[:10]:
        try:
            d = json.loads(Path(p).read_text())
            rows.append({
                "version": Path(p).parent.name,
                "symbol": d.get("base", {}).get("symbol", ""),
                "interval": d.get("base", {}).get("interval", ""),
                "model_type": d.get("train", {}).get("model_type", ""),
                "total_steps": d.get("train", {}).get("total_timesteps", ""),
            })
        except Exception:
            pass

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption("模型文件路径: " + str(BASE_DIR / "models"))


def page_backtest_results():
    st.title("📈 回测结果")

    reports = _load_backtest_reports()
    if not reports:
        st.info("暂无回测结果。请先运行: python -m etf_t0_quant backtest")
        return

    st.subheader("指标汇总")
    rows = []
    for r in reports:
        row = {"window": r.get("window_label", ""), "passed": r.get("passed_threshold", False)}
        row.update(r.get("metrics", {}))
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("window"), use_container_width=True)

    # Find and display chart PNGs
    chart_files = sorted(
        glob.glob(str(BASE_DIR / "backtests" / "walk_forward" / "*_chart.png")),
        reverse=True,
    )
    if chart_files:
        st.subheader("最近回测图表")
        for p in chart_files[:3]:
            st.image(p, use_container_width=True)


def page_publish_status():
    st.title("🚀 发布状态")

    # Look for a "published_model.json" pointer file
    pub_file = BASE_DIR / "models" / "published_model.json"
    if pub_file.exists():
        d = json.loads(pub_file.read_text())
        st.success(f"当前已发布模型: {d.get('version', '未知')}")
        st.json(d)
    else:
        st.warning("尚无已发布模型。需通过 `python -m etf_t0_quant publish <model_dir>` 手动发布。")

    reports = _load_backtest_reports()
    passing = [r for r in reports if r.get("passed_threshold")]
    st.metric("达标窗口数", len(passing), f"/ {len(reports)} total")


def page_monitoring():
    st.title("🖥️ 运行监控")

    log_dir = BASE_DIR / "logs"
    if not log_dir.exists():
        st.info("暂无日志。")
        return

    for log_file in sorted(log_dir.glob("*.log"), reverse=True)[:5]:
        with st.expander(f"📄 {log_file.name}"):
            lines = log_file.read_text(errors="replace").strip().split("\n")
            st.code("\n".join(lines[-50:]))


# ---------------------------------------------------------------------------
# Main navigation
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="ETF T0 Quant",
        page_icon="📡",
        layout="wide",
    )

    pages = {
        "数据概览": page_data_overview,
        "训练概览": page_train_overview,
        "回测结果": page_backtest_results,
        "发布状态": page_publish_status,
        "运行监控": page_monitoring,
    }

    with st.sidebar:
        st.title("ETF T0 Quant")
        st.caption(f"标的: {cfg.base.symbol}  周期: {cfg.base.interval}")
        selected = st.radio("导航", list(pages.keys()))

    pages[selected]()


if __name__ == "__main__":
    main()
