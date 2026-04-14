"""
Structured logging module for ETF T0 Quant System.

Every module obtains its logger via get_logger(module_name).
Each new pipeline run should call new_run_id() to obtain a fresh
run_id, then bind_run_id(run_id) so all subsequent log entries
carry run_id automatically.

Log files:
    logs/data.log
    logs/feature.log
    logs/train.log
    logs/backtest.log
    logs/risk.log
    logs/live.log
"""
from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path
from typing import Optional

try:
    from loguru import logger as _loguru_logger  # type: ignore
    _USE_LOGURU = True
except ImportError:  # pragma: no cover
    _USE_LOGURU = False

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_run_id: str = "init"
_logs_dir: Path = Path("etf_t0_quant/logs")
_initialized: bool = False

MODULE_FILES = {
    "data": "data.log",
    "feature": "feature.log",
    "train": "train.log",
    "backtest": "backtest.log",
    "risk": "risk.log",
    "live": "live.log",
}


def new_run_id() -> str:
    """Generate a fresh UUID4 run_id."""
    return uuid.uuid4().hex[:12]


def get_run_id() -> str:
    return _run_id


def bind_run_id(run_id: str) -> None:
    global _run_id
    _run_id = run_id


def setup_logging(logs_dir: Path | str, run_id: Optional[str] = None) -> None:
    """Initialize log sinks. Call once per process start."""
    global _logs_dir, _run_id, _initialized
    _logs_dir = Path(logs_dir)
    _logs_dir.mkdir(parents=True, exist_ok=True)
    if run_id:
        _run_id = run_id
    _initialized = True

    if _USE_LOGURU:
        _loguru_logger.remove()
        # Console sink
        _loguru_logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{extra[module]: <10}</cyan> | "
                   "{message}",
        )
        # Per-module file sinks
        for mod, filename in MODULE_FILES.items():
            _loguru_logger.add(
                str(_logs_dir / filename),
                level="DEBUG",
                rotation="00:00",
                retention="30 days",
                filter=lambda rec, m=mod: rec["extra"].get("module") == m,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | run={extra[run_id]} | "
                       "sym={extra[symbol]} | {message}",
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class _StdlibAdapter:
    """Thin adapter around stdlib logging when loguru is not installed."""

    def __init__(self, module: str) -> None:
        self._log = logging.getLogger(f"etf_t0.{module}")
        self._module = module
        if not self._log.handlers:
            h = logging.StreamHandler(sys.stderr)
            h.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
            )
            self._log.addHandler(h)
            self._log.setLevel(logging.DEBUG)

    def _fmt(self, msg: str, **kw) -> str:
        kw_str = " ".join(f"{k}={v}" for k, v in kw.items())
        prefix = f"[run={_run_id}]"
        return f"{prefix} {msg} {kw_str}".strip()

    def debug(self, msg: str, **kw) -> None:
        self._log.debug(self._fmt(msg, **kw))

    def info(self, msg: str, **kw) -> None:
        self._log.info(self._fmt(msg, **kw))

    def warning(self, msg: str, **kw) -> None:
        self._log.warning(self._fmt(msg, **kw))

    def error(self, msg: str, **kw) -> None:
        self._log.error(self._fmt(msg, **kw))

    def critical(self, msg: str, **kw) -> None:
        self._log.critical(self._fmt(msg, **kw))


class _LoguruAdapter:
    """Loguru-backed logger bound to a specific module."""

    def __init__(self, module: str) -> None:
        self._module = module
        self._logger = _loguru_logger.bind(module=module, run_id=_run_id, symbol="")

    def _bound(self):
        return _loguru_logger.bind(module=self._module, run_id=_run_id, symbol="")

    def debug(self, msg: str, **kw) -> None:
        self._bound().debug(msg, **kw)

    def info(self, msg: str, **kw) -> None:
        self._bound().info(msg, **kw)

    def warning(self, msg: str, **kw) -> None:
        self._bound().warning(msg, **kw)

    def error(self, msg: str, **kw) -> None:
        self._bound().error(msg, **kw)

    def critical(self, msg: str, **kw) -> None:
        self._bound().critical(msg, **kw)


def get_logger(module: str):
    """Return a logger for *module* (one of the MODULE_FILES keys or any string)."""
    if _USE_LOGURU:
        return _LoguruAdapter(module)
    return _StdlibAdapter(module)
