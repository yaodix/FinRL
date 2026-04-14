"""
Feishu (Lark) webhook notifier.

Usage::

    notifier = FeishuNotifier(cfg.notifier)
    notifier.send("train_complete", "Walk-forward training done", run_id="abc123")

Design constraints (doc 09):
  - Failure MUST NOT block training / backtest main flow.
  - High-frequency signals should NOT be sent one-by-one; caller is responsible
    for throttling.
  - Critical alerts should repeat until acknowledged (caller responsibility).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from .config import NotifierConfig
from .logger import get_logger

log = get_logger("live")

_EVENT_ICONS = {
    "data_ok": "✅",
    "data_fail": "❌",
    "train_complete": "🏋️",
    "backtest_complete": "📊",
    "model_candidate": "🚀",
    "risk_warning": "⚠️",
    "risk_error": "🔴",
    "risk_critical": "🚨",
    "signal": "📡",
    "order_result": "📋",
    "default": "ℹ️",
}


class FeishuNotifier:
    """Send structured messages to a Feishu (Lark) incoming webhook."""

    def __init__(self, cfg: NotifierConfig) -> None:
        self._enabled = cfg.enabled
        self._webhook = cfg.feishu_webhook
        self._session = None  # lazy-imported requests.Session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(
        self,
        event_type: str,
        message: str,
        *,
        run_id: str = "",
        symbol: str = "",
        interval: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send a notification.  Returns True on success.  Never raises."""
        if not self._enabled or not self._webhook:
            return False
        try:
            return self._post(event_type, message, run_id, symbol, interval, extra or {})
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Feishu notification failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(
        self,
        event_type: str,
        message: str,
        run_id: str,
        symbol: str,
        interval: str,
        extra: Dict[str, Any],
    ) -> bool:
        import requests  # type: ignore

        icon = _EVENT_ICONS.get(event_type, _EVENT_ICONS["default"])
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        lines = [
            f"{icon} **{event_type.upper()}**",
            f"时间: {ts}",
        ]
        if symbol:
            lines.append(f"标的: {symbol}  周期: {interval}")
        if run_id:
            lines.append(f"run_id: `{run_id}`")
        lines.append("")
        lines.append(message)
        for k, v in extra.items():
            lines.append(f"- {k}: {v}")

        payload = {
            "msg_type": "text",
            "content": {"text": "\n".join(lines)},
        }

        resp = requests.post(
            self._webhook,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        resp.raise_for_status()
        return True
