"""
QMT Gateway – interface stub (doc 07_实盘qmt交易 – qmt_gateway).

THIS MODULE IS A STUB.  Real QMT execution code must NOT be added until:
  1. Walk-Forward validation is complete.
  2. Simulated inference has been running for ≥ N trading days.
  3. Risk review is signed off manually.

The gateway defines the contract that any real QMT integration must satisfy.
All methods raise NotImplementedError unless explicitly noted as safe to run
in simulation mode.

Design principles:
  - Research / training code MUST NOT import this module.
  - qmt_gateway is the ONLY module that talks to the QMT API.
  - account_sync, execution_logger are separate concerns handled here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from etf_t0_quant.config import AppConfig
from etf_t0_quant.logger import get_logger

log = get_logger("live")


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class AccountState:
    """Snapshot of the trading account."""
    timestamp: datetime = field(default_factory=datetime.now)
    cash: float = 0.0
    position_shares: float = 0.0
    position_cost: float = 0.0   # average cost basis
    nav: float = 0.0
    frozen_cash: float = 0.0


@dataclass
class Order:
    order_id: str = ""
    symbol: str = ""
    action: str = ""       # "buy" or "sell"
    price: float = 0.0
    quantity: float = 0.0
    status: str = "pending"  # pending / filled / cancelled / partial
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    error_msg: str = ""


# ---------------------------------------------------------------------------
# QMT Gateway stub
# ---------------------------------------------------------------------------

class QMTGateway:
    """
    QMT broker gateway interface.

    All methods annotated '# STUB' raise NotImplementedError.
    Methods annotated '# SIM' work in simulation mode (no real orders).
    """

    def __init__(self, cfg: AppConfig, simulation: bool = True) -> None:
        self.cfg = cfg
        self.simulation = simulation
        self._orders: Dict[str, Order] = {}  # SIM order store

        if not simulation:
            log.critical(
                "QMTGateway initialized in LIVE mode. "
                "Ensure QMT SDK is installed and account reviewed."
            )

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account_state(self) -> AccountState:  # STUB
        """Query account cash, position, and NAV from QMT."""
        if self.simulation:
            return AccountState(
                cash=self.cfg.env.initial_cash,
                nav=self.cfg.env.initial_cash,
                timestamp=datetime.now(),
            )
        raise NotImplementedError(
            "QMT live account query not implemented. "
            "Integrate xtquant.xttrader here."
        )

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_order(self, signal: Dict) -> Order:  # STUB (partially SIM)
        """Convert a validated signal into a QMT order and submit it."""
        if self.simulation:
            return self._sim_place_order(signal)
        raise NotImplementedError(
            "QMT live order submission not implemented. "
            "Call xttrader.order_stock() here."
        )

    def cancel_order(self, order_id: str) -> bool:  # STUB
        if self.simulation:
            if order_id in self._orders:
                self._orders[order_id].status = "cancelled"
                return True
            return False
        raise NotImplementedError("QMT cancel_order not implemented.")

    def get_order_status(self, order_id: str) -> Order:  # STUB
        if self.simulation:
            return self._orders.get(order_id, Order(order_id=order_id, status="unknown"))
        raise NotImplementedError("QMT get_order_status not implemented.")

    def sync_positions(self) -> List[Dict]:  # STUB
        """Return list of current positions from QMT."""
        if self.simulation:
            return []
        raise NotImplementedError("QMT sync_positions not implemented.")

    # ------------------------------------------------------------------
    # Simulation helpers (SIM mode only)
    # ------------------------------------------------------------------

    def _sim_place_order(self, signal: Dict) -> Order:
        """Create a simulated order without touching real QMT."""
        import uuid
        oid = uuid.uuid4().hex[:12]
        order = Order(
            order_id=oid,
            symbol=signal.get("symbol", ""),
            action=signal.get("action", "hold"),
            price=0.0,  # market order – price unknown
            quantity=0.0,
            status="sim_pending",
            timestamp=datetime.now(),
        )
        self._orders[oid] = order
        log.info(
            f"[SIM] Order placed: id={oid} action={order.action} "
            f"symbol={order.symbol}"
        )
        return order

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:  # STUB
        if self.simulation:
            log.info("[SIM] QMTGateway connected (simulation mode)")
            return
        raise NotImplementedError(
            "QMT connection not implemented. "
            "Set up xtquant.xtclient.XtQuantTraderCallback and call connect()."
        )

    def disconnect(self) -> None:
        if self.simulation:
            log.info("[SIM] QMTGateway disconnected")
            return
        # Real disconnect would call xttrader.disconnect()
        log.info("QMT disconnected")
