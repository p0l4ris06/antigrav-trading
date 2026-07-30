"""
Paper Trading Engine for Antigravity.

Manages virtual account balance ($100,000 initial), order execution,
live position tracking, risk management (SL/TP), and PnL metrics for
both automated bot trades and human manual orders.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PaperOrder(BaseModel):
  id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
  symbol: str
  side: str  # 'BUY' | 'SELL'
  order_type: str = 'MARKET'  # 'MARKET' | 'LIMIT'
  quantity: float
  price: float = 0.0
  stop_loss: Optional[float] = None
  take_profit: Optional[float] = None
  status: str = 'OPEN'  # 'OPEN' | 'FILLED' | 'CANCELLED' | 'REJECTED'
  source: str = 'MANUAL'  # 'MANUAL' | 'BOT'
  timestamp: str = Field(
      default_factory=lambda: datetime.now(timezone.utc).isoformat()
  )


class PaperPosition(BaseModel):
  symbol: str
  side: str  # 'LONG' | 'SHORT'
  size: float
  entry_price: float
  current_price: float
  unrealized_pnl: float = 0.0
  unrealized_pnl_pct: float = 0.0
  realized_pnl: float = 0.0
  margin_used: float = 0.0
  stop_loss: Optional[float] = None
  take_profit: Optional[float] = None
  opened_at: str = Field(
      default_factory=lambda: datetime.now(timezone.utc).isoformat()
  )
  opened_timestamp_ms: float = Field(
      default_factory=lambda: time.time() * 1000
  )


class PaperAccountSummary(BaseModel):
  cash_balance: float = 100_000.0
  equity: float = 100_000.0
  realized_pnl: float = 0.0
  unrealized_pnl: float = 0.0
  margin_used: float = 0.0
  win_rate: float = 0.0
  total_trades: int = 0
  winning_trades: int = 0
  positions: List[PaperPosition] = []
  open_orders: List[PaperOrder] = []


class PaperTradingEngine:
  """In-memory paper trading simulation engine with order matching & PnL tracking."""

  def __init__(self, initial_balance: float = 100_000.0):
    self.initial_balance = initial_balance
    self.cash_balance = initial_balance
    self.realized_pnl = 0.0
    self.total_trades = 0
    self.winning_trades = 0

    self.positions: Dict[str, PaperPosition] = {}
    self.open_orders: List[PaperOrder] = []
    self.trade_history: List[Dict[str, Any]] = []
    self.reloop_buffer: List[Dict[str, Any]] = []

  def get_summary(self) -> PaperAccountSummary:
    unrealized_total = sum(p.unrealized_pnl for p in self.positions.values())
    margin_total = sum(p.margin_used for p in self.positions.values())
    equity = self.cash_balance + unrealized_total
    win_rate = (
        (self.winning_trades / self.total_trades * 100)
        if self.total_trades > 0
        else 0.0
    )

    return PaperAccountSummary(
        cash_balance=round(self.cash_balance, 2),
        equity=round(equity, 2),
        realized_pnl=round(self.realized_pnl, 2),
        unrealized_pnl=round(unrealized_total, 2),
        margin_used=round(margin_total, 2),
        win_rate=round(win_rate, 1),
        total_trades=self.total_trades,
        winning_trades=self.winning_trades,
        positions=list(self.positions.values()),
        open_orders=self.open_orders,
    )

  def on_tick(self, symbol: str, last_price: float) -> None:
    """Updates position mark prices, unrealized PnL, and checks TP/SL triggers."""
    if symbol not in self.positions:
      return

    pos = self.positions[symbol]
    pos.current_price = last_price

    # Calculate unrealized PnL
    if pos.side == 'LONG':
      pnl = (last_price - pos.entry_price) * pos.size
    else:
      pnl = (pos.entry_price - last_price) * pos.size

    pos.unrealized_pnl = round(pnl, 2)
    pos.unrealized_pnl_pct = (
        round((pnl / (pos.entry_price * pos.size)) * 100, 2)
        if pos.entry_price > 0
        else 0.0
    )

    # Check Stop Loss / Take Profit triggers
    if pos.side == 'LONG':
      if pos.stop_loss and last_price <= pos.stop_loss:
        logger.info(
            f"SL triggered for {symbol} LONG at {last_price} (SL:"
            f" {pos.stop_loss})"
        )
        self.close_position(symbol, last_price, reason='STOP_LOSS')
      elif pos.take_profit and last_price >= pos.take_profit:
        logger.info(
            f"TP triggered for {symbol} LONG at {last_price} (TP:"
            f" {pos.take_profit})"
        )
        self.close_position(symbol, last_price, reason='TAKE_PROFIT')
    elif pos.side == 'SHORT':
      if pos.stop_loss and last_price >= pos.stop_loss:
        logger.info(
            f"SL triggered for {symbol} SHORT at {last_price} (SL:"
            f" {pos.stop_loss})"
        )
        self.close_position(symbol, last_price, reason='STOP_LOSS')
      elif pos.take_profit and last_price <= pos.take_profit:
        logger.info(
            f"TP triggered for {symbol} SHORT at {last_price} (TP:"
            f" {pos.take_profit})"
        )
        self.close_position(symbol, last_price, reason='TAKE_PROFIT')

  def submit_order(
      self,
      symbol: str,
      side: str,
      quantity: float,
      market_price: float,
      order_type: str = 'MARKET',
      price: Optional[float] = None,
      stop_loss: Optional[float] = None,
      take_profit: Optional[float] = None,
      source: str = 'MANUAL',
  ) -> PaperOrder:
    """Executes a paper order immediately for MARKET type or registers LIMIT order."""
    exec_price = market_price if order_type == 'MARKET' else (price or market_price)
    
    order = PaperOrder(
        symbol=symbol,
        side=side.upper(),
        order_type=order_type.upper(),
        quantity=quantity,
        price=exec_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        status='FILLED' if order_type == 'MARKET' else 'OPEN',
        source=source,
    )

    if order_type == 'MARKET':
      self._execute_fill(order, exec_price)
    else:
      self.open_orders.append(order)

    return order

  def _execute_fill(self, order: PaperOrder, fill_price: float) -> None:
    symbol = order.symbol
    side = 'LONG' if order.side == 'BUY' else 'SHORT'
    order_cost = fill_price * order.quantity

    if symbol in self.positions:
      existing = self.positions[symbol]
      if existing.side == side:
        # Average into position
        total_size = existing.size + order.quantity
        new_entry = (
            (existing.entry_price * existing.size) + (fill_price * order.quantity)
        ) / total_size
        existing.size = total_size
        existing.entry_price = round(new_entry, 4)
        existing.margin_used = round(total_size * new_entry, 2)
      else:
        # Configurable minimum position age via AG_MIN_REVERSAL_AGE_MS (default: 3000ms)
        min_reversal_env = os.getenv("AG_MIN_REVERSAL_AGE_MS", "3000.0")
        try:
          MIN_REVERSAL_AGE_MS = float(min_reversal_env)
        except ValueError:
          MIN_REVERSAL_AGE_MS = 3000.0

        now_ms = time.time() * 1000
        age_ms = now_ms - getattr(existing, 'opened_timestamp_ms', 0)
        if age_ms < MIN_REVERSAL_AGE_MS:
          logger.info(
              f"reversal.blocked_too_young: {symbol} age={round(age_ms, 1)}ms"
              f" < min={MIN_REVERSAL_AGE_MS}ms"
          )
          return

        self.close_position(symbol, fill_price, reason='REVERSAL')
        self._open_new_position(order, fill_price, side, order_cost)
    else:
      self._open_new_position(order, fill_price, side, order_cost)

  def _open_new_position(
      self, order: PaperOrder, fill_price: float, side: str, margin: float
  ) -> None:
    pos = PaperPosition(
        symbol=order.symbol,
        side=side,
        size=order.quantity,
        entry_price=fill_price,
        current_price=fill_price,
        margin_used=round(margin, 2),
        stop_loss=order.stop_loss,
        take_profit=order.take_profit,
    )
    self.positions[order.symbol] = pos

  def close_position(
      self, symbol: str, current_price: float = 0.0, reason: str = 'MANUAL'
  ) -> Optional[PaperPosition]:
    if symbol not in self.positions:
      return None

    pos = self.positions.pop(symbol)
    close_price = current_price if current_price > 0 else pos.current_price

    if pos.side == 'LONG':
      pnl = (close_price - pos.entry_price) * pos.size
    else:
      pnl = (pos.entry_price - close_price) * pos.size

    pnl = round(pnl, 2)
    self.realized_pnl += pnl
    self.cash_balance += pnl
    self.total_trades += 1
    if pnl > 0:
      self.winning_trades += 1

    trade_record = {
        'symbol': symbol,
        'side': pos.side,
        'size': pos.size,
        'entry_price': pos.entry_price,
        'close_price': close_price,
        'pnl': pnl,
        'reason': reason,
        'closed_at': datetime.now(timezone.utc).isoformat(),
    }
    self.trade_history.append(trade_record)

    sample = {
        'sample_id': f"EXP-{int(time.time() * 1000)}",
        'symbol': symbol,
        'side': pos.side,
        'size': pos.size,
        'entry_price': pos.entry_price,
        'exit_price': close_price,
        'realized_pnl': pnl,
        'pnl_pct': round((pnl / (pos.entry_price * pos.size)) * 100, 4) if (pos.entry_price * pos.size) > 0 else 0.0,
        'reward': round(pnl - 0.0005 * close_price * pos.size, 4),  # Net reward after slippage/cost
        'reason': reason,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'relooped_to_rl': True,
    }
    self.reloop_buffer.append(sample)

    # Immediately persist trade experience for RL & Autoresearch Engine
    try:
        from antigravity.rl.reloop import reloop_engine
        reloop_engine.ingest_samples([sample])
    except Exception as exc:
        logger.warning(f"reloop.auto_ingest_failed: {exc}")

    # Log trade record to persistent CSV and JSON files with 10MB size-based rotation
    try:
        import csv
        import json
        import os

        # 1. Append to CSV log data/paper_trade_log.csv (Rotate if > 10MB)
        csv_path = "data/paper_trade_log.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 10 * 1024 * 1024:
            os.rename(csv_path, f"{csv_path}.{int(time.time())}.bak")
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "symbol", "side", "size", "entry_price", "close_price", "realized_pnl", "pnl_pct", "reason"])
            writer.writerow([
                sample["timestamp"],
                symbol,
                pos.side,
                pos.size,
                pos.entry_price,
                close_price,
                pnl,
                sample["pnl_pct"],
                reason
            ])

        # 2. Append to JSON audit log data/paper_trades_audit.jsonl (Rotate if > 10MB)
        json_path = "data/paper_trades_audit.jsonl"
        if os.path.exists(json_path) and os.path.getsize(json_path) > 10 * 1024 * 1024:
            os.rename(json_path, f"{json_path}.{int(time.time())}.bak")
        with open(json_path, "a") as f:
            f.write(json.dumps(sample) + "\n")
    except Exception as exc:
        logger.warning(f"trade_logger.file_write_failed: {exc}")

    return pos

  def set_balance(self, new_balance: float) -> None:
    self.initial_balance = new_balance
    self.cash_balance = new_balance
    self.realized_pnl = 0.0
    self.total_trades = 0
    self.winning_trades = 0
    self.positions.clear()
    self.open_orders.clear()
    self.trade_history.clear()
    self.reloop_buffer.clear()
    logger.info(f"Paper trading account balance set to ${new_balance:,.2f}")

  def reset_account(self) -> None:
    self.set_balance(self.initial_balance)
