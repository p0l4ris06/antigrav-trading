"""
Unit tests for PaperTradingEngine timing, reversal protection, and PnL calculations.
"""

import time
import pytest
from antigravity.core.paper_engine import PaperTradingEngine, PaperOrder, PaperPosition


@pytest.fixture
def engine():
    return PaperTradingEngine(initial_balance=100_000.0)


def test_paper_position_initialization(engine):
    summary = engine.get_summary()
    assert summary.cash_balance == 100_000.0
    assert summary.equity == 100_000.0
    assert summary.total_trades == 0
    assert len(summary.positions) == 0


def test_submit_market_buy_order(engine):
    order = engine.submit_order(
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.1,
        market_price=68000.0,
        order_type="MARKET",
        source="BOT"
    )

    assert order.status == "FILLED"
    assert "BTCUSDT" in engine.positions
    pos = engine.positions["BTCUSDT"]
    assert pos.side == "LONG"
    assert pos.size == 0.1
    assert pos.entry_price == 68000.0


def test_reversal_blocked_under_3000ms(engine):
    # Open LONG position
    engine.submit_order(
        symbol="ETHUSDT",
        side="BUY",
        quantity=1.0,
        market_price=3400.0,
        order_type="MARKET",
        source="BOT"
    )

    # Immediately submit opposing SELL order (< 3000ms)
    engine.submit_order(
        symbol="ETHUSDT",
        side="SELL",
        quantity=1.0,
        market_price=3390.0,
        order_type="MARKET",
        source="BOT"
    )

    # Position should NOT reverse because age is under 3000ms
    assert "ETHUSDT" in engine.positions
    assert engine.positions["ETHUSDT"].side == "LONG"


def test_reversal_allowed_after_3000ms(engine):
    # Open LONG position
    pos_order = engine.submit_order(
        symbol="SOLUSDT",
        side="BUY",
        quantity=10.0,
        market_price=190.0,
        order_type="MARKET",
        source="BOT"
    )

    # Simulate position opened 4000ms ago
    engine.positions["SOLUSDT"].opened_timestamp_ms = (time.time() * 1000) - 4000.0

    # Submit opposing SELL order (> 3000ms)
    engine.submit_order(
        symbol="SOLUSDT",
        side="SELL",
        quantity=10.0,
        market_price=192.0,
        order_type="MARKET",
        source="BOT"
    )

    # Position should cleanly reverse to SHORT
    assert "SOLUSDT" in engine.positions
    assert engine.positions["SOLUSDT"].side == "SHORT"
    assert engine.realized_pnl == 20.0  # (192 - 190) * 10 = +$20 profit


def test_take_profit_trigger(engine):
    engine.submit_order(
        symbol="NVDA",
        side="BUY",
        quantity=10.0,
        market_price=130.0,
        order_type="MARKET",
        take_profit=135.0,
        stop_loss=125.0,
        source="BOT"
    )

    # Tick price touches TP at 136.0
    engine.on_tick("NVDA", 136.0)

    # Position should close automatically on TP
    assert "NVDA" not in engine.positions
    assert engine.total_trades == 1
    assert engine.winning_trades == 1
    assert engine.realized_pnl == 60.0  # (136 - 130) * 10 = +$60


def test_custom_min_reversal_age_ms(monkeypatch):
    # Test instance override (e.g. 1000ms)
    custom_eng = PaperTradingEngine(min_reversal_age_ms=1000.0)
    assert custom_eng.min_reversal_age_ms == 1000.0

    # Test env var override (e.g. 5000ms)
    monkeypatch.setenv("AG_MIN_REVERSAL_AGE_MS", "5000.0")
    env_eng = PaperTradingEngine()
    assert env_eng.min_reversal_age_ms == 5000.0


def test_atomic_log_rotation(tmp_path):
    from antigravity.core.paper_engine import rotate_if_needed
    test_log = tmp_path / "test_audit.jsonl"
    
    # Create file > 10MB
    test_log.write_bytes(b"X" * (10 * 1024 * 1024 + 100))
    rotate_if_needed(str(test_log))

    # Original path should no longer exist, rotated backup should exist
    assert not test_log.exists()
    backups = list(tmp_path.glob("test_audit.jsonl.*.bak"))
    assert len(backups) == 1

