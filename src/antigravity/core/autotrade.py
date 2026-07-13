"""
Autotrade engine.
Signal logic: regime signal + OBI confirmation both required before placing order.

Entry conditions:
  LONG:  current_regime in BULLISH_REGIMES AND obi > OBI_THRESHOLD
  SHORT: current_regime in BEARISH_REGIMES AND obi < -OBI_THRESHOLD
  (Short = sell existing position only — no shorting on spot)

Cooldown: minimum 60s between orders to prevent thrashing.
"""
import asyncio
import time
import structlog
from antigravity.execution.cryptocom import CryptoComClient

log = structlog.get_logger(__name__)

# Regime names that indicate bullish/bearish bias
# Adjust these to match your actual regime labels
BULLISH_REGIMES = {"trending_up", "trending↑", "regime_0", "0"}
BEARISH_REGIMES = {"trending_down", "trending↓", "regime_1", "1"}

OBI_THRESHOLD     = 0.05   # minimum |OBI| to confirm signal
COOLDOWN_SECONDS  = 60     # minimum seconds between orders
MIN_USDT_ORDER    = 10.0   # minimum order size in USDT


class AutotradeEngine:
    def __init__(
        self,
        client: CryptoComClient,
        instrument: str,            # e.g. "BTC_USDT"
        max_position_usdt: float,   # max USDT to deploy per position
    ):
        self.client = client
        self.instrument = instrument
        self.max_position_usdt = max_position_usdt
        self._last_order_time: float = 0.0
        self._last_side: str | None = None
        self._running = False

    def _in_cooldown(self) -> bool:
        return (time.time() - self._last_order_time) < COOLDOWN_SECONDS

    def _regime_signal(self, regime: str) -> str | None:
        """Returns 'BUY', 'SELL', or None based on current regime."""
        r = regime.lower().strip()
        if r in BULLISH_REGIMES:
            return "BUY"
        if r in BEARISH_REGIMES:
            return "SELL"
        return None

    def _obi_signal(self, obi: float) -> str | None:
        """Returns 'BUY', 'SELL', or None based on OBI value."""
        if obi > OBI_THRESHOLD:
            return "BUY"
        if obi < -OBI_THRESHOLD:
            return "SELL"
        return None

    def evaluate_signal(self, regime: str, obi: float) -> str | None:
        """
        Both signals must agree for an order to fire.
        Returns 'BUY', 'SELL', or None.
        """
        rs = self._regime_signal(regime)
        os = self._obi_signal(obi)

        if rs is None or os is None:
            return None
        if rs != os:
            log.debug("autotrade.signal_conflict", regime_signal=rs, obi_signal=os)
            return None

        return rs

    async def tick(self, regime: str, obi: float) -> dict | None:
        """
        Called on every status poll. Evaluates signal and fires order if conditions met.
        Returns order dict if an order was placed, else None.
        """
        if self._in_cooldown():
            return None

        signal = self.evaluate_signal(regime, obi)
        if signal is None:
            return None

        # Don't repeat same side consecutively
        if signal == self._last_side:
            return None

        log.info("autotrade.signal_confirmed",
                 signal=signal, regime=regime, obi=obi,
                 instrument=self.instrument)

        try:
            if signal == "BUY":
                # Check balance first
                balances = await self.client.get_balance()
                usdt_available = balances.get("USDT", 0.0)
                notional = min(self.max_position_usdt, usdt_available * 0.95)  # use max 95% of available
                if notional < MIN_USDT_ORDER:
                    log.warning("autotrade.insufficient_balance",
                                usdt_available=usdt_available, required=MIN_USDT_ORDER)
                    return None
                order = await self.client.create_market_order(self.instrument, "BUY", notional)
            else:
                # SELL — cancel any existing orders first, then sell
                await self.client.cancel_all_orders(self.instrument)
                # Get coin balance to sell
                coin = self.instrument.split("_")[0]
                balances = await self.client.get_balance()
                # Re-fetch full balance for coin currency
                result = await self.client._post(
                    "private/get-account-summary", {"currency": coin}
                )
                accounts = result.get("accounts", [])
                coin_balance = float(accounts[0]["available"]) if accounts else 0.0
                if coin_balance <= 0:
                    log.info("autotrade.nothing_to_sell", coin=coin)
                    return None
                order = await self.client.create_market_order(
                    self.instrument, "SELL", coin_balance, is_quantity=True
                )

            self._last_order_time = time.time()
            self._last_side = signal
            log.info("autotrade.order_complete", order=order)
            return order

        except Exception as exc:
            log.error("autotrade.order_failed", error=str(exc))
            return None

    async def stop(self) -> None:
        await self.client.close()
