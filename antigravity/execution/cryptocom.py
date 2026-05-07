"""
Crypto.com App API client — routes through the official crypto-com-app skill scripts.
Auth is handled by the skill via CDC_API_KEY / CDC_API_SECRET env vars.
Never calls the API directly.
"""
import asyncio
import json
import os
import structlog

log = structlog.get_logger(__name__)

# Path to the installed skill scripts — adjust if npx resolves differently
SKILL_BASE = "npx tsx"


async def _run_script(script: str, *args: str) -> dict:
    """Run a skill script and return parsed JSON output."""
    cmd = f"{SKILL_BASE} {script} {' '.join(args)}"
    log.debug("cryptocom.skill_call", cmd=cmd)

    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ},  # passes CDC_API_KEY and CDC_API_SECRET through
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        err = stderr.decode().strip()
        log.error("cryptocom.skill_error", cmd=cmd, error=err)
        raise RuntimeError(f"Skill script failed: {err}")

    try:
        return json.loads(stdout.decode().strip())
    except json.JSONDecodeError:
        # Some scripts return plain text — wrap it
        return {"raw": stdout.decode().strip()}


async def _find_script(name: str) -> str:
    """Locate the skill script path via npx."""
    proc = await asyncio.create_subprocess_shell(
        f"npx skills path crypto-com-app {name}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    path = stdout.decode().strip()
    if not path:
        raise RuntimeError(f"Cannot find skill script: {name}")
    return path


class CryptoComClient:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        # Inject into environment so skill scripts pick them up
        os.environ["CDC_API_KEY"] = api_key
        os.environ["CDC_API_SECRET"] = api_secret
        log.info("cryptocom.client_init", key_prefix=api_key[:8])

    async def get_balance(self) -> dict[str, float]:
        """Returns {currency: available_balance}"""
        try:
            result = await _run_script("account.ts", "balances", "all")
            # Skill returns list of {symbol, balance} or similar — normalise
            if isinstance(result, list):
                return {item["symbol"]: float(item.get("balance", item.get("available", 0))) for item in result}
            if isinstance(result, dict) and "balances" in result:
                return {item["symbol"]: float(item.get("balance", 0)) for item in result["balances"]}
            return result
        except Exception as exc:
            log.error("cryptocom.get_balance_failed", error=str(exc))
            raise

    async def get_quote(self, trade_type: str, params: dict) -> dict:
        """
        Get a trade quote. trade_type: 'purchase' | 'sale' | 'exchange'
        Returns quotation_id and expiry.
        """
        params_json = json.dumps(params)
        result = await _run_script("trade.ts", "quote", trade_type, f"'{params_json}'")
        log.info("cryptocom.quote", trade_type=trade_type, result=result)
        return result

    async def confirm_quote(self, trade_type: str, quotation_id: str) -> dict:
        """Execute a previously obtained quote."""
        result = await _run_script("trade.ts", "confirm", trade_type, quotation_id)
        log.info("cryptocom.confirm", trade_type=trade_type, quotation_id=quotation_id, result=result)
        return result

    async def create_market_order(
        self,
        instrument: str,   # e.g. "BTC_USDT"
        side: str,         # "BUY" or "SELL"
        notional: float,
        is_quantity: bool = False,
    ) -> dict:
        """
        Two-step quote → confirm flow required by the skill.
        BUY:  purchase fiat→crypto, notional = USDT amount
        SELL: sale crypto→fiat, notional = coin quantity
        """
        coin, quote_currency = instrument.split("_")

        if side.upper() == "BUY":
            params = {
                "from_currency": quote_currency,
                "to_currency": coin,
                "from_amount": str(notional),
            }
            quote = await self.get_quote("purchase", params)
        else:
            params = {
                "from_currency": coin,
                "to_currency": quote_currency,
                "from_amount": str(notional),
            }
            quote = await self.get_quote("sale", params)

        quotation_id = quote.get("quotation_id") or quote.get("id")
        if not quotation_id:
            raise RuntimeError(f"No quotation_id in quote response: {quote}")

        trade_type = "purchase" if side.upper() == "BUY" else "sale"
        order = await self.confirm_quote(trade_type, str(quotation_id))
        log.info("cryptocom.order_placed",
                 instrument=instrument, side=side, notional=notional, order=order)
        return order

    async def cancel_all_orders(self, instrument: str) -> None:
        # App API doesn't support order cancellation the same way — 
        # quotes expire automatically so this is a no-op
        log.info("cryptocom.cancel_all_noop", instrument=instrument)

    async def get_trade_history(self) -> list:
        result = await _run_script("trade.ts", "history")
        if isinstance(result, list):
            return result
        return result.get("transactions", [])

    async def close(self) -> None:
        pass  # No persistent connection to close