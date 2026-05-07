import ccxt.async_support as ccxt
import asyncio

class RiskManager:
    def __init__(self, max_account_risk_pct=0.05):
        self.max_account_risk_pct = max_account_risk_pct

    def calculate_lot_size(self, account_equity: float, kelly_fraction: float, atr: float, contract_multiplier: float = 1.0) -> float:
        safe_fraction = min(kelly_fraction, self.max_account_risk_pct)
        capital_at_risk = account_equity * safe_fraction
        stop_loss_distance = atr * 2.0
        
        if stop_loss_distance <= 0: return 0.0
        position_size = capital_at_risk / (stop_loss_distance * contract_multiplier)
        return round(position_size, 5)

class CryptoComAdapter:
    def __init__(self, api_key, secret):
        self.exchange = ccxt.cryptocom({'apiKey': api_key, 'secret': secret, 'enableRateLimit': True})

    async def execute_ticket(self, symbol, bias, lot_size):
        side = 'buy' if bias == 1 else 'sell'
        try:
            print(f"[EXECUTION] Routing {side} {lot_size} {symbol} to Crypto.com")
            # await self.exchange.create_market_order(symbol, side, lot_size)
        except Exception as e:
            print(f"Execution Failed: {e}")

class OmniGateway:
    def __init__(self, crypto_config=None):
        self.adapters = {}
        if crypto_config:
            adapter = CryptoComAdapter(**crypto_config)
            self.adapters['CRYPTO'] = adapter
            self.adapters['CRYPTOCOM'] = adapter
        self.risk_engine = RiskManager()

    async def route_action(self, target_exchange: str, symbol: str, action_vector: list, account_equity: float, current_atr: float):
        if target_exchange not in self.adapters:
            raise ValueError(f"Exchange {target_exchange} not initialized.")
            
        adapter = self.adapters[target_exchange]
        bias = 1 if action_vector[0] > 0 else -1
        kelly_confidence = float(action_vector[1]) 
        
        target_lot_size = self.risk_engine.calculate_lot_size(
            account_equity=account_equity,
            kelly_fraction=kelly_confidence,
            atr=current_atr,
            contract_multiplier=1.0 # 1.0 for spot crypto
        )
        
        if target_lot_size > 0.0001: # Crypto fractional threshold
            await adapter.execute_ticket(symbol, bias, target_lot_size)
