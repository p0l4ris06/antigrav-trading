import { useState, useEffect, useRef, useCallback, createContext, useContext } from 'react';
import { useWebSocket } from './useWebSocket';
import { apiConfig } from '../config';

export type Symbol = string; // e.g., 'BTCUSDT'

export type SymbolInfo = {
  symbol: Symbol;
  name: string;
  category: 'Crypto' | 'Stock' | 'ETF';
  basePrice: number;
  icon: string;
};

export const AVAILABLE_SYMBOLS: SymbolInfo[] = [
  // Crypto
  { symbol: 'BTCUSDT', name: 'Bitcoin', category: 'Crypto', basePrice: 68500.0, icon: '₿' },
  { symbol: 'ETHUSDT', name: 'Ethereum', category: 'Crypto', basePrice: 3250.0, icon: 'Ξ' },
  { symbol: 'SOLUSDT', name: 'Solana', category: 'Crypto', basePrice: 192.00, icon: '◎' },
  { symbol: 'AVAXUSDT', name: 'Avalanche', category: 'Crypto', basePrice: 32.50, icon: '🔺' },
  { symbol: 'DOGEUSDT', name: 'Dogecoin', category: 'Crypto', basePrice: 0.138, icon: '🐕' },
  // Stocks
  { symbol: 'NVDA', name: 'NVIDIA Corp', category: 'Stock', basePrice: 196.50, icon: '🟢' },
  { symbol: 'AAPL', name: 'Apple Inc', category: 'Stock', basePrice: 233.00, icon: '🍎' },
  { symbol: 'TSLA', name: 'Tesla Inc', category: 'Stock', basePrice: 272.00, icon: '⚡' },
  { symbol: 'MSFT', name: 'Microsoft Corp', category: 'Stock', basePrice: 468.00, icon: '🪟' },
  { symbol: 'AMZN', name: 'Amazon.com Inc', category: 'Stock', basePrice: 205.00, icon: '📦' },
  { symbol: 'AMD', name: 'Adv Micro Devices', category: 'Stock', basePrice: 164.00, icon: '🔴' },
  // ETFs
  { symbol: 'SPY', name: 'SPDR S&P 500 ETF', category: 'ETF', basePrice: 570.00, icon: '📊' },
  { symbol: 'QQQ', name: 'Invesco QQQ Trust', category: 'ETF', basePrice: 510.00, icon: '📈' },
];

export type Tick = {
  symbol: string;
  bid_price: number; ask_price: number;
  bid_size: number; ask_size: number;
  last_price: number; last_size: number;
  trade_id: number;
};

export type Candle = {
  time: number; open: number; high: number; low: number; close: number; volume: number;
};

export type OBISnapshot = {
  bid_price: number; ask_price: number;
  bid_size: number; ask_size: number;
  spread: number; obi: number;
};

export type SystemStatus = {
  current_regime: string; overseer_state: string;
  rolling_sharpe: number; drift_detected: boolean;
  shadow_fork_active: boolean; ticks_ingested: number;
  buffer_height: number; uptime_seconds: number;
  overseer_events: Array<{ timestamp?: string; event?: string; [key: string]: unknown }>;
  regime_probabilities: number[];
  portfolio_weights: number[];
  ppo_model_loaded: boolean;
  prices: Record<string, number>;
  target_stop_triggered?: boolean;
};

export type SymbolState = {
  latestTick: Tick | null;
  candles: Candle[];
  obi: OBISnapshot | null;
};

export type PaperOrderParams = {
  symbol: string;
  side: 'BUY' | 'SELL';
  order_type?: 'MARKET' | 'LIMIT';
  quantity: number;
  price?: number;
  stop_loss?: number;
  take_profit?: number;
  source?: 'MANUAL' | 'BOT';
};

export type PaperPosition = {
  symbol: string;
  side: 'LONG' | 'SHORT';
  size: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  margin_used: number;
};

export type PaperAccountSummary = {
  cash_balance: number;
  equity: number;
  realized_pnl: number;
  unrealized_pnl: number;
  margin_used: number;
  win_rate: number;
  total_trades: number;
  winning_trades: number;
  positions: PaperPosition[];
  open_orders: unknown[];
};

export type RefreshSpeed = '10Hz' | '2Hz' | '0.5Hz';
export type TradeSignalType = 'STRONG BUY' | 'BUY' | 'NEUTRAL' | 'SELL' | 'STRONG SELL';

export type TradeMarker = {
  id: string;
  time: number;
  timeStr: string;
  price: number;
  type: 'BUY' | 'SELL';
  signal: TradeSignalType;
  symbol: string;
};

const BUCKET = 5; // 5-second candles for visual updating
const SIGNAL_HOLD_MS = 3000; // 3-second hold window for human visual latching

function useMarketDataProviderValue() {
  const ws = useWebSocket(`${apiConfig.wsBaseUrl}/ws/simulated_feed`);

  const [selectedSymbol, setSelectedSymbol] = useState<Symbol>('BTCUSDT');
  const [symbolData, setSymbolData] = useState<Record<Symbol, SymbolState>>({});
  const [isPaused, setIsPaused] = useState(false);
  const [isAutoTradeEnabled, setIsAutoTradeEnabled] = useState(true);
  const [refreshSpeed, setRefreshSpeed] = useState<RefreshSpeed>('10Hz');

  // Per-asset auto-trading toggles state
  const [enabledAssets, setEnabledAssets] = useState<Record<string, boolean>>(() => {
    const saved = typeof window !== 'undefined' ? localStorage.getItem('antigrav_enabled_assets') : null;
    if (saved) {
      try { return JSON.parse(saved); } catch {}
    }
    const defaultMap: Record<string, boolean> = {};
    AVAILABLE_SYMBOLS.forEach(s => { defaultMap[s.symbol] = true; });
    return defaultMap;
  });
  const enabledAssetsRef = useRef(enabledAssets);
  enabledAssetsRef.current = enabledAssets;

  const toggleAssetTrading = useCallback(async (symbol: string, enabled: boolean) => {
    setEnabledAssets(prev => {
      const next = { ...prev, [symbol]: enabled };
      if (typeof window !== 'undefined') {
        localStorage.setItem('antigrav_enabled_assets', JSON.stringify(next));
      }
      const activeList = Object.keys(next).filter(k => next[k]);
      fetch(`${apiConfig.baseUrl}/api/execution/symbols`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols: activeList }),
      }).catch(() => {});
      return next;
    });
  }, []);

  // Account Balance Target Stop state (Take-Profit Target & Drawdown Stop)
  const [targetProfitEquity, setTargetProfitEquity] = useState<number | null>(() => {
    const saved = typeof window !== 'undefined' ? localStorage.getItem('antigrav_target_profit_equity') : null;
    return saved ? parseFloat(saved) : null;
  });
  const [maxDrawdownEquity, setMaxDrawdownEquity] = useState<number | null>(() => {
    const saved = typeof window !== 'undefined' ? localStorage.getItem('antigrav_max_drawdown_equity') : null;
    return saved ? parseFloat(saved) : null;
  });
  const [targetStopTriggered, setTargetStopTriggered] = useState(false);

  const setAccountTargetStops = useCallback(async (targetProfit?: number | null, maxDrawdown?: number | null) => {
    setTargetProfitEquity(targetProfit ?? null);
    setMaxDrawdownEquity(maxDrawdown ?? null);
    setTargetStopTriggered(false);
    setIsAutoTradeEnabled(true);

    if (typeof window !== 'undefined') {
      if (targetProfit !== undefined && targetProfit !== null) {
        localStorage.setItem('antigrav_target_profit_equity', targetProfit.toString());
      } else {
        localStorage.removeItem('antigrav_target_profit_equity');
      }

      if (maxDrawdown !== undefined && maxDrawdown !== null) {
        localStorage.setItem('antigrav_max_drawdown_equity', maxDrawdown.toString());
      } else {
        localStorage.removeItem('antigrav_max_drawdown_equity');
      }
    }

    try {
      await fetch(`${apiConfig.baseUrl}/api/execution/target_stops`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_profit_equity: targetProfit ?? null,
          max_drawdown_equity: maxDrawdown ?? null,
          enabled: true,
        }),
      });
    } catch {}
  }, []);

  // Smoothed signals state per symbol
  const [smoothedObi, setSmoothedObi] = useState(0);
  const [latchedSignal, setLatchedSignal] = useState<TradeSignalType>('NEUTRAL');
  const [isSignalLatched, setIsSignalLatched] = useState(false);
  const [tradeMarkers, setTradeMarkers] = useState<TradeMarker[]>([]);

  // Paper trading account & RL Reloop state (initialize custom capital from localStorage)
  const [paperAccount, setPaperAccount] = useState<PaperAccountSummary | null>(() => {
    const savedCap = typeof window !== 'undefined' ? localStorage.getItem('antigrav_custom_capital') : null;
    const initialCap = savedCap ? parseFloat(savedCap) : 100000.0;
    return {
      cash_balance: initialCap,
      equity: initialCap,
      realized_pnl: 0.0,
      unrealized_pnl: 0.0,
      margin_used: 0.0,
      win_rate: 0.0,
      total_trades: 0,
      winning_trades: 0,
      positions: [],
      open_orders: [],
    };
  });

  const [reloopTelemetry, setReloopTelemetry] = useState<{
    reloop_active: boolean;
    relooped_samples_count: number;
    average_relooped_reward: number;
    latest_policy_loss: number;
  }>({
    reloop_active: true,
    relooped_samples_count: 14,
    average_relooped_reward: 12.45,
    latest_policy_loss: 0.0142,
  });

  const [status, setStatus] = useState<SystemStatus>({
    current_regime: '—', overseer_state: 'MONITORING',
    rolling_sharpe: 0, drift_detected: false,
    shadow_fork_active: false, ticks_ingested: 0,
    buffer_height: 0, uptime_seconds: 0,
    overseer_events: [], regime_probabilities: [], portfolio_weights: [],
    ppo_model_loaded: false,
    prices: {},
  });

  const [gatewayConnected, setGatewayConnected] = useState(false);

  // Buffer refs for refresh rate throttling & smoothing
  const symbolDataRef = useRef<Record<Symbol, SymbolState>>({});
  const pendingTicksRef = useRef<Tick[]>([]);
  const emaObiMapRef = useRef<Record<string, number>>({});
  const latchedSignalMapRef = useRef<Record<string, TradeSignalType>>({});
  const lastSignalTimeMapRef = useRef<Record<string, number>>({});
  const customCapitalRef = useRef<number | null>(
    typeof window !== 'undefined' && localStorage.getItem('antigrav_custom_capital') 
      ? parseFloat(localStorage.getItem('antigrav_custom_capital')!) 
      : null
  );
  const statusPricesRef = useRef<Record<string, number>>({});
  const isPausedRef = useRef(isPaused);
  isPausedRef.current = isPaused;
  const refreshSpeedRef = useRef(refreshSpeed);
  refreshSpeedRef.current = refreshSpeed;
  const selectedSymbolRef = useRef(selectedSymbol);
  selectedSymbolRef.current = selectedSymbol;
  const isAutoTradeEnabledRef = useRef(isAutoTradeEnabled);
  isAutoTradeEnabledRef.current = isAutoTradeEnabled;
  const targetStopTriggeredRef = useRef(targetStopTriggered);
  targetStopTriggeredRef.current = targetStopTriggered;

  const applyTick = useCallback((prev: Record<Symbol, SymbolState>, sym: Symbol, tick: Tick): Record<Symbol, SymbolState> => {
    const now = Math.floor(Date.now() / 1000);
    const bucket = now - (now % BUCKET);
    const price = tick.last_price;
    const total = tick.bid_size + tick.ask_size;

    const rawObi = total > 0 ? (tick.bid_size - tick.ask_size) / total : 0;
    
    // EMA smoothing per symbol (alpha = 0.25)
    const prevEma = emaObiMapRef.current[sym] ?? 0;
    const currentEma = 0.25 * rawObi + 0.75 * prevEma;
    emaObiMapRef.current[sym] = currentEma;
    const currentSmoothed = +currentEma.toFixed(4);

    const obi: OBISnapshot = {
      bid_price: tick.bid_price, ask_price: tick.ask_price,
      bid_size: tick.bid_size, ask_size: tick.ask_size,
      spread: +(tick.ask_price - tick.bid_price).toFixed(2),
      obi: currentSmoothed,
    };

    const prevState = prev[sym] || { latestTick: null, candles: [], obi: null };
    const prevCandles = prevState.candles;
    const last = prevCandles[prevCandles.length - 1];
    
    let candles: Candle[];
    if (last && last.time === bucket) {
      const updated = { ...last, high: Math.max(last.high, price), low: Math.min(last.low, price), close: price, volume: last.volume + tick.last_size };
      candles = [...prevCandles.slice(0, -1), updated];
    } else {
      candles = [...prevCandles.slice(-199), { time: bucket, open: last?.close ?? price, high: price, low: price, close: price, volume: tick.last_size }];
    }

    return { ...prev, [sym]: { latestTick: tick, candles, obi } };
  }, []);

  // Generate multi-symbol synthetic ticks seeded from live Alpaca prices
  const expandMultiSymbolTicks = useCallback((primaryTick: Tick): Tick[] => {
    const ticks: Tick[] = [primaryTick];
    const baseSym = primaryTick.symbol;

    AVAILABLE_SYMBOLS.forEach((info) => {
      if (info.symbol === baseSym) return;

      const currentPriceMap = statusPricesRef.current[info.symbol] ?? symbolDataRef.current[info.symbol]?.latestTick?.last_price ?? info.basePrice;
      const noisePct = (Math.random() - 0.5) * 0.002; // ±0.1% random walk
      const newPrice = +(currentPriceMap * (1 + noisePct)).toFixed(2);
      const spread = +(newPrice * 0.0002).toFixed(2);

      const bidSize = +(Math.random() * 5 + 0.1).toFixed(4);
      const askSize = +(Math.random() * 5 + 0.1).toFixed(4);

      ticks.push({
        symbol: info.symbol,
        bid_price: +(newPrice - spread / 2).toFixed(2),
        ask_price: +(newPrice + spread / 2).toFixed(2),
        bid_size: bidSize,
        ask_size: askSize,
        last_price: newPrice,
        last_size: +(Math.random() * 2 + 0.01).toFixed(4),
        trade_id: primaryTick.trade_id,
      });
    });

    return ticks;
  }, []);

  // Submit paper trade order handler
  const submitOrder = useCallback(async (params: PaperOrderParams): Promise<boolean> => {
    try {
      const res = await fetch(`${apiConfig.baseUrl}/api/paper/order`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });

      if (res.ok) {
        const data = await res.json();
        if (data.status === 'target_reached' || data.status === 'drawdown_triggered' || data.status === 'execution_disabled') {
          setIsAutoTradeEnabled(false);
          setTargetStopTriggered(true);
          return false;
        }

        const accRes = await fetch(`${apiConfig.baseUrl}/api/paper/account`);
        if (accRes.ok) {
          const acc = await accRes.json();
          if (customCapitalRef.current !== null) {
            acc.cash_balance = customCapitalRef.current + acc.realized_pnl;
            acc.equity = customCapitalRef.current + acc.realized_pnl + acc.unrealized_pnl;
          }
          setPaperAccount(acc);
        }
        return true;
      }
      return false;
    } catch {
      return false;
    }
  }, []);

  // Close open position handler
  const closePosition = useCallback(async (sym: string): Promise<boolean> => {
    try {
      const res = await fetch(`${apiConfig.baseUrl}/api/paper/position/close`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: sym }),
      });

      if (res.ok) {
        const accRes = await fetch(`${apiConfig.baseUrl}/api/paper/account`);
        if (accRes.ok) {
          const acc = await accRes.json();
          if (customCapitalRef.current !== null) {
            acc.cash_balance = customCapitalRef.current + acc.realized_pnl;
            acc.equity = customCapitalRef.current + acc.realized_pnl + acc.unrealized_pnl;
          }
          setPaperAccount(acc);
        }
        return true;
      }
      return false;
    } catch {
      return false;
    }
  }, []);

  // Reset paper account balance handler
  const resetPaperAccount = useCallback(async (): Promise<boolean> => {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('antigrav_custom_capital');
    }
    customCapitalRef.current = null;

    try {
      await fetch(`${apiConfig.baseUrl}/api/paper/reset`, { method: 'POST' });
    } catch {
      // ignore
    }

    setTargetStopTriggered(false);
    setIsAutoTradeEnabled(true);

    setPaperAccount({
      cash_balance: 100000.0,
      equity: 100000.0,
      realized_pnl: 0.0,
      unrealized_pnl: 0.0,
      margin_used: 0.0,
      win_rate: 0.0,
      total_trades: 0,
      winning_trades: 0,
      positions: [],
      open_orders: [],
    });
    return true;
  }, []);

  // Set custom paper balance handler
  const setPaperBalance = useCallback(async (amount: number): Promise<boolean> => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('antigrav_custom_capital', amount.toString());
    }
    customCapitalRef.current = amount;

    const newAcc: PaperAccountSummary = {
      cash_balance: amount,
      equity: amount,
      realized_pnl: 0.0,
      unrealized_pnl: 0.0,
      margin_used: 0.0,
      win_rate: 0.0,
      total_trades: 0,
      winning_trades: 0,
      positions: [],
      open_orders: [],
    };

    setPaperAccount(newAcc);
    setTargetStopTriggered(false);
    setIsAutoTradeEnabled(true);

    try {
      await fetch(`${apiConfig.baseUrl}/api/paper/balance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ balance: amount }),
      });
    } catch {
      // local custom state maintained
    }
    return true;
  }, []);

  // Ingest paper experiences & trigger RL reloop learning step
  const triggerReloop = useCallback(async (): Promise<boolean> => {
    try {
      const res = await fetch(`${apiConfig.baseUrl}/api/rl/reloop/trigger`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        if (data.telemetry) setReloopTelemetry(data.telemetry);
        return true;
      }
      return false;
    } catch {
      return false;
    }
  }, []);

  // Signal derivation & latching algorithm per symbol
  const updateSignalAndMarkers = useCallback((activeSym: Symbol, symState: SymbolState) => {
    if (!symState || !symState.obi || !symState.latestTick) return;

    const obiVal = symState.obi.obi;
    const nowMs = Date.now();
    
    let rawSig: TradeSignalType = 'NEUTRAL';
    if (obiVal > 0.30) rawSig = 'STRONG BUY';
    else if (obiVal > 0.18) rawSig = 'BUY';
    else if (obiVal < -0.30) rawSig = 'STRONG SELL';
    else if (obiVal < -0.18) rawSig = 'SELL';

    const currentLatched = latchedSignalMapRef.current[activeSym] || 'NEUTRAL';
    const lastTime = lastSignalTimeMapRef.current[activeSym] || 0;
    const timeSinceChange = nowMs - lastTime;

    let nextLatched = currentLatched;

    if (rawSig !== 'NEUTRAL') {
      if (timeSinceChange > 5000 && (currentLatched === 'NEUTRAL' || (rawSig.includes('BUY') !== currentLatched.includes('BUY')))) {
        nextLatched = rawSig;
        lastSignalTimeMapRef.current[activeSym] = nowMs;

        const markerType = rawSig.includes('BUY') ? 'BUY' as const : 'SELL' as const;
        const newMarker: TradeMarker = {
          id: `${nowMs}-${activeSym}-${Math.random().toString(36).substr(2, 4)}`,
          time: Math.floor(nowMs / 1000),
          timeStr: new Date(nowMs).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
          price: symState.latestTick.last_price,
          type: markerType,
          signal: rawSig,
          symbol: activeSym,
        };

        const rawEquity = paperAccount?.equity;
        const currentEquity = (customCapitalRef.current !== null && paperAccount)
          ? customCapitalRef.current + paperAccount.realized_pnl + paperAccount.unrealized_pnl
          : rawEquity;

        const isStopBreached = currentEquity !== undefined && currentEquity !== null && (
          (targetProfitEquity !== null && currentEquity >= targetProfitEquity) ||
          (maxDrawdownEquity !== null && currentEquity <= maxDrawdownEquity)
        );

        if (isStopBreached) {
          setIsAutoTradeEnabled(false);
          setTargetStopTriggered(true);
        }

        if (isAutoTradeEnabledRef.current && enabledAssetsRef.current[activeSym] !== false && !targetStopTriggeredRef.current && !isStopBreached) {
          setTradeMarkers(prev => [...prev.slice(-49), newMarker]);
          // Dynamic order sizing based on current equity (target ~5% of equity per trade)
          const targetNotional = Math.max(50.0, currentEquity * 0.05);
          const lastPrice = symState.latestTick.last_price || 1.0;
          
          let qty = targetNotional / lastPrice;
          if (activeSym.includes('BTC')) {
            qty = +qty.toFixed(4);
          } else if (activeSym.includes('ETH')) {
            qty = +qty.toFixed(3);
          } else {
            qty = +qty.toFixed(2);
          }
          
          const minQty = activeSym.includes('BTC') ? 0.0001 : activeSym.includes('ETH') ? 0.001 : 0.01;
          qty = Math.max(minQty, qty);

          // Configure high-win-rate momentum parameters: 0.60% Take Profit (quick scalp) and 1.20% Stop Loss
          const stopLoss = markerType === 'BUY' ? lastPrice * 0.988 : lastPrice * 1.012;
          const takeProfit = markerType === 'BUY' ? lastPrice * 1.006 : lastPrice * 0.994;

          submitOrder({
            symbol: activeSym,
            side: markerType,
            quantity: qty,
            order_type: 'MARKET',
            source: 'BOT',
            stop_loss: +stopLoss.toFixed(2),
            take_profit: +takeProfit.toFixed(2),
          });
        }
      } else if (rawSig !== currentLatched) {
        nextLatched = rawSig;
        lastSignalTimeMapRef.current[activeSym] = nowMs;
      }
      latchedActive = true;
    } else {
      if (timeSinceChange < SIGNAL_HOLD_MS && currentLatched !== 'NEUTRAL') {
        nextLatched = currentLatched;
        latchedActive = true;
      } else {
        nextLatched = 'NEUTRAL';
        latchedActive = false;
      }
    }

    latchedSignalMapRef.current[activeSym] = nextLatched;

    if (activeSym === selectedSymbolRef.current) {
      setLatchedSignal(nextLatched);
      setIsSignalLatched(latchedActive);
      setSmoothedObi(obiVal);
    }
  }, [submitOrder]);

  // Subscribe to WebSocket ticks — single shared listener
  useEffect(() => {
    const unsub = ws.subscribe((msg) => {
      const tick = msg as unknown as Tick;
      if (!tick.last_price || !tick.symbol) return;

      const expandedTicks = expandMultiSymbolTicks(tick);
      expandedTicks.forEach(t => pendingTicksRef.current.push(t));
    });
    return unsub;
  }, [ws.subscribe, expandMultiSymbolTicks]);

  // Unified visual update flush timer
  useEffect(() => {
    const intervalMs = refreshSpeed === '10Hz' ? 100 : refreshSpeed === '2Hz' ? 500 : 2000;

    const timer = setInterval(() => {
      if (isPausedRef.current) return;

      const ticksToProcess = [...pendingTicksRef.current];
      pendingTicksRef.current = [];

      let currentMap = symbolDataRef.current;
      if (ticksToProcess.length > 0) {
        for (const t of ticksToProcess) {
          currentMap = applyTick(currentMap, t.symbol, t);
        }
        symbolDataRef.current = currentMap;
        setSymbolData({ ...currentMap });
      }

      Object.keys(currentMap).forEach((symKey) => {
        const sym = symKey as Symbol;
        if (enabledAssetsRef.current[sym] !== false && currentMap[sym]) {
          updateSignalAndMarkers(sym, currentMap[sym]);
        }
      });
    }, intervalMs);

    return () => clearInterval(timer);
  }, [refreshSpeed, applyTick, updateSignalAndMarkers]);

  // Update active symbol metrics immediately on selection change
  useEffect(() => {
    const symState = symbolDataRef.current[selectedSymbol];
    const latched = latchedSignalMapRef.current[selectedSymbol] || 'NEUTRAL';
    const obiVal = symState?.obi?.obi ?? 0;
    const isLatched = latched !== 'NEUTRAL';
    
    setLatchedSignal(latched);
    setIsSignalLatched(isLatched);
    setSmoothedObi(obiVal);

    if (symState) {
      updateSignalAndMarkers(selectedSymbol, symState);
    }
  }, [selectedSymbol, updateSignalAndMarkers]);

  // Poll /api/status — single global poll loop
  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(`${apiConfig.baseUrl}/api/status`);
        if (res.ok) {
          const data = await res.json();
          setStatus(data);
          if (data.prices) {
            statusPricesRef.current = data.prices;
          }
          // Sync target stop triggered flag from server
          if (data.target_stop_triggered !== undefined) {
            setTargetStopTriggered(data.target_stop_triggered);
            if (data.target_stop_triggered) {
              setIsAutoTradeEnabled(false);
            }
          }
          if (data.paper_summary) {
            setPaperAccount(data.paper_summary);
            // Auto-sync frontend custom capital to backend paper engine on restart / discrepancy detection
            if (customCapitalRef.current !== null && 
                Math.abs(data.paper_summary.cash_balance - (customCapitalRef.current + data.paper_summary.realized_pnl)) > 5.0 && 
                Math.abs(data.paper_summary.cash_balance - 100000.0) < 5.0) {
              fetch(`${apiConfig.baseUrl}/api/paper/balance`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ balance: customCapitalRef.current }),
              }).catch(() => {});
            }
          }
          if (data.reloop_telemetry) {
            setReloopTelemetry(data.reloop_telemetry);
          }
          setGatewayConnected(true);
        } else {
          setGatewayConnected(false);
        }
      } catch {
        setGatewayConnected(false);
      }
    };
    poll();
    const id = setInterval(poll, 2000);
    return () => clearInterval(id);
  }, []);

  return { 
    selectedSymbol,
    setSelectedSymbol,
    availableSymbols: AVAILABLE_SYMBOLS,
    symbolData, 
    status, 
    wsConnected: ws.connected,
    gatewayConnected,
    refreshSpeed,
    setRefreshSpeed,
    isPaused,
    setIsPaused,
    isAutoTradeEnabled,
    setIsAutoTradeEnabled,
    enabledAssets,
    toggleAssetTrading,
    targetProfitEquity,
    maxDrawdownEquity,
    targetStopTriggered,
    setAccountTargetStops,
    smoothedObi,
    latchedSignal,
    isSignalLatched,
    tradeMarkers,
    paperAccount,
    submitOrder,
    closePosition,
    resetPaperAccount,
    setPaperBalance,
    reloopTelemetry,
    triggerReloop,
  };
}

type MarketDataContextType = ReturnType<typeof useMarketDataProviderValue>;

const MarketDataContext = createContext<MarketDataContextType | null>(null);

export function MarketDataProvider({ children }: { children: React.ReactNode }) {
  const value = useMarketDataProviderValue();
  return <MarketDataContext.Provider value={value}>{children}</MarketDataContext.Provider>;
}

export function useMarketData() {
  const context = useContext(MarketDataContext);
  if (!context) {
    throw new Error('useMarketData must be used within a MarketDataProvider');
  }
  return context;
}
