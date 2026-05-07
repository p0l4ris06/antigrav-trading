import { useState, useEffect, useRef } from 'react';
import { useWebSocket } from './useWebSocket';

const WS_URL = `${import.meta.env.VITE_WS_BASE_URL ?? 'ws://localhost:8000'}/ws/simulated_feed`;
const API_URL = `${import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'}/api/status`;

export const SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'NVDAUSDT'] as const;
export type Symbol = typeof SYMBOLS[number];

export const SYMBOL_META: Record<Symbol, { label: string; basePrice: number; volatility: number }> = {
  BTCUSDT:  { label: 'BTC / USDT',  basePrice: 50000, volatility: 5.0  },
  ETHUSDT:  { label: 'ETH / USDT',  basePrice: 3000,  volatility: 2.0  },
  SOLUSDT:  { label: 'SOL / USDT',  basePrice: 170,   volatility: 0.8  },
  NVDAUSDT: { label: 'NVDA / USD',  basePrice: 138,   volatility: 0.4  },
};

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
};

// Per-symbol simulated state (for non-BTC symbols the backend doesn't stream)
type SymbolState = {
  latestTick: Tick | null;
  candles: Candle[];
  obi: OBISnapshot | null;
};

const BUCKET = 5;

function makeSimTick(symbol: Symbol, prevPrice: number): Tick {
  const meta = SYMBOL_META[symbol];
  const price = prevPrice + (Math.random() - 0.5) * meta.volatility * 2;
  const spread = Math.abs((Math.random() - 0.5) * meta.volatility * 0.1);
  return {
    symbol,
    bid_price: +(price - spread / 2).toFixed(2),
    ask_price: +(price + spread / 2).toFixed(2),
    bid_size: +(Math.random() * 2).toFixed(4),
    ask_size: +(Math.random() * 2).toFixed(4),
    last_price: +price.toFixed(2),
    last_size: +(Math.random() * 0.5).toFixed(4),
    trade_id: Date.now(),
  };
}

export function useMarketData() {
  const ws = useWebSocket(WS_URL);

  const [symbolData, setSymbolData] = useState<Record<Symbol, SymbolState>>(() =>
    Object.fromEntries(SYMBOLS.map(s => [s, { latestTick: null, candles: [], obi: null }])) as Record<Symbol, SymbolState>
  );

  const [status, setStatus] = useState<SystemStatus>({
    current_regime: '—', overseer_state: 'MONITORING',
    rolling_sharpe: 0, drift_detected: false,
    shadow_fork_active: false, ticks_ingested: 0,
    buffer_height: 0, uptime_seconds: 0,
    overseer_events: [], regime_probabilities: [], portfolio_weights: [],
    ppo_model_loaded: false,
  });

  const simPrices = useRef<Record<Symbol, number>>(
    Object.fromEntries(SYMBOLS.map(s => [s, SYMBOL_META[s].basePrice])) as Record<Symbol, number>
  );

  function applyTick(prev: Record<Symbol, SymbolState>, sym: Symbol, tick: Tick): Record<Symbol, SymbolState> {
    const now = Math.floor(Date.now() / 1000);
    const bucket = now - (now % BUCKET);
    const price = tick.last_price;
    const total = tick.bid_size + tick.ask_size;

    const obi: OBISnapshot = {
      bid_price: tick.bid_price, ask_price: tick.ask_price,
      bid_size: tick.bid_size, ask_size: tick.ask_size,
      spread: +(tick.ask_price - tick.bid_price).toFixed(2),
      obi: total > 0 ? +((tick.bid_size - tick.ask_size) / total).toFixed(4) : 0,
    };

    const prevCandles = prev[sym].candles;
    const last = prevCandles[prevCandles.length - 1];
    let candles: Candle[];
    if (last && last.time === bucket) {
      const updated = { ...last, high: Math.max(last.high, price), low: Math.min(last.low, price), close: price, volume: last.volume + tick.last_size };
      candles = [...prevCandles.slice(0, -1), updated];
    } else {
      candles = [...prevCandles.slice(-199), { time: bucket, open: last?.close ?? price, high: price, low: price, close: price, volume: tick.last_size }];
    }

    return { ...prev, [sym]: { latestTick: tick, candles, obi } };
  }

  // Real BTC stream from backend
  useEffect(() => {
    const unsub = ws.subscribe((msg) => {
      const tick = msg as unknown as Tick;
      if (!tick.last_price) return;
      simPrices.current['BTCUSDT'] = tick.last_price;
      setSymbolData(prev => applyTick(prev, 'BTCUSDT', { ...tick, symbol: 'BTCUSDT' }));
    });
    return unsub;
  }, [ws.subscribe]);

  // Simulated streams for other symbols at same ~10/s rate
  useEffect(() => {
    const otherSymbols = SYMBOLS.filter(s => s !== 'BTCUSDT');
    const intervals = otherSymbols.map(sym =>
      setInterval(() => {
        const price = simPrices.current[sym];
        const tick = makeSimTick(sym, price);
        simPrices.current[sym] = tick.last_price;
        setSymbolData(prev => applyTick(prev, sym, tick));
      }, 100)
    );
    return () => intervals.forEach(clearInterval);
  }, []);

  // Poll /api/status every 2s
  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(API_URL);
        if (res.ok) setStatus(await res.json());
      } catch {}
    };
    poll();
    const id = setInterval(poll, 2000);
    return () => clearInterval(id);
  }, []);

  return { symbolData, status, connected: ws.connected };
}
