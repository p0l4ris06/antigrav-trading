import { useState } from 'react';
import { useMarketData, SYMBOLS, SYMBOL_META, Symbol } from '@/hooks/useMarketData';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from 'recharts';

function MiniChart({ symbol, active, onClick }: { symbol: Symbol; active: boolean; onClick: () => void }) {
  const { symbolData } = useMarketData();
  const data = symbolData[symbol];
  const candles = data.candles.slice(-80);
  const tick = data.latestTick;
  const obi = data.obi;

  const chartData = candles.map(c => ({
    time: new Date(c.time * 1000).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    price: c.close,
  }));

  const prices = candles.map(c => c.close);
  const first = prices[0] ?? 0;
  const last = tick?.last_price ?? prices[prices.length - 1] ?? 0;
  const isUp = last >= first;
  const changePct = first > 0 ? ((last - first) / first) * 100 : 0;
  const strokeColor = isUp ? '#32D74B' : '#FF453A';
  const gradId = `grad-${symbol}`;

  const obiSignal = !obi ? '—'
    : obi.obi > 0.05 ? '▲ BUY'
    : obi.obi < -0.05 ? '▼ SELL'
    : '— NEUTRAL';
  const obiColor = !obi ? 'var(--text-tertiary)'
    : obi.obi > 0.05 ? 'var(--green)'
    : obi.obi < -0.05 ? 'var(--red)'
    : 'var(--text-secondary)';

  return (
    <div
      onClick={onClick}
      style={{
        background: active ? 'var(--bg-elevated)' : 'var(--bg-surface)',
        border: `1px solid ${active ? 'rgba(255,255,255,0.16)' : 'var(--border)'}`,
        borderRadius: 12,
        display: 'flex', flexDirection: 'column',
        overflow: 'hidden', cursor: 'pointer',
        transition: 'all 200ms cubic-bezier(0.22,1,0.36,1)',
        boxShadow: active ? '0 0 0 1px rgba(255,255,255,0.08)' : 'none',
      }}
    >
      {/* Header */}
      <div style={{ padding: '12px 14px 0', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontSize: 10, color: 'var(--text-tertiary)', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 3 }}>
            {SYMBOL_META[symbol].label}
          </div>
          <div style={{ fontSize: 18, fontFamily: 'var(--font-mono)', fontWeight: 300, color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
            ${last.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4 }}>
          <span style={{
            fontSize: 10, padding: '2px 8px', borderRadius: 5, fontFamily: 'var(--font-mono)',
            background: isUp ? 'rgba(50,215,75,0.12)' : 'rgba(255,69,58,0.12)',
            color: isUp ? '#32D74B' : '#FF453A',
          }}>
            {isUp ? '+' : ''}{changePct.toFixed(3)}%
          </span>
          <span style={{ fontSize: 10, color: obiColor, fontFamily: 'var(--font-mono)' }}>{obiSignal}</span>
        </div>
      </div>

      {/* Chart */}
      <div style={{ flex: 1, minHeight: 0, padding: '6px 0 0' }}>
        {chartData.length < 2 ? (
          <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <span style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>Loading...</span>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 2, right: 0, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={strokeColor} stopOpacity={0.20} />
                  <stop offset="100%" stopColor={strokeColor} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="none" stroke="rgba(255,255,255,0.03)" vertical={false} />
              <XAxis dataKey="time" hide />
              <YAxis domain={['auto', 'auto']} hide />
              <Tooltip
                contentStyle={{
                  background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                  borderRadius: 8, fontSize: 10, fontFamily: 'var(--font-mono)',
                }}
                labelStyle={{ color: 'var(--text-tertiary)' }}
                itemStyle={{ color: 'var(--text-primary)' }}
                formatter={(v: number) => [`$${v.toLocaleString('en-US', { minimumFractionDigits: 2 })}`, 'Price']}
                labelFormatter={(l) => `Time: ${l}`}
              />
              <Area
                type="monotone" dataKey="price"
                stroke={strokeColor} strokeWidth={1.5}
                fill={`url(#${gradId})`} dot={false}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Footer */}
      <div style={{ padding: '6px 14px 10px', display: 'flex', gap: 12 }}>
        {obi && <>
          <span style={{ fontSize: 9, color: 'var(--text-tertiary)' }}>
            SPREAD <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>${obi.spread}</span>
          </span>
          <span style={{ fontSize: 9, color: 'var(--text-tertiary)' }}>
            OBI <span style={{ fontFamily: 'var(--font-mono)', color: obiColor }}>{obi.obi.toFixed(3)}</span>
          </span>
        </>}
        {symbol !== 'BTCUSDT' && (
          <span style={{ fontSize: 9, color: 'var(--text-tertiary)', marginLeft: 'auto', fontStyle: 'italic' }}>simulated</span>
        )}
      </div>
    </div>
  );
}

export function ChartGrid() {
  const [activeSymbol, setActiveSymbol] = useState<Symbol>('BTCUSDT');

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: '12px', gap: 10 }}>
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gridTemplateRows: '1fr 1fr',
        gap: 10, flex: 1, minHeight: 0,
      }}>
        {SYMBOLS.map(sym => (
          <MiniChart
            key={sym}
            symbol={sym}
            active={activeSymbol === sym}
            onClick={() => setActiveSymbol(sym)}
          />
        ))}
      </div>
    </div>
  );
}
