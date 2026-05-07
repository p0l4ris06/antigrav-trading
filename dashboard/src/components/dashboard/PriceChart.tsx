import { useMarketData } from '@/hooks/useMarketData';
import { SectionLabel } from './Card';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';

export function PriceChart() {
  const { symbolData, status } = useMarketData();
  const candles = symbolData['BTCUSDT'].candles;
  const latestTick = symbolData['BTCUSDT'].latestTick;
  const obi = symbolData['BTCUSDT'].obi;

  const data = candles.map(c => ({
    time: new Date(c.time * 1000).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    price: c.close,
    high: c.high,
    low: c.low,
  }));

  const prices = candles.map(c => c.close);
  const minP = prices.length ? Math.min(...prices) : 0;
  const maxP = prices.length ? Math.max(...prices) : 0;
  const firstPrice = prices[0] ?? 0;
  const lastPrice = latestTick?.last_price ?? prices[prices.length - 1] ?? 0;
  const isUp = lastPrice >= firstPrice;

  const strokeColor = isUp ? 'var(--green)' : 'var(--red)';
  const gradientId = isUp ? 'priceGradUp' : 'priceGradDown';

  return (
    <div style={{ padding: '16px 16px 8px', height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Title row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 4 }}>
        <div>
          <SectionLabel>Live Alpha · BTCUSDT</SectionLabel>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
            <span style={{ fontSize: 24, fontFamily: 'var(--font-mono)', fontWeight: 300, color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
              ${lastPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
            {obi && (
              <span style={{ fontSize: 11, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>
                {obi.bid_price.toFixed(2)} / {obi.ask_price.toFixed(2)}
              </span>
            )}
          </div>
        </div>
        {prices.length > 1 && (
          <span style={{
            fontSize: 11, padding: '4px 10px', borderRadius: 6,
            fontFamily: 'var(--font-mono)', fontWeight: 500,
            background: isUp ? 'var(--green-muted)' : 'var(--red-muted)',
            color: isUp ? 'var(--green)' : 'var(--red)',
          }}>
            {isUp ? '+' : ''}{((lastPrice - firstPrice) / firstPrice * 100).toFixed(3)}%
          </span>
        )}
      </div>

      {/* Chart */}
      <div style={{ flex: 1, minHeight: 0 }}>
        {data.length < 2 ? (
          <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <span style={{ color: 'var(--text-tertiary)', fontSize: 12 }}>Waiting for data...</span>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 4 }}>
              <defs>
                <linearGradient id="priceGradUp" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#32D74B" stopOpacity={0.25} />
                  <stop offset="100%" stopColor="#32D74B" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="priceGradDown" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#FF453A" stopOpacity={0.25} />
                  <stop offset="100%" stopColor="#FF453A" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="none" stroke="rgba(255,255,255,0.04)" vertical={false} />
              <XAxis
                dataKey="time"
                tick={{ fill: 'rgba(255,255,255,0.26)', fontSize: 9, fontFamily: 'var(--font-mono)' }}
                tickLine={false} axisLine={false}
                interval="preserveStartEnd"
              />
              <YAxis
                domain={[minP * 0.9999, maxP * 1.0001]}
                tick={{ fill: 'rgba(255,255,255,0.26)', fontSize: 9, fontFamily: 'var(--font-mono)' }}
                tickLine={false} axisLine={false} width={72}
                tickFormatter={v => v.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                  borderRadius: 8, fontSize: 11, fontFamily: 'var(--font-mono)',
                }}
                labelStyle={{ color: 'rgba(255,255,255,0.48)' }}
                itemStyle={{ color: 'var(--text-primary)' }}
                formatter={(v: number) => [`$${v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, 'Price']}
                labelFormatter={(label) => `Time: ${label}`}
              />
              <ReferenceLine y={firstPrice} stroke="rgba(255,255,255,0.10)" strokeDasharray="3 3" />
              <Area
                type="monotone" dataKey="price"
                stroke={strokeColor} strokeWidth={1.5}
                fill={`url(#${gradientId})`} dot={false}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
      <div style={{
        padding: '6px 16px 10px',
        display: 'flex', alignItems: 'center', gap: 16,
        borderTop: '1px solid var(--border)',
      }}>
        <span style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>
          TICKS <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
            {status.ticks_ingested.toLocaleString()}
          </span>
        </span>
        <span style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>
          BUFFER <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
            {status.buffer_height}
          </span>
        </span>
        <span style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>
          CANDLES <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
            {candles.length}
          </span>
        </span>
      </div>
    </div>
  );
}
