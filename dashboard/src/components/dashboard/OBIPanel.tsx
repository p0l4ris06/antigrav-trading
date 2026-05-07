import { useMarketData } from '@/hooks/useMarketData';
import { SectionLabel } from './Card';

export function OBIPanel() {
  const { symbolData } = useMarketData();
  const obi = symbolData['BTCUSDT'].obi;
  const latestTick = symbolData['BTCUSDT'].latestTick;

  const signal = !obi ? 'NEUTRAL'
    : obi.obi > 0.1 ? 'STRONG BUY'
    : obi.obi > 0.05 ? 'BUY'
    : obi.obi < -0.1 ? 'STRONG SELL'
    : obi.obi < -0.05 ? 'SELL'
    : 'NEUTRAL';

  const sigColor = signal.includes('BUY') ? 'var(--green)'
    : signal.includes('SELL') ? 'var(--red)'
    : 'var(--text-secondary)';

  const obiVal = obi?.obi ?? 0;
  const obiAbs = Math.abs(obiVal);
  const bidPct = obi ? (obi.bid_size / (obi.bid_size + obi.ask_size)) * 100 : 50;

  return (
    <div style={{ padding: '14px 14px', borderBottom: '1px solid var(--border)' }}>
      <SectionLabel>Order Book Imbalance</SectionLabel>

      {/* Signal */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 14,
      }}>
        <span style={{
          fontSize: 16, fontWeight: 700, color: sigColor,
          letterSpacing: '-0.01em',
          textShadow: signal !== 'NEUTRAL' ? `0 0 20px ${sigColor}66` : 'none',
          transition: 'all 300ms ease',
        }}>
          {signal}
        </span>
        <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: sigColor }}>
          {obiVal >= 0 ? '+' : ''}{obiVal.toFixed(4)}
        </span>
      </div>

      {/* Bid/Ask size bar */}
      <div style={{ marginBottom: 10 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
          <span style={{ fontSize: 10, color: 'var(--green)' }}>BID {obi?.bid_size.toFixed(3) ?? '—'}</span>
          <span style={{ fontSize: 10, color: 'var(--red)' }}>ASK {obi?.ask_size.toFixed(3) ?? '—'}</span>
        </div>
        <div style={{ height: 8, borderRadius: 4, overflow: 'hidden', display: 'flex', gap: 1 }}>
          <div style={{
            width: `${bidPct}%`, background: 'var(--green)',
            transition: 'width 300ms cubic-bezier(0.22,1,0.36,1)',
          }} />
          <div style={{ flex: 1, background: 'var(--red)' }} />
        </div>
      </div>

      {/* Imbalance bar (centred) */}
      <div style={{ marginBottom: 14 }}>
        <div style={{ height: 4, background: 'rgba(255,255,255,0.06)', borderRadius: 2, position: 'relative', overflow: 'hidden' }}>
          <div style={{
            position: 'absolute', top: 0,
            left: obiVal >= 0 ? '50%' : `${50 - obiAbs * 50}%`,
            width: `${obiAbs * 50}%`,
            height: '100%',
            background: obiVal >= 0 ? 'var(--green)' : 'var(--red)',
            borderRadius: 2,
            transition: 'all 300ms cubic-bezier(0.22,1,0.36,1)',
          }} />
          <div style={{ position: 'absolute', left: '50%', top: 0, width: 1, height: '100%', background: 'rgba(255,255,255,0.2)' }} />
        </div>
      </div>

      {/* Stats grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
        {[
          { label: 'Bid', value: obi?.bid_price.toFixed(2) ?? '—' },
          { label: 'Ask', value: obi?.ask_price.toFixed(2) ?? '—' },
          { label: 'Spread', value: obi ? `$${obi.spread}` : '—' },
          { label: 'Last Sz', value: latestTick?.last_size.toFixed(4) ?? '—' },
        ].map(({ label, value }) => (
          <div key={label} style={{ background: 'var(--bg-glass)', border: '1px solid var(--border)', borderRadius: 6, padding: '5px 8px' }}>
            <div style={{ fontSize: 9, color: 'var(--text-tertiary)', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 2 }}>{label}</div>
            <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
