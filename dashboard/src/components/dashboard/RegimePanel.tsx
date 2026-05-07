import { useMarketData } from '@/hooks/useMarketData';
import { SectionLabel } from './Card';

const REGIME_LABELS = ['Trending↑', 'Trending↓', 'Mean Rev', 'Volatile', 'Ranging', 'Breakout'];
const REGIME_COLORS = ['#32D74B', '#FF453A', '#0A84FF', '#FFD60A', '#BF5AF2', '#FF9F0A'];

function Bar({ value, max, color, label, sublabel }: {
  value: number; max: number; color: string; label: string; sublabel?: string;
}) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
        <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{label}</span>
        <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color }}>{sublabel ?? value.toFixed(3)}</span>
      </div>
      <div style={{ height: 4, background: 'rgba(255,255,255,0.06)', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${pct}%`,
          background: color, borderRadius: 2,
          transition: 'width 400ms cubic-bezier(0.22,1,0.36,1)',
          boxShadow: pct > 60 ? `0 0 8px ${color}66` : 'none',
        }} />
      </div>
    </div>
  );
}

export function RegimePanel() {
  const { status } = useMarketData();
  const probs = status.regime_probabilities ?? [];
  const weights = status.portfolio_weights ?? [];

  // Find dominant regime
  const domIdx = probs.indexOf(Math.max(...probs));
  const domLabel = REGIME_LABELS[domIdx] ?? status.current_regime;
  const domColor = REGIME_COLORS[domIdx] ?? 'var(--blue)';
  const domPct = probs[domIdx] ? (probs[domIdx] * 100).toFixed(1) : null;

  const sharpeColor = status.rolling_sharpe > 1.5
    ? 'var(--green)' : status.rolling_sharpe > 0
    ? 'var(--amber)' : 'var(--red)';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>

      {/* Dominant regime hero */}
      <div style={{
        padding: '16px 14px 14px',
        borderBottom: '1px solid var(--border)',
        background: `linear-gradient(135deg, ${domColor}0a 0%, transparent 60%)`,
      }}>
        <SectionLabel>Active Regime</SectionLabel>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
          <span style={{ fontSize: 20, fontWeight: 600, color: domColor, letterSpacing: '-0.02em' }}>
            {domLabel}
          </span>
          {domPct && (
            <span style={{ fontSize: 13, fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)' }}>
              {domPct}%
            </span>
          )}
        </div>
        <div style={{ display: 'flex', gap: 6, marginTop: 10, flexWrap: 'wrap' }}>
          <span style={{
            fontSize: 10, padding: '2px 8px', borderRadius: 4,
            background: status.drift_detected ? 'rgba(255,214,10,0.15)' : 'rgba(50,215,75,0.12)',
            color: status.drift_detected ? 'var(--amber)' : 'var(--green)',
            fontWeight: 500, letterSpacing: '0.06em',
          }}>
            {status.drift_detected ? '⚠ DRIFT' : '✓ STABLE'}
          </span>
          {status.shadow_fork_active && (
            <span style={{
              fontSize: 10, padding: '2px 8px', borderRadius: 4,
              background: 'rgba(10,132,255,0.15)', color: 'var(--blue)',
              fontWeight: 500, letterSpacing: '0.06em',
            }}>
              ⑂ SHADOW FORK
            </span>
          )}
        </div>
      </div>

      {/* Regime probabilities */}
      <div style={{ padding: '14px 14px', borderBottom: '1px solid var(--border)' }}>
        <SectionLabel>Regime Probabilities</SectionLabel>
        {probs.length > 0 ? probs.map((p, i) => (
          <Bar
            key={i}
            label={REGIME_LABELS[i] ?? `Regime ${i}`}
            value={p}
            max={1}
            color={REGIME_COLORS[i] ?? 'var(--blue)'}
            sublabel={`${Math.min(p * 100, 100).toFixed(1)}%`}
          />
        )) : (
          <p style={{ fontSize: 11, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>Awaiting regime data...</p>
        )}
      </div>

      {/* Portfolio weights */}
      {weights.length > 0 && (
        <div style={{ padding: '14px 14px', borderBottom: '1px solid var(--border)' }}>
          <SectionLabel>Portfolio Weights</SectionLabel>
          {weights.map((w, i) => (
            <Bar
              key={i}
              label={`Asset ${i + 1}`}
              value={Math.abs(w)}
              max={1}
              color={w >= 0 ? 'var(--green)' : 'var(--red)'}
              sublabel={`${w >= 0 ? '+' : ''}${(w * 100).toFixed(1)}%`}
            />
          ))}
        </div>
      )}

      {/* System health */}
      <div style={{ padding: '14px 14px', marginTop: 'auto' }}>
        <SectionLabel>System Health</SectionLabel>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          {[
            { label: 'Sharpe', value: status.rolling_sharpe.toFixed(3), color: sharpeColor },
            { label: 'Ticks', value: status.ticks_ingested.toLocaleString(), color: 'var(--text-primary)' },
            { label: 'Buffer', value: `${status.buffer_height}`, color: 'var(--text-primary)' },
            { label: 'Uptime', value: `${(status.uptime_seconds / 3600).toFixed(1)}h`, color: 'var(--text-primary)' },
          ].map(({ label, value, color }) => (
            <div key={label} style={{
              background: 'var(--bg-glass)', border: '1px solid var(--border)',
              borderRadius: 8, padding: '8px 10px',
            }}>
              <div style={{ fontSize: 9, color: 'var(--text-tertiary)', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 3 }}>{label}</div>
              <div style={{ fontSize: 13, fontFamily: 'var(--font-mono)', color, fontWeight: 500 }}>{value}</div>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
