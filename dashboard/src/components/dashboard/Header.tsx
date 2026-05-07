import { useMarketData } from '@/hooks/useMarketData';

export function Header({ onAccountClick }: { onAccountClick: () => void }) {
  const { status, connected, symbolData } = useMarketData();
  const obi = symbolData['BTCUSDT'].obi;
  const latestTick = symbolData['BTCUSDT'].latestTick;

  const isUp = (obi?.obi ?? 0) >= 0;
  const sharpeColor = status.rolling_sharpe > 1 ? 'var(--green)' : status.rolling_sharpe > 0 ? 'var(--amber)' : 'var(--red)';

  const Chip = ({ label, value, color, bg }: { label: string; value: string; color?: string; bg?: string }) => (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 7,
      background: bg ?? 'var(--bg-glass)', border: '1px solid var(--border)',
      borderRadius: 8, padding: '4px 12px',
    }}>
      <span style={{ fontSize: 10, color: 'var(--text-tertiary)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>{label}</span>
      <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', fontWeight: 600, color: color ?? 'var(--text-primary)' }}>{value}</span>
    </div>
  );

  return (
    <header style={{
      height: 52, display: 'flex', alignItems: 'center',
      justifyContent: 'space-between', padding: '0 18px',
      borderBottom: '1px solid var(--border)',
      background: 'rgba(8,8,8,0.92)', backdropFilter: 'blur(24px)',
      flexShrink: 0, gap: 12,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}>
        <span style={{ fontSize: 13, fontWeight: 700, letterSpacing: '-0.03em', color: 'var(--text-primary)' }}>
          ZENITH<span style={{ color: 'var(--text-tertiary)', fontWeight: 400 }}>APEX</span>
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <div style={{
            width: 6, height: 6, borderRadius: '50%',
            background: connected ? 'var(--green)' : 'var(--red)',
            boxShadow: connected ? '0 0 8px rgba(50,215,75,0.8)' : 'none',
          }} />
          <span style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>{connected ? 'LIVE' : 'OFFLINE'}</span>
        </div>
      </div>

      {/* Key trading chips */}
      <div style={{ display: 'flex', gap: 6, flex: 1, justifyContent: 'center', flexWrap: 'nowrap', overflow: 'hidden' }}>
        <Chip label="Regime" value={status.current_regime.toUpperCase()} color="var(--blue)" />
        <Chip
          label="Sharpe"
          value={status.rolling_sharpe.toFixed(3)}
          color={sharpeColor}
          bg={status.rolling_sharpe < 0 ? 'rgba(255,69,58,0.08)' : 'var(--bg-glass)'}
        />
        <Chip
          label="OBI Signal"
          value={obi ? (obi.obi > 0.05 ? '▲ BUY PRESSURE' : obi.obi < -0.05 ? '▼ SELL PRESSURE' : '— NEUTRAL') : '—'}
          color={obi ? (obi.obi > 0.05 ? 'var(--green)' : obi.obi < -0.05 ? 'var(--red)' : 'var(--text-secondary)') : 'var(--text-secondary)'}
          bg={obi ? (obi.obi > 0.05 ? 'rgba(50,215,75,0.08)' : obi.obi < -0.05 ? 'rgba(255,69,58,0.08)' : 'var(--bg-glass)') : 'var(--bg-glass)'}
        />
        <Chip
          label="Drift"
          value={status.drift_detected ? 'DETECTED' : 'STABLE'}
          color={status.drift_detected ? 'var(--amber)' : 'var(--green)'}
        />
        <Chip
          label="Shadow Fork"
          value={status.shadow_fork_active ? 'ACTIVE' : 'IDLE'}
          color={status.shadow_fork_active ? 'var(--blue)' : 'var(--text-tertiary)'}
        />
        {latestTick && (
          <Chip label="Spread" value={`$${obi?.spread ?? '—'}`} />
        )}
        <Chip
          label="Model"
          value={status.ppo_model_loaded ? 'LOADED' : 'MISSING'}
          color={status.ppo_model_loaded ? 'var(--green)' : 'var(--amber)'}
        />
      </div>

      <div style={{ display: 'flex', gap: 6, flexShrink: 0 }}>
        <button
          onClick={onAccountClick}
          style={{
            background: 'rgba(10,132,255,0.10)', border: '1px solid rgba(10,132,255,0.30)',
            color: 'var(--blue)', borderRadius: 8, padding: '4px 14px',
            fontSize: 11, cursor: 'pointer', fontFamily: 'var(--font)', fontWeight: 600,
          }}
        >
          Connect Account
        </button>
        <button
          onClick={async () => {
            await fetch(`${import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'}/api/control/action`,
              { method: 'POST', headers: {'Content-Type':'application/json'},
                body: JSON.stringify({ action: 'retrain' }) });
          }}
          style={{ background: 'var(--bg-glass)', border: '1px solid var(--border-strong)',
            color: 'var(--text-secondary)', borderRadius: 8, padding: '4px 14px',
            fontSize: 11, cursor: 'pointer', fontFamily: 'var(--font)' }}
        >Force Retrain</button>
        <button
          onClick={async () => {
            if (!confirm('Activate kill switch? This halts all live trading.')) return;
            await fetch(`${import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'}/api/control/action`,
              { method: 'POST', headers: {'Content-Type':'application/json'},
                body: JSON.stringify({ action: 'kill' }) });
          }}
          style={{ background: 'rgba(255,69,58,0.10)', border: '1px solid rgba(255,69,58,0.30)',
            color: 'var(--red)', borderRadius: 8, padding: '4px 14px',
            fontSize: 11, cursor: 'pointer', fontFamily: 'var(--font)', fontWeight: 600 }}
        >Kill Switch</button>
      </div>
    </header>
  );
}
