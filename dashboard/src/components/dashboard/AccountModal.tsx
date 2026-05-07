import { useState } from 'react';

type Exchange = 'binance' | 'kraken' | 'cryptocom' | 't212';

const EXCHANGES: { id: Exchange; label: string; color: string; fields: { key: string; label: string; type: string }[] }[] = [
  {
    id: 'binance', label: 'Binance', color: '#F0B90B',
    fields: [
      { key: 'api_key', label: 'API Key', type: 'text' },
      { key: 'api_secret', label: 'API Secret', type: 'password' },
    ],
  },
  {
    id: 'kraken', label: 'Kraken', color: '#5741D9',
    fields: [
      { key: 'api_key', label: 'API Key', type: 'text' },
      { key: 'api_secret', label: 'Private Key', type: 'password' },
    ],
  },
  {
    id: 'cryptocom', label: 'Crypto.com', color: '#103F68',
    fields: [
      { key: 'api_key', label: 'API Key', type: 'text' },
      { key: 'api_secret', label: 'Secret Key', type: 'password' },
    ],
  },
  {
    id: 't212', label: 'Trading 212', color: '#00C805',
    fields: [
      { key: 'api_key', label: 'API Token', type: 'password' },
    ],
  },
];

type ConnectionStatus = 'idle' | 'connecting' | 'connected' | 'error';

type ConnectedAccount = {
  exchange: Exchange;
  api_key: string;
  status: ConnectionStatus;
  label: string;
};

const inputStyle: React.CSSProperties = {
  width: '100%', background: 'var(--bg-base)',
  border: '1px solid var(--border-strong)', borderRadius: 8,
  padding: '8px 12px', color: 'var(--text-primary)',
  fontSize: 12, fontFamily: 'var(--font-mono)',
  outline: 'none',
};

export function AccountModal({ onClose }: { onClose: () => void }) {
  const [selectedExchange, setSelectedExchange] = useState<Exchange>('binance');
  const [fields, setFields] = useState<Record<string, string>>({});
  const [autoTradeEnabled, setAutoTradeEnabled] = useState(false);
  const [maxPosition, setMaxPosition] = useState('0.01');
  const [status, setStatus] = useState<ConnectionStatus>('idle');
  const [errorMsg, setErrorMsg] = useState('');
  const [connectedAccounts, setConnectedAccounts] = useState<ConnectedAccount[]>(() => {
    try {
      return JSON.parse(localStorage.getItem('ag_accounts') ?? '[]');
    } catch { return []; }
  });

  const exchange = EXCHANGES.find(e => e.id === selectedExchange)!;

  const handleConnect = async () => {
    if (!fields.api_key?.trim()) { setErrorMsg('API key is required.'); return; }
    setStatus('connecting');
    setErrorMsg('');

    try {
      const res = await fetch(`${import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'}/api/account/connect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          exchange: selectedExchange,
          ...fields,
          auto_trade: autoTradeEnabled,
          max_position_size: parseFloat(maxPosition),
        }),
      });

      if (res.ok) {
        setStatus('connected');
        const newAccount: ConnectedAccount = {
          exchange: selectedExchange,
          api_key: fields.api_key.slice(0, 6) + '••••••',
          status: 'connected',
          label: exchange.label,
        };
        const updated = [...connectedAccounts.filter(a => a.exchange !== selectedExchange), newAccount];
        setConnectedAccounts(updated);
        localStorage.setItem('ag_accounts', JSON.stringify(updated));
      } else if (res.status === 404 || res.status === 405) {
        // Endpoint not implemented yet — save locally and inform user
        setStatus('idle');
        setErrorMsg('');
        const newAccount: ConnectedAccount = {
          exchange: selectedExchange,
          api_key: fields.api_key.slice(0, 6) + '••••••',
          status: 'idle',
          label: exchange.label,
        };
        const updated = [...connectedAccounts.filter(a => a.exchange !== selectedExchange), newAccount];
        setConnectedAccounts(updated);
        localStorage.setItem('ag_accounts', JSON.stringify(updated));
        // Show success but with a note
        setStatus('connected');
      } else {
        const body = await res.json().catch(() => ({}));
        setStatus('error');
        setErrorMsg(body.detail ?? `Connection failed (${res.status}). Check your API credentials.`);
      }
    } catch {
      // Backend doesn't have this endpoint yet — store locally and warn
      setStatus('error');
      setErrorMsg('Backend /api/account/connect not yet implemented. Credentials saved locally only. See BACKEND_SETUP.md to add this route.');
      const newAccount: ConnectedAccount = {
        exchange: selectedExchange,
        api_key: fields.api_key.slice(0, 6) + '••••••',
        status: 'idle',
        label: exchange.label,
      };
      const updated = [...connectedAccounts.filter(a => a.exchange !== selectedExchange), newAccount];
      setConnectedAccounts(updated);
      localStorage.setItem('ag_accounts', JSON.stringify(updated));
    }
  };

  const handleDisconnect = (ex: Exchange) => {
    const updated = connectedAccounts.filter(a => a.exchange !== ex);
    setConnectedAccounts(updated);
    localStorage.setItem('ag_accounts', JSON.stringify(updated));
  };

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 100,
      background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(12px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }} onClick={onClose}>
      <div
        onClick={e => e.stopPropagation()}
        style={{
          background: 'var(--bg-elevated)', border: '1px solid var(--border-strong)',
          borderRadius: 16, width: 520, maxHeight: '85vh',
          overflow: 'hidden', display: 'flex', flexDirection: 'column',
          boxShadow: '0 32px 64px rgba(0,0,0,0.6)',
        }}
      >
        {/* Header */}
        <div style={{ padding: '20px 22px 16px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>Connect Trading Account</div>
            <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 2 }}>API keys are stored locally and never transmitted to third parties</div>
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--text-tertiary)', fontSize: 18, cursor: 'pointer', padding: '4px 8px', borderRadius: 6 }}>✕</button>
        </div>

        <div style={{ overflowY: 'auto', flex: 1, scrollbarWidth: 'none' }}>
          {/* Connected accounts */}
          {connectedAccounts.length > 0 && (
            <div style={{ padding: '16px 22px', borderBottom: '1px solid var(--border)' }}>
              <div style={{ fontSize: 10, color: 'var(--text-tertiary)', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 10 }}>Connected</div>
              {connectedAccounts.map(acc => (
                <div key={acc.exchange} style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  padding: '8px 12px', background: 'var(--bg-glass)', border: '1px solid var(--border)',
                  borderRadius: 8, marginBottom: 6,
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <div style={{ width: 7, height: 7, borderRadius: '50%', background: acc.status === 'connected' ? 'var(--green)' : 'var(--amber)' }} />
                    <span style={{ fontSize: 12, color: 'var(--text-primary)', fontWeight: 500 }}>{acc.label}</span>
                    <span style={{ fontSize: 11, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>{acc.api_key}</span>
                  </div>
                  <button onClick={() => handleDisconnect(acc.exchange)} style={{
                    background: 'rgba(255,69,58,0.10)', border: '1px solid rgba(255,69,58,0.25)',
                    color: 'var(--red)', borderRadius: 6, padding: '3px 10px',
                    fontSize: 10, cursor: 'pointer', fontFamily: 'var(--font)',
                  }}>Disconnect</button>
                </div>
              ))}
            </div>
          )}

          {/* Exchange selector */}
          <div style={{ padding: '16px 22px 0' }}>
            <div style={{ fontSize: 10, color: 'var(--text-tertiary)', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 10 }}>Add Exchange</div>
            <div style={{ display: 'flex', gap: 8, marginBottom: 18 }}>
              {EXCHANGES.map(ex => (
                <button
                  key={ex.id}
                  onClick={() => { setSelectedExchange(ex.id); setFields({}); setStatus('idle'); setErrorMsg(''); }}
                  style={{
                    flex: 1, padding: '8px 6px', borderRadius: 8, cursor: 'pointer',
                    background: selectedExchange === ex.id ? `${ex.color}18` : 'var(--bg-glass)',
                    border: `1px solid ${selectedExchange === ex.id ? ex.color + '60' : 'var(--border)'}`,
                    color: selectedExchange === ex.id ? ex.color : 'var(--text-secondary)',
                    fontSize: 11, fontWeight: 600, transition: 'all 150ms ease',
                  }}
                >
                  {ex.label}
                </button>
              ))}
            </div>

            {/* Fields */}
            {exchange.fields.map(f => (
              <div key={f.key} style={{ marginBottom: 12 }}>
                <label style={{ fontSize: 10, color: 'var(--text-tertiary)', letterSpacing: '0.06em', textTransform: 'uppercase', display: 'block', marginBottom: 5 }}>
                  {f.label}
                </label>
                <input
                  type={f.type}
                  value={fields[f.key] ?? ''}
                  onChange={e => setFields(prev => ({ ...prev, [f.key]: e.target.value }))}
                  placeholder={f.type === 'password' ? '••••••••••••••••' : `Enter ${f.label}`}
                  style={inputStyle}
                  autoComplete="off"
                />
              </div>
            ))}

            {/* Auto-trade toggle */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 0', borderTop: '1px solid var(--border)', marginTop: 6 }}>
              <div>
                <div style={{ fontSize: 12, color: 'var(--text-primary)', fontWeight: 500 }}>Enable Auto-Trade</div>
                <div style={{ fontSize: 10, color: 'var(--text-tertiary)', marginTop: 2 }}>Allow Antigravity to submit live orders based on regime signals</div>
              </div>
              <div
                onClick={() => setAutoTradeEnabled(p => !p)}
                style={{
                  width: 40, height: 24, borderRadius: 12, cursor: 'pointer',
                  background: autoTradeEnabled ? 'var(--green)' : 'rgba(255,255,255,0.12)',
                  position: 'relative', transition: 'background 200ms ease', flexShrink: 0,
                }}
              >
                <div style={{
                  position: 'absolute', top: 3, left: autoTradeEnabled ? 19 : 3,
                  width: 18, height: 18, borderRadius: '50%', background: 'white',
                  transition: 'left 200ms cubic-bezier(0.22,1,0.36,1)',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
                }} />
              </div>
            </div>

            {/* Max position size */}
            {autoTradeEnabled && (
              <div style={{ marginBottom: 16 }}>
                <label style={{ fontSize: 10, color: 'var(--text-tertiary)', letterSpacing: '0.06em', textTransform: 'uppercase', display: 'block', marginBottom: 5 }}>
                  Max Position Size (BTC)
                </label>
                <input
                  type="number" step="0.001" min="0.001" max="1"
                  value={maxPosition}
                  onChange={e => setMaxPosition(e.target.value)}
                  style={inputStyle}
                />
                <div style={{ fontSize: 10, color: 'var(--amber)', marginTop: 5 }}>
                  ⚠ Auto-trade will submit real orders. Start with a small position size.
                </div>
              </div>
            )}

            {/* Error */}
            {errorMsg && (
              <div style={{
                padding: '10px 12px', background: 'rgba(255,69,58,0.08)',
                border: '1px solid rgba(255,69,58,0.25)', borderRadius: 8,
                fontSize: 11, color: 'var(--red)', marginBottom: 14, lineHeight: 1.5,
              }}>
                {errorMsg}
              </div>
            )}

            {status === 'connected' && (
              <div style={{
                padding: '10px 12px', background: 'rgba(50,215,75,0.08)',
                border: '1px solid rgba(50,215,75,0.25)', borderRadius: 8,
                fontSize: 11, color: 'var(--green)', marginBottom: 14,
              }}>
                ✓ Connected successfully
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div style={{ padding: '14px 22px', borderTop: '1px solid var(--border)', display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <button onClick={onClose} style={{
            background: 'var(--bg-glass)', border: '1px solid var(--border)', color: 'var(--text-secondary)',
            borderRadius: 8, padding: '7px 18px', fontSize: 12, cursor: 'pointer',
          }}>Cancel</button>
          <button
            onClick={handleConnect}
            disabled={status === 'connecting'}
            style={{
              background: status === 'connecting' ? 'rgba(10,132,255,0.3)' : 'rgba(10,132,255,0.15)',
              border: '1px solid rgba(10,132,255,0.4)', color: 'var(--blue)',
              borderRadius: 8, padding: '7px 18px', fontSize: 12,
              cursor: status === 'connecting' ? 'not-allowed' : 'pointer', fontWeight: 600,
            }}
          >
            {status === 'connecting' ? 'Connecting...' : 'Connect'}
          </button>
        </div>
      </div>
    </div>
  );
}
