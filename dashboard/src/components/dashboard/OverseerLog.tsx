import { useEffect, useRef } from 'react';
import { useMarketData } from '@/hooks/useMarketData';
import { SectionLabel } from './Card';

type OverseerEvent = { timestamp?: string; event?: string; pid?: number; sharpe?: number; [key: string]: unknown };

const EVENT_COLOR: Record<string, string> = {
  regime_refitted: 'var(--green)',
  regime_refit_triggered: 'var(--blue)',
  drift_detected: 'var(--amber)',
  shadow_fork_spawning: 'var(--blue)',
  shadow_fork_started: 'var(--blue)',
  shadow_fork_promoted: 'var(--green)',
  kill_switch: 'var(--red)',
};

function formatEvent(e: OverseerEvent): { label: string; detail: string; color: string } {
  const event = e.event ?? 'unknown';
  const color = EVENT_COLOR[event] ?? 'var(--text-secondary)';
  const time = e.timestamp
    ? new Date(e.timestamp).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    : '';

  let detail = '';
  if (e.sharpe !== undefined) detail = `sharpe=${Number(e.sharpe).toFixed(3)}`;
  if (e.pid !== undefined) detail += ` pid=${e.pid}`;

  return { label: event.replace(/_/g, ' ').toUpperCase(), detail: detail.trim(), color };
}

export function OverseerLog() {
  const { status } = useMarketData();
  const ref = useRef<HTMLDivElement>(null);
  const events = (status.overseer_events ?? []) as OverseerEvent[];

  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [events.length]);

  const deduped = events
    .reduceRight<typeof events>((acc, e) => {
      if (acc.length === 0 || acc[0].event !== e.event) acc.unshift(e);
      return acc;
    }, [])
    .slice(-40); // show last 40 unique events only

  return (
    <div style={{ padding: '14px 14px', display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>
      <SectionLabel>Overseer Events</SectionLabel>
      <div ref={ref} style={{ flex: 1, overflowY: 'auto', scrollbarWidth: 'none', minHeight: 0 }}>
        {events.length === 0 ? (
          <p style={{ fontSize: 11, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>Monitoring...</p>
        ) : (
          deduped.map((e, i) => {
            const { label, detail, color } = formatEvent(e);
            const time = e.timestamp
              ? new Date(e.timestamp).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
              : '';
            return (
              <div key={i} style={{
                display: 'flex', gap: 8, padding: '5px 0',
                borderBottom: '1px solid rgba(255,255,255,0.04)',
                alignItems: 'flex-start',
              }}>
                <span style={{ fontSize: 9, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)', flexShrink: 0, paddingTop: 1 }}>{time}</span>
                <div>
                  <span style={{ fontSize: 10, fontWeight: 600, color, letterSpacing: '0.04em' }}>{label}</span>
                  {detail && <span style={{ fontSize: 10, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)', marginLeft: 6 }}>{detail}</span>}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
