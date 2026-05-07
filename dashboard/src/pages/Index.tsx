import { useState } from 'react';
import { Header } from '@/components/dashboard/Header';
import { ChartGrid } from '@/components/dashboard/ChartGrid';
import { RegimePanel } from '@/components/dashboard/RegimePanel';
import { OBIPanel } from '@/components/dashboard/OBIPanel';
import { OverseerLog } from '@/components/dashboard/OverseerLog';
import { AccountModal } from '@/components/dashboard/AccountModal';

export default function Index() {
  const [showAccount, setShowAccount] = useState(false);

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg-base)', overflow: 'hidden' }}>
      <Header onAccountClick={() => setShowAccount(true)} />

      <main style={{
        flex: 1, minHeight: 0, display: 'grid',
        gridTemplateColumns: '260px 1fr 240px',
        gap: 0,
      }}>
        {/* LEFT */}
        <aside style={{ borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <RegimePanel />
        </aside>

        {/* CENTRE — 2×2 chart grid */}
        <section style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', background: 'var(--bg-base)' }}>
          <ChartGrid />
        </section>

        {/* RIGHT */}
        <aside style={{ borderLeft: '1px solid var(--border)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <OBIPanel />
          <div style={{ flex: 1, borderTop: '1px solid var(--border)', minHeight: 0 }}>
            <OverseerLog />
          </div>
        </aside>
      </main>

      {showAccount && <AccountModal onClose={() => setShowAccount(false)} />}
    </div>
  );
}
