import { useEffect, useRef, useState } from "react";

type Tick = { t: number; p: number };
type Ticker = { symbol: string; label: string; base: number; vol: number; price: number; series: Tick[] };

const TICKERS: Omit<Ticker, "price" | "series">[] = [
  { symbol: "BTCUSDT", label: "BINANCE", base: 64210, vol: 180 },
  { symbol: "ETHUSDT", label: "BINANCE", base: 3142, vol: 12 },
  { symbol: "SOLUSDT", label: "CRYPTO_COM", base: 168.4, vol: 1.2 },
  { symbol: "DOGEUSDT", label: "CRYPTO_COM", base: 0.142, vol: 0.004 },
  { symbol: "AAPL", label: "NASDAQ", base: 224.6, vol: 0.6 },
  { symbol: "TSLA", label: "NASDAQ", base: 261.3, vol: 1.4 },
  { symbol: "NVDA", label: "NASDAQ", base: 138.5, vol: 0.9 },
  { symbol: "MSFT", label: "NASDAQ", base: 432.1, vol: 0.8 },
];

const REGIMES = ["ALPHA_TREND", "MEAN_REVERT", "VOL_SHOCK", "LIQUIDITY_VOID", "MOMENTUM_BURST"];
const ANOMALIES = ["OBI_SPIKE", "ICEBERG_DETECTED", "SPOOF_PATTERN", "LATENCY_ARB", "DARK_POOL_FLOW"];

function formatPrice(p: number) {
  if (p >= 1000) return `$${p.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  if (p >= 1) return `$${p.toFixed(2)}`;
  return `$${p.toFixed(4)}`;
}

function ts(d = new Date()) {
  return d.toTimeString().slice(0, 8);
}

const ApexTerminal = () => {
  const [tickers, setTickers] = useState<Ticker[]>(() =>
    TICKERS.map((t) => ({ ...t, price: t.base, series: [{ t: Date.now(), p: t.base }] }))
  );
  const [regime, setRegime] = useState("INITIALIZING");
  const [stability, setStability] = useState(1.0);
  const [alphaScore, setAlphaScore] = useState(0.0);
  const [obi, setObi] = useState(0.0);
  const [shadowSync, setShadowSync] = useState(0.0);
  const [logs, setLogs] = useState<string[]>([]);
  const [killed, setKilled] = useState(false);
  const chartRef = useRef<HTMLCanvasElement | null>(null);
  const obiRef = useRef<HTMLCanvasElement | null>(null);
  const focusSymbol = "BTCUSDT";

  // API Integration
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Detect if we are running on Vite dev server (e.g. 5173, 8080) and point to gateway (8000)
        const baseUrl = ["5173", "8080", "3000"].includes(window.location.port) ? "http://localhost:8000" : "";
        const res = await fetch(`${baseUrl}/api/status`);
        if (!res.ok) return;
        const data = await res.json();
        
        setRegime(data.current_regime.toUpperCase());
        setAlphaScore(Math.abs(data.rolling_sharpe * 100)); // Map sharpe to a 0-100 score for UI
        setShadowSync(data.shadow_fork_active ? 1.0 : 0.0);
        setStability(data.drift_detected ? 0.6 : 0.98);
        
        // Update focus ticker with real price
        // Update all tickers with real prices from backend if available
        setTickers((prev) =>
          prev.map((tk) => {
            const next = data.prices?.[tk.symbol] || (tk.symbol === focusSymbol ? data.last_price : 0) || tk.price;
            const series = [...tk.series, { t: Date.now(), p: next }].slice(-180);
            return { ...tk, price: next, series };
          })
        );

        if (data.overseer_events) {
          const newLogs = data.overseer_events.map((e: any) => 
            `[${e.timestamp?.slice(11, 19) || ts()}] >> ${e.event} :: ${JSON.stringify(e.data || e)}`
          ).reverse();
          setLogs(newLogs);
        }
      } catch (err) {
        console.error("API Fetch Error:", err);
      }
    };

    const id = setInterval(fetchData, 1000);
    return () => clearInterval(id);
  }, []);

  const handleKill = async () => {
    try {
      const baseUrl = window.location.port === "5173" ? "http://localhost:8000" : "";
      await fetch(`${baseUrl}/api/control/${killed ? "resume" : "pause"}`, { method: "POST" });
      setKilled(!killed);
    } catch (err) {
      console.error("Kill action failed:", err);
    }
  };

  // Live chart canvas
  useEffect(() => {
    const cv = chartRef.current;
    if (!cv) return;
    let raf = 0;
    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = cv.clientWidth;
      const h = cv.clientHeight;
      cv.width = w * dpr;
      cv.height = h * dpr;
      const ctx = cv.getContext("2d")!;
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, w, h);

      const focus = tickers.find((t) => t.symbol === focusSymbol);
      if (!focus || focus.series.length < 2) {
        raf = requestAnimationFrame(draw);
        return;
      }
      const data = focus.series;
      const min = Math.min(...data.map((d) => d.p));
      const max = Math.max(...data.map((d) => d.p));
      const pad = (max - min) * 0.15 || 1;
      const lo = min - pad;
      const hi = max + pad;

      // grid
      ctx.strokeStyle = "hsla(220, 30%, 60%, 0.06)";
      ctx.lineWidth = 1;
      for (let i = 1; i < 5; i++) {
        const y = (h / 5) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      // y-axis labels
      ctx.fillStyle = "hsla(220, 8%, 55%, 0.8)";
      ctx.font = "10px 'JetBrains Mono', monospace";
      for (let i = 0; i <= 4; i++) {
        const v = hi - ((hi - lo) / 4) * i;
        ctx.fillText(v.toFixed(v >= 100 ? 0 : 2), 4, (h / 4) * i + 10);
      }

      // gradient fill area
      const xAt = (i: number) => (i / (data.length - 1)) * (w - 50) + 45;
      const yAt = (p: number) => h - ((p - lo) / (hi - lo)) * (h - 20) - 10;

      const grad = ctx.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, "hsla(70, 95%, 55%, 0.22)");
      grad.addColorStop(1, "hsla(70, 95%, 55%, 0)");
      ctx.beginPath();
      ctx.moveTo(xAt(0), h);
      data.forEach((d, i) => ctx.lineTo(xAt(i), yAt(d.p)));
      ctx.lineTo(xAt(data.length - 1), h);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();

      // line stroke — single acid signal, no glow
      ctx.strokeStyle = "hsl(70, 95%, 60%)";
      ctx.lineWidth = 1.4;
      ctx.shadowBlur = 0;
      ctx.beginPath();
      data.forEach((d, i) => (i === 0 ? ctx.moveTo(xAt(i), yAt(d.p)) : ctx.lineTo(xAt(i), yAt(d.p))));
      ctx.stroke();

      // last point — square marker, brutalist
      const last = data[data.length - 1];
      ctx.fillStyle = "hsl(40, 18%, 95%)";
      ctx.fillRect(xAt(data.length - 1) - 3, yAt(last.p) - 3, 6, 6);

      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [tickers]);

  // OBI depth canvas
  useEffect(() => {
    const cv = obiRef.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth;
    const h = cv.clientHeight;
    cv.width = w * dpr;
    cv.height = h * dpr;
    const ctx = cv.getContext("2d")!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const bars = 40;
    const mid = w / 2;
    for (let i = 0; i < bars; i++) {
      const bidH = Math.random() * (h - 20) * (1 - i / bars);
      const askH = Math.random() * (h - 20) * (1 - i / bars);
      const bw = (mid - 10) / bars;
      // bid (left, acid signal)
      ctx.fillStyle = `hsla(70, 95%, 55%, ${0.18 + (1 - i / bars) * 0.7})`;
      ctx.fillRect(mid - 4 - (i + 1) * bw, h - bidH - 4, bw - 1, bidH);
      // ask (right, bone)
      ctx.fillStyle = `hsla(40, 18%, 92%, ${0.12 + (1 - i / bars) * 0.55})`;
      ctx.fillRect(mid + 4 + i * bw, h - askH - 4, bw - 1, askH);
    }
    // mid line
    ctx.strokeStyle = "hsla(40, 18%, 92%, 0.4)";
    ctx.setLineDash([2, 4]);
    ctx.beginPath();
    ctx.moveTo(mid, 0);
    ctx.lineTo(mid, h);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = "hsla(220, 8%, 55%, 0.9)";
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.fillText("BIDS", 8, 14);
    ctx.textAlign = "right";
    ctx.fillText("ASKS", w - 8, 14);
    ctx.textAlign = "left";
  }, [obi]);

  const focus = tickers.find((t) => t.symbol === focusSymbol);
  const obiBias = obi >= 0 ? "BULLISH" : "BEARISH";

  return (
    <main className="min-h-screen w-full p-4 md:p-6">
      <header
        className="apex-panel flex flex-wrap items-center justify-between gap-4 px-6 py-4 animate-apex-rise"
        style={{ animationDelay: "0ms" }}
      >
        <div className="flex items-center gap-4">
          <div className="apex-logo text-3xl md:text-4xl">
            ZENITH<span className="apex-logo-accent">APEX</span>
          </div>
          <span className="apex-mono text-[0.65rem] text-muted-foreground hidden md:inline">
            ANTIGRAVITY-Ω · MISSION CONTROL · v1.0.M
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <span className="apex-badge flex items-center gap-2">
            <span className="apex-pulse-dot inline-block h-1.5 w-1.5 rounded-full bg-[hsl(var(--apex-cyan))]" />
            REGIME: {regime}
          </span>
          <span className="apex-badge apex-badge-purple flex items-center gap-2">
            <span className="apex-pulse-dot inline-block h-1.5 w-1.5 rounded-full bg-[hsl(var(--apex-purple))]" />
            FORK: {killed ? "HALTED" : "LIVE"}
          </span>
          <button
            onClick={handleKill}
            className="apex-mono apex-hover rounded-md border border-[hsl(var(--destructive))]/60 bg-[hsl(var(--destructive))]/10 px-3 py-2 text-[0.7rem] uppercase tracking-widest text-[hsl(var(--destructive))] hover:-translate-y-0.5 hover:bg-[hsl(var(--destructive))]/20 active:translate-y-0 active:scale-[0.98]"
          >
            ◉ KILL-SWITCH
          </button>
        </div>
      </header>

      {/* KPI strip */}
      <section className="mt-4 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <KpiCard label="Regime Sensor" value={regime} accent="cyan" delay={80} />
        <KpiCard label="Shadow-Fork Sync" value={shadowSync.toFixed(3)} sub={`drift ${(shadowSync * 100).toFixed(1)}%`} accent="purple" delay={140} />
        <KpiCard label="Alpha Score" value={alphaScore.toFixed(1)} sub={`stability ${(stability * 100).toFixed(1)}%`} accent="cyan" delay={200} />
        <KpiCard label="System Health" value={killed ? "HALTED" : "OPTIMAL"} sub="logging depth · deep" accent={killed ? "red" : "green"} delay={260} />
      </section>

      {/* Main grid */}
      <section className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-[320px_1fr_320px]">
        {/* Left: tickers */}
        <div className="apex-panel p-5 animate-apex-rise" style={{ animationDelay: "320ms" }}>
          <h2 className="apex-label mb-4">Global Pulse</h2>
          <ul className="space-y-1">
            {tickers.map((tk) => {
              const first = tk.series[0]?.p ?? tk.price;
              const delta = ((tk.price - first) / first) * 100;
              const up = delta >= 0;
              return (
                <li
                  key={tk.symbol}
                  className="apex-hover flex items-center justify-between border-b border-white/5 py-2.5 hover:translate-x-0.5 hover:border-white/15"
                >
                  <div className="min-w-0">
                    <div className="apex-mono text-[0.6rem] uppercase tracking-widest text-muted-foreground">
                      {tk.label}:{tk.symbol}
                    </div>
                    <div className="apex-mono text-base font-semibold text-foreground">
                      {formatPrice(tk.price)}
                    </div>
                  </div>
                  <div
                    className="apex-mono text-[0.72rem]"
                    style={{
                      color: up
                        ? "hsl(var(--apex-bullish))"
                        : "hsl(var(--apex-bearish))",
                    }}
                  >
                    {up ? "▲" : "▼"} {Math.abs(delta).toFixed(2)}%
                  </div>
                </li>
              );
            })}
          </ul>
        </div>

        {/* Center: live chart */}
        <div className="apex-panel flex h-[480px] flex-col p-5 animate-apex-rise" style={{ animationDelay: "380ms" }}>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="apex-label">Live Alpha Execution · {focusSymbol}</h2>
            <div className="apex-mono text-[0.7rem] text-[hsl(var(--apex-cyan))]">
              {focus ? formatPrice(focus.price) : "—"}
            </div>
          </div>
          <div className="relative flex-1">
            <canvas ref={chartRef} className="h-full w-full animate-apex-meridian" style={{ transformOrigin: "center" }} />
          </div>
        </div>

        {/* Right: overseer audit */}
        <div className="apex-panel flex h-[480px] flex-col p-5 animate-apex-rise" style={{ animationDelay: "440ms" }}>
          <h2 className="apex-label mb-3">Overseer Audit</h2>
          <div className="apex-mono text-[0.72rem] leading-5 text-[hsl(var(--apex-cyan))]/85 overflow-y-auto pr-1 flex-1 space-y-1">
            {logs.map((l, i) => (
              <div key={l + i} className="opacity-90 animate-apex-fade">
                {l}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Bottom panels */}
      <section className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="apex-panel p-5 lg:col-span-2 h-[260px] flex flex-col animate-apex-rise" style={{ animationDelay: "520ms" }}>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="apex-label">Microstructure Depth · OBI</h2>
            <div className="apex-mono text-[0.7rem]">
              <span className="text-muted-foreground">BIAS </span>
              <span
                style={{
                  color:
                    obi >= 0
                      ? "hsl(var(--apex-bullish))"
                      : "hsl(var(--apex-bearish))",
                }}
              >
                {obiBias} {obi.toFixed(2)}
              </span>
            </div>
          </div>
          <canvas ref={obiRef} className="h-full w-full flex-1" />
        </div>
        <div className="apex-panel p-5 h-[260px] animate-apex-rise" style={{ animationDelay: "600ms" }}>
          <h2 className="apex-label mb-3">Organism Vitals</h2>
          <Vital label="Regime Stability" value={stability * 100} suffix="%" color="cyan" />
          <Vital label="Shadow-Fork Sync" value={shadowSync * 100} suffix="%" color="purple" />
          <Vital label="Alpha Score" value={alphaScore} suffix="" color="cyan" />
          <Vital label="OBI Bias" value={Math.abs(obi) * 100} suffix="%" color={obi >= 0 ? "green" : "red"} />
        </div>
      </section>

      <footer className="apex-mono mt-4 flex items-center justify-between px-2 text-[0.65rem] uppercase tracking-widest text-muted-foreground animate-apex-fade" style={{ animationDelay: "700ms" }}>
        <span>ZENITH TERMINAL APEX · BARE-METAL CANVAS RUNTIME</span>
        <span>{killed ? "◉ HALTED" : "● STREAMING"}</span>
      </footer>
    </main>
  );
};

const KpiCard = ({
  label,
  value,
  sub,
  accent,
  delay = 0,
}: {
  label: string;
  value: string;
  sub?: string;
  accent: "cyan" | "purple" | "green" | "red";
  delay?: number;
}) => {
  const color =
    accent === "cyan"
      ? "hsl(var(--apex-cyan))"
      : accent === "purple"
      ? "hsl(var(--apex-purple))"
      : accent === "green"
      ? "hsl(var(--apex-bullish))"
      : "hsl(var(--apex-bearish))";
  return (
    <div
      className="apex-panel apex-hover p-4 animate-apex-rise hover:-translate-y-0.5 hover:border-white/20"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="apex-label">{label}</div>
      <div className="apex-mono mt-2 text-xl font-semibold transition-colors duration-600 ease-[cubic-bezier(0.22,1,0.36,1)]" style={{ color }}>
        {value}
      </div>
      {sub && <div className="apex-mono mt-1 text-[0.65rem] text-muted-foreground">{sub}</div>}
    </div>
  );
};

const Vital = ({
  label,
  value,
  suffix,
  color,
}: {
  label: string;
  value: number;
  suffix: string;
  color: "cyan" | "purple" | "green" | "red";
}) => {
  const c =
    color === "cyan"
      ? "hsl(var(--apex-cyan))"
      : color === "purple"
      ? "hsl(var(--apex-purple))"
      : color === "green"
      ? "hsl(var(--apex-bullish))"
      : "hsl(var(--apex-bearish))";
  const pct = Math.max(0, Math.min(100, value));
  return (
    <div className="mb-3">
      <div className="mb-1 flex justify-between">
        <span className="apex-mono text-[0.65rem] uppercase tracking-widest text-muted-foreground">
          {label}
        </span>
        <span className="apex-mono text-[0.7rem] transition-colors duration-600 ease-[cubic-bezier(0.22,1,0.36,1)]" style={{ color: c }}>
          {value.toFixed(1)}{suffix}
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/5">
        <div
          className="h-full rounded-full"
          style={{
            width: `${pct}%`,
            background: c,
            boxShadow: `0 0 10px ${c}`,
            transition: "width 0.9s cubic-bezier(0.22, 1, 0.36, 1), background-color 0.6s cubic-bezier(0.22, 1, 0.36, 1)",
          }}
        />
      </div>
    </div>
  );
};

export default ApexTerminal;