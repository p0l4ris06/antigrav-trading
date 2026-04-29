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
          const newLogs = data.overseer_events.map((e: any) => {
            const time = e.timestamp?.slice(11, 19) || ts();
            // Clean up event data for display
            const { event, timestamp, ...rest } = e;
            const dataStr = Object.keys(rest).length > 0 ? ` :: ${JSON.stringify(rest)}` : "";
            return `[${time}] ${event.toUpperCase()}${dataStr}`;
          }).reverse();
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
      const baseUrl = ["5173", "8080", "3000"].includes(window.location.port) ? "http://localhost:8000" : "";
      await fetch(`${baseUrl}/api/control/${killed ? "resume" : "pause"}`, { method: "POST" });
      setKilled(!killed);
    } catch (err) {
      console.error("Kill action failed:", err);
    }
  };

  const handleRetrain = async () => {
    try {
      const baseUrl = ["5173", "8080", "3000"].includes(window.location.port) ? "http://localhost:8000" : "";
      await fetch(`${baseUrl}/api/control/force_retrain`, { method: "POST" });
      // Log will show the event
    } catch (err) {
      console.error("Retrain action failed:", err);
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

      // Filter out initialization placeholder if it creates a massive jump
      let data = focus.series;
      if (data.length >= 2 && Math.abs(data[0].p - focus.base) < 0.0001 && Math.abs(data[1].p - data[0].p) > (data[1].p * 0.001)) {
        data = data.slice(1);
      }

      const min = Math.min(...data.map((d) => d.p));
      const max = Math.max(...data.map((d) => d.p));
      const labelCount = 5;

      const niceNum = (range: number, round: boolean) => {
        const exponent = Math.floor(Math.log10(range));
        const fraction = range / Math.pow(10, exponent);
        let niceFraction;
        if (round) {
          if (fraction < 1.5) niceFraction = 1;
          else if (fraction < 3) niceFraction = 2;
          else if (fraction < 7) niceFraction = 5;
          else niceFraction = 10;
        } else {
          if (fraction <= 1) niceFraction = 1;
          else if (fraction <= 2) niceFraction = 2;
          else if (fraction <= 5) niceFraction = 5;
          else niceFraction = 10;
        }
        return niceFraction * Math.pow(10, exponent);
      };

      const range = niceNum(max - min || min * 0.001, false);
      const step = niceNum(range / (labelCount - 1), true);
      const lo = Math.floor(min / step) * step;
      const hi = Math.ceil(max / step) * step;

      // Adjust hi to ensure exactly labelCount intervals if possible, 
      // or at least cover the range.
      const actualHi = Math.max(hi, lo + step * (labelCount - 1));

      const leftPad = 85;
      const rightPad = 25;
      const topPad = 30;
      const bottomPad = 40;
      const chartW = w - leftPad - rightPad;
      const chartH = h - topPad - bottomPad;

      const xAt = (i: number) => (i / (data.length - 1)) * chartW + leftPad;
      const yAt = (p: number) => topPad + chartH - ((p - lo) / (actualHi - lo)) * chartH;

      // grid & y-axis labels
      ctx.strokeStyle = "hsla(220, 20%, 60%, 0.04)";
      ctx.lineWidth = 1;
      ctx.fillStyle = "hsla(220, 10%, 65%, 0.8)";
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.textAlign = "right";

      for (let i = 0; i < labelCount; i++) {
        const y = topPad + (chartH / (labelCount - 1)) * i;
        const val = actualHi - ((actualHi - lo) / (labelCount - 1)) * i;
        
        ctx.beginPath();
        ctx.moveTo(leftPad, y);
        ctx.lineTo(w - rightPad, y);
        ctx.stroke();

        ctx.fillText(
          val.toLocaleString(undefined, { 
            minimumFractionDigits: focus.price > 100 ? 2 : 4,
            maximumFractionDigits: focus.price > 100 ? 2 : 4 
          }), 
          leftPad - 15, 
          y + 3
        );
      }

      // x-axis base line
      ctx.strokeStyle = "hsla(220, 20%, 60%, 0.15)";
      ctx.beginPath();
      ctx.moveTo(leftPad, topPad + chartH);
      ctx.lineTo(w - rightPad, topPad + chartH);
      ctx.stroke();

      // x-axis time labels (3 labels)
      ctx.textAlign = "center";
      const xLabels = 3;
      for (let i = 0; i < xLabels; i++) {
        const idx = Math.floor((i / (xLabels - 1)) * (data.length - 1));
        const d = data[idx];
        const x = xAt(idx);
        const timeStr = new Date(d.t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
        ctx.fillText(timeStr, x, h - 10);
      }

      // Clip the drawing area - allow some overflow at top for line thickness, but strict at bottom
      ctx.save();
      ctx.beginPath();
      ctx.rect(leftPad, topPad - 4, chartW, chartH); 
      ctx.clip();

      // gradient fill area
      const grad = ctx.createLinearGradient(0, topPad, 0, topPad + chartH);
      grad.addColorStop(0, "hsla(70, 95%, 55%, 0.12)");
      grad.addColorStop(1, "hsla(70, 95%, 55%, 0)");
      
      ctx.beginPath();
      ctx.moveTo(xAt(0), topPad + chartH);
      data.forEach((d, i) => ctx.lineTo(xAt(i), yAt(d.p)));
      ctx.lineTo(xAt(data.length - 1), topPad + chartH);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();

      // line stroke with glow
      ctx.shadowBlur = 12;
      ctx.shadowColor = "hsla(70, 95%, 60%, 0.45)";
      ctx.strokeStyle = "hsl(70, 95%, 60%)";
      ctx.lineWidth = 2.8;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      data.forEach((d, i) => (i === 0 ? ctx.moveTo(xAt(i), yAt(d.p)) : ctx.lineTo(xAt(i), yAt(d.p))));
      ctx.stroke();
      ctx.shadowBlur = 0;
      
      ctx.restore();

      // --- Final Overlay Pass (Grid & Labels again to ensure they stay on top) ---
      ctx.strokeStyle = "hsla(220, 20%, 60%, 0.08)";
      ctx.lineWidth = 1;
      for (let i = 0; i < labelCount; i++) {
        const y = topPad + (chartH / (labelCount - 1)) * i;
        ctx.beginPath();
        ctx.moveTo(leftPad, y);
        ctx.lineTo(w - rightPad, y);
        ctx.stroke();
      }
      // x-axis floor
      ctx.strokeStyle = "hsla(220, 20%, 60%, 0.25)";
      ctx.beginPath();
      ctx.moveTo(leftPad, topPad + chartH);
      ctx.lineTo(w - rightPad, topPad + chartH);
      ctx.stroke();

      // last point focus
      const last = data[data.length - 1];
      const lx = xAt(data.length - 1);
      const ly = yAt(last.p);
      
      ctx.fillStyle = "white";
      ctx.shadowBlur = 10;
      ctx.shadowColor = "hsl(70, 95%, 55%)";
      ctx.beginPath();
      ctx.arc(lx, ly, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      // horizontal price line
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "hsla(70, 95%, 60%, 0.4)";
      ctx.beginPath();
      ctx.moveTo(leftPad, ly);
      ctx.lineTo(lx, ly);
      ctx.stroke();
      ctx.setLineDash([]);

      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [tickers, focusSymbol]);

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
      
      // bid (left, acid signal) - with gradient-like alpha
      const bAlpha = 0.2 + (1 - i / bars) * 0.7;
      ctx.fillStyle = `hsla(70, 95%, 55%, ${bAlpha})`;
      // rounded top for bars
      const bx = mid - 4 - (i + 1) * bw;
      const by = h - bidH - 4;
      ctx.fillRect(bx, by, bw - 1, bidH);
      
      // ask (right, bone)
      const aAlpha = 0.15 + (1 - i / bars) * 0.55;
      ctx.fillStyle = `hsla(40, 18%, 92%, ${aAlpha})`;
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
    <main className="h-screen w-full flex flex-col gap-2 p-2 bg-black/95 overflow-hidden">
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
            onClick={handleRetrain}
            className="apex-mono apex-hover rounded-md border border-[hsl(var(--apex-cyan))]/40 bg-[hsl(var(--apex-cyan))]/5 px-3 py-2 text-[0.7rem] uppercase tracking-widest text-[hsl(var(--apex-cyan))] hover:-translate-y-0.5 hover:bg-[hsl(var(--apex-cyan))]/15 active:translate-y-0 active:scale-[0.98]"
          >
            ↻ FORCE RETRAIN
          </button>
          <button
            onClick={handleKill}
            className="apex-mono apex-hover rounded-md border border-[hsl(var(--destructive))]/60 bg-[hsl(var(--destructive))]/10 px-3 py-2 text-[0.7rem] uppercase tracking-widest text-[hsl(var(--destructive))] hover:-translate-y-0.5 hover:bg-[hsl(var(--destructive))]/20 active:translate-y-0 active:scale-[0.98]"
          >
            ◉ KILL-SWITCH
          </button>
        </div>
      </header>

      {/* KPI strip */}
      <section className="grid grid-cols-2 gap-2 lg:grid-cols-4 flex-none">
        <KpiCard label="Regime Sensor" value={regime} accent="cyan" delay={80} />
        <KpiCard label="Shadow-Fork Sync" value={shadowSync.toFixed(3)} sub={`drift ${(shadowSync * 100).toFixed(1)}%`} accent="purple" delay={140} />
        <KpiCard label="Alpha Score" value={alphaScore.toFixed(1)} sub={`stability ${(stability * 100).toFixed(1)}%`} accent="cyan" delay={200} />
        <KpiCard label="System Health" value={killed ? "HALTED" : "OPTIMAL"} sub="logging depth · deep" accent={killed ? "red" : "green"} delay={260} />
      </section>

      {/* Main grid */}
      <section className="flex-[1.4] min-h-0 grid grid-cols-1 gap-2 lg:grid-cols-[280px_1fr_300px]">
        {/* Left: tickers */}
        <div className="apex-panel p-4 animate-apex-rise h-full overflow-hidden flex flex-col" style={{ animationDelay: "320ms" }}>
          <h2 className="apex-label mb-4">Global Pulse</h2>
          <ul className="space-y-1 overflow-y-auto no-scrollbar flex-1">
            {tickers.map((tk) => {
              const first = tk.series[0]?.p ?? tk.price;
              const delta = ((tk.price - first) / first) * 100;
              const up = delta >= 0;
              return (
                <li
                  key={tk.symbol}
                  className="apex-hover flex items-center justify-between border-b border-white/5 py-1.5 hover:translate-x-0.5 hover:border-white/15"
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
        <div className="apex-panel flex h-full flex-col p-4 animate-apex-rise" style={{ animationDelay: "380ms" }}>
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
        <div className="apex-panel flex h-full flex-col p-4 animate-apex-rise" style={{ animationDelay: "440ms" }}>
          <h2 className="apex-label mb-3">Overseer Audit</h2>
          <div className="apex-mono text-[0.72rem] leading-5 text-[hsl(var(--apex-cyan))]/85 overflow-y-auto pr-1 flex-1 space-y-1 no-scrollbar">
            {logs.map((l, i) => (
              <div key={l + i} className="opacity-90 animate-apex-fade">
                {l}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Bottom panels */}
      <section className="flex-1 min-h-0 grid grid-cols-1 gap-2 lg:grid-cols-3">
        <div className="apex-panel p-4 lg:col-span-2 h-full flex flex-col animate-apex-rise" style={{ animationDelay: "520ms" }}>
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
        <div className="apex-panel p-4 h-full animate-apex-rise overflow-hidden" style={{ animationDelay: "600ms" }}>
          <h2 className="apex-label mb-3">Organism Vitals</h2>
          <Vital label="Regime Stability" value={stability * 100} suffix="%" color="cyan" />
          <Vital label="Shadow-Fork Sync" value={shadowSync * 100} suffix="%" color="purple" />
          <Vital label="Alpha Score" value={alphaScore} suffix="" color="cyan" />
          <Vital label="OBI Bias" value={Math.abs(obi) * 100} suffix="%" color={obi >= 0 ? "green" : "red"} />
        </div>
      </section>

      <footer className="flex-none apex-mono mt-auto flex items-center justify-between px-2 text-[0.65rem] uppercase tracking-widest text-muted-foreground animate-apex-fade" style={{ animationDelay: "700ms" }}>
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
      className="apex-panel apex-hover p-3 animate-apex-rise hover:-translate-y-0.5 hover:border-white/20"
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