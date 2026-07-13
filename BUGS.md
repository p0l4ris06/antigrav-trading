# BUGS.md — Phase 0 Audit Findings & Fixes

Branch: `pivot/4h-microstructure`  
Audit date: 2026-07-13  
All file references are post-refactor names.

---

## Bug 1 — `core/features.py` L62–L73: Swing Lookback Logic (No Leak, Logically Weak)

**Severity:** Low (no data leak, silent performance degradation)

**Finding:**  
`_identify_swings()` declares `left = right = swing_length = 5` and computes `window_size = 11`, but
uses only a trailing 11-bar `rolling_max`/`rolling_min`. The `right` variable is dead code — it has
zero effect on the Polars computation. The detector labels a bar as a swing high if it is the max of
the **trailing** 11 bars, not the max of ±5 bars. No future bar is read. Not a lookahead bias, but
the swing labels are weak (fire on any 11-bar local max, ~9% of bars).

**Fix:** None in Phase 0. Swing logic redesign deferred to Phase 2 feature extension work.

---

## Bug 2 — `core/agent.py` L7 / L40: No Spread/Fee Cost in Training Environment

**Severity:** Critical (root cause of live dry-run capital bleed)

**Finding:**  
`KellyConvexEnv.step()` computed portfolio return with no transaction cost:
```python
portfolio_return = fraction_to_risk * bias * real_asset_return * 0.05
# No fee deduction. Agent trained in a frictionless world, deployed into reality.
```
The `0.05` scalar is an arbitrary scale factor, not a fee model. The agent learned to trade
aggressively because every trade was free. At 15m with Alpaca fees ~0.15–0.25% per side, the
expected value of a low-conviction trade is negative — but the training env never showed the agent that.

**Fix applied:** Added `spread_pct=0.0020` (0.20% one-way, configurable) as a named parameter.
On every bar where `|fraction_to_risk| > 1e-6`, the cost is deducted from `portfolio_return` before
the log-reward is computed:
```python
if abs(fraction_to_risk) > 1e-6:
    spread_cost = abs(fraction_to_risk) * self.spread_pct
    portfolio_return -= spread_cost
```
`train.py` now accepts `--spread-pct` (default `0.0020`). `auto_optimizer.py` default
`train_command` includes `--spread-pct 0.0020`. Pass `--spread-pct 0.0` to reproduce old
frictionless baseline.

**Files changed:**
- `core/agent.py` L7, L18–19, L40–55
- `train.py` L78–80, L160–167, L180–189
- `auto_optimizer.py` L83–85

---

## Bug 3 — `train.py` L144–193: Walk-Forward Windows

**Severity:** None (investigation finding — no bug)

**Finding:**  
75/25 chronological split with `VecNormalize.obs_rms` computed on train, applied read-only to test
(`training=False`). Multi-asset files concatenated sequentially (not time-interleaved). OOS fitness
metric is genuinely out-of-sample. No leakage.

**Limitation noted:** Single 75/25 split, not rolling multi-fold walk-forward. Architectural
limitation, not a bug. The auto_optimizer uses this single split for every LLM iteration —
acceptable for the current research loop.

---

## Bug 4 — `core/alpaca_bridge.py` L27–44 vs `train.py`: Live/Training Feature Path

**Severity:** Low (precision rounding only, no functional divergence)

**Finding:**  
Live path: Alpaca SDK → Pandas MultiIndex → `pl.from_pandas()` → Float32.  
Training path: Parquet → `pl.read_parquet()` → Float64 → `np.float32` cast.  
Both paths produce identical Float32 observations at the numpy boundary. The Pandas intermediary
is undesirable for future-proofing but does not affect trading decisions today.

**Fix:** None in Phase 0. Tracked for Phase 1 (if switching to a ccxt data path for 4H candles).

---

## Bug 5 — `core/features.py` L35–36: `norm_atr` Inconsistent Null Handling

**Severity:** Medium (silent null propagation; caught by safety nets but violates invariant)

**Finding:**  
Every other feature uses `.fill_nan(0.0).fill_null(0.0)`. `norm_atr` used `.fill_nan(None)`,
converting NaN to Polars null. Polars nulls propagate through arithmetic. In training, `drop_nulls()`
at L15 removed affected rows silently. In live inference, `np.nan_to_num` in `terminal.py` was the
final safety net. The system's stated invariant was violated.

**Fix applied:** Both `norm_atr` lines now use `.fill_nan(0.0).fill_null(0.0)`:
```python
# Before
norm_atr = (atr / pl.col("close").clip(lower_bound=eps)).fill_nan(None)
norm_atr = (norm_atr / (1.0 + atr_std / (atr + eps))).fill_nan(None)

# After
norm_atr = (atr / pl.col("close").clip(lower_bound=eps)).fill_nan(0.0).fill_null(0.0)
norm_atr = (norm_atr / (1.0 + atr_std / (atr + eps))).fill_nan(0.0).fill_null(0.0)
```

**File changed:** `core/features.py` L35–36

---

## Bug 6 — `terminal.py` L133–139: Long-Only Clamp Timing

**Severity:** None (passes audit)

**Finding:**  
Order: feature scaling → VecNormalize → `model.predict()` → extract action → apply clamp.
The `raw_bias <= 0 → safe_kelly = 0.0` and `kelly_raw > 0.20` threshold are applied post-inference,
after all feature scaling is complete. Correct.

---

## Bug 7 — `terminal.py` (Alpaca Path): Missing Circuit Breaker

**Severity:** Critical (potential full-equity market order on model error)

**Finding:**  
`terminal.py` + `core/alpaca_bridge.py` had no:
- Max-notional-per-order cap
- Daily drawdown kill switch
- Consecutive-loss counter

The only sizing constraint was `min(equity × kelly, buying_power)`. A PPO output of
`action[1] ≈ 1.0` with `raw_bias > 0` and `kelly_raw > 0.20` would have submitted a
near-full-equity market order.

(Note: `live_daemon.py` — the ccxt path — had full `check_drawdown()` and
`check_circuit_breaker()` guards already. The Alpaca path was the gap.)

**Fix applied:** Two independent guards added to `terminal.py` L150–180, **before**
`execute_kelly_trade()` is called. They reduce `safe_kelly` only — no changes to the
order submission function itself:

1. **Session HWM tracking:** `_session_equity_hwm` initialised on first successful equity fetch.
2. **Daily drawdown kill switch:** If `equity < HWM × (1 - MAX_DAILY_DD)`, `safe_kelly = 0.0`
   and `_trading_halted` latches True. Trading stays halted until terminal is restarted.
   Threshold: `MAX_DAILY_DD = 0.05` (5%).
3. **Hard max-notional cap:** If `safe_kelly × equity > MAX_ORDER_NOTIONAL`, `safe_kelly` is
   back-calculated as `MAX_ORDER_NOTIONAL / equity`. The order execution receives a valid [0,1]
   fraction; the cap is applied transparently.
   Threshold: `MAX_ORDER_NOTIONAL = $5,000`.

**File changed:** `terminal.py` L82–90 (constants), L150–180 (guards)

---

## Summary Table

| # | File | Lines | Severity | Status |
|---|------|--------|----------|--------|
| 1 | `core/features.py` | L62–73 | Low | No fix needed (Phase 2 redesign) |
| 2 | `core/agent.py` | L7, L40 | **Critical** | ✅ Fixed — spread_pct=0.0020 |
| 3 | `train.py` | L144–193 | None | Investigation only |
| 4 | `core/alpaca_bridge.py` | L27–44 | Low | Deferred to Phase 1 |
| 5 | `core/features.py` | L35–36 | Medium | ✅ Fixed — fill_nan(0.0) |
| 6 | `terminal.py` | L133–139 | None | Passes |
| 7 | `terminal.py` | (live loop) | **Critical** | ✅ Fixed — MAX_DD=5%, MAX_NOTIONAL=$5k |
