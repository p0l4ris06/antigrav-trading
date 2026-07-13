# VALIDATION.md — Phase 3 Walk-Forward Evaluation

Branch: pivot/4h-microstructure
Date: 2026-07-13
Model: PPO · KellyConvexEnv · 75/25 OOS split
Features: BASE (9 dims)

## Results

| Run | Dataset   | spread_pct | Steps | FITNESS_SCORE |
|-----|-----------|------------|-------|---------------|
| 1   | 15m 210k rows | 0.0    | 50k   | 0.9956        |
| 2   | 15m 210k rows | 0.0020 | 50k   | 0.9993        |
| 3   | 4H  13k rows  | 0.0020 | 50k   | 1.0000        |

Run 4 (4H, 250k steps): PENDING
Run 5 (4H extended, 250k steps): PENDING

## Key findings
- Per-trade fee model validated: adding spread on 15m improves selectivity (0.9956 -> 0.9993)
- 4H timeframe reaches cash parity at 50k steps — hypothesis survives first contact
- 50k steps insufficient to detect positive alpha on 4H; Run 4 at 250k is the decisive test
