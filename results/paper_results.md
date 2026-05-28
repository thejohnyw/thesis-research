# Paper Results
_Generated from OOS walk-forward CV — 4 folds, seed=42, no retraining_
**Strategy**: Kalshi price in [0.4,0.6], edge ±0.05, flat $50/bet, 7% fee
---
## 1. Eligible Universe
| | Count |
|---|---|
| OOS games (folds 2–4) | 760 |
| Eligible coin-flip games (Kalshi ∈ [0.4,0.6]) | **230** |
| Fraction eligible | 30.3% |

## 2. Buy/Sell Breakdown by Model
| Model | Side | N | Win Rate (%) | PnL ($) |
|---|---|---|---|---|
| M1 | BUY | 77 | 39.0% | -409.05 |
| M1 | SELL | 90 | 48.9% | -73.47 |
| M2 | BUY | 81 | 46.9% | -129.62 |
| M2 | SELL | 92 | 52.2% | +81.57 |
| M3 | BUY | 68 | 47.1% | -104.20 |
| M3 | SELL | 87 | 56.3% | +257.86 |

## 3. Wilson 95% CI — M3 Overall Win Rate
| Metric | Value |
|---|---|
| M3 trades | 155 |
| Wins | 81 |
| Observed win rate | 52.3% |
| Wilson 95% CI lower | 44.4% |
| Wilson 95% CI upper | 60.0% |

## 4. PnL Concentration — M3 Top Trades
| Metric | Value |
|---|---|
| Total M3 PnL | $+153.66 |
| Top-3 trades sum | $+83.70 |
| Top-3 as % of total | 54.5% |

| Rank | Date | Match | Side | Mkt Price | Model Prob | PnL |
|---|---|---|---|---|---|---|
| 1 | 2025-11-26 | Houston Rockets @ Golden State Warriors | SELL | 0.600 | 0.424 | $+27.90 |
| 2 | 2026-01-05 | Golden State Warriors @ LA Clippers | BUY | 0.400 | 0.571 | $+27.90 |
| 3 | 2026-01-09 | Milwaukee Bucks @ Los Angeles Lakers | SELL | 0.600 | 0.464 | $+27.90 |

## 5. Leakage Verification
| Check | Status |
|---|---|
| Reddit cutoff | ✓ `midnight UTC(game_date)` — `_game_utc()` in `src/create_training_data.py` truncates to date-only |
| Kalshi pregame price cutoff | ✓ `expected_expiration_time − 3h` (≈30–60 min before tip-off) |
| Walk-forward CV (no future data in training) | ✓ `TimeSeriesSplit`, chronological order |

**Note on Reddit window:** `cutoff_utc` in `games_api.csv` stores only the calendar date (e.g. `2026-02-24`). `_game_utc()` always converts this to midnight UTC, making the effective window `[game_date 00:00 UTC − 48h, game_date 00:00 UTC)`. For US games tipping off at 7–10 PM ET, this cutoff falls 5–19 hours *before* tip-off. Game-day posts (injury reports, lineup news) are excluded — conservative and leakage-free.

### Spot-check: 3 sample M3 trades
| Game | Cutoff (UTC) | Posts before cutoff | Posts after cutoff | Note |
|---|---|---|---|---|
| 2026-01-11 SAS @ MIN | 2026-01-11 00:00 UTC | 12 | 0 | ✓ Clean |
| 2026-02-24 OKC @ TOR | 2026-02-24 00:00 UTC | 26 | 1 | Post is 00:00–01:00 UTC Feb 24 (≈7 PM ET Feb 23, well before tip-off) — correctly excluded by pipeline |
| 2025-12-05 DEN @ ATL | 2025-12-05 00:00 UTC | 4 | 0 | ✓ Clean |

