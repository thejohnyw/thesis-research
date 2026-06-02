# NBA Prediction & Kalshi Trading Bot

Thesis project: does Reddit fan sentiment add predictive signal for NBA game outcomes, and can that signal generate trading edge on Kalshi prediction markets?

**Answer:** Yes on both counts — marginally. The M3 model (structured stats + full sentiment distribution) achieves AUC 0.696 vs 0.689 for stats alone. The SELL side of the model-edge strategy hits 56.3% win rate in coin-flip games (+$258 over 87 trades in OOS backtest). Full results in [RESULTS.md](RESULTS.md).


Claude 
---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install fastapi uvicorn apscheduler python-dotenv statsmodels transformers
```

Create `.env` with Kalshi credentials:
```
KALSHI_KEY_ID=your_key_id
KALSHI_KEY_PATH=kalshi_key.pem
```

---

## Quick start — run the analysis

```bash
# Reproduce the 3-model comparison (M1/M2/M3, OOS metrics + trading sim)
python scripts/compare_models.py

# Generate thesis paper tables and trade log
python scripts/extract_paper_results.py
# → results/paper_results.md
# → results/trade_log.csv

# Run the strategy backtest (AntiBotClean + 3 other strategies)
python scripts/backtest_strategies.py
```

---

## Quick start — live trading bot

```bash
# 1. Train and serialize the M3 model (one-time, re-run when data updates)
python scripts/train_m3_model.py
# → models/m3_rf.pkl, models/m3_scaler.pkl, models/m3_feature_cols.json

# 2. Start the FastAPI server (paper trading mode by default)
uvicorn backend.api.main:app --reload

# 3. Start the scheduler (Kalshi price collection + scan + settlement)
curl -X POST http://localhost:8000/api/bot/start
```

The bot runs in **paper trading mode** by default (`PAPER_TRADING = True` in `backend/config.py`). Set to `False` to go live.

**Dashboard:** `GET /api/dashboard` — bankroll, open positions, recent trades, current edges.

### Daily operations
The scheduler handles everything automatically once started:
| Job | Frequency | What it does |
|---|---|---|
| `collect_job` | Every 5 min (game window) | Fetch live Kalshi prices |
| `scan_job` | Every 5 min | Run M3EdgeStrategy, place trades |
| `settle_job` | Every 2 min | Resolve finished games, book P&L |
| `reddit_job` | Every 4 hours | Fetch + score new Reddit posts |
| `pregame_job` | Daily 16:00 UTC (noon ET) | Compute M3 model probabilities for today's games |
| `daily_reset_job` | Midnight UTC | Save daily stats, reset daily P&L |

---

## How the signal works (M3EdgeStrategy)

1. **Noon ET daily:** `pregame_job` computes 53 features per upcoming game:
   - 41 structured (rolling win rates, scoring, streaks, rest — from `data/processed/games_api.csv`)
   - 12 sentiment (mean/std/entropy from Reddit posts in trading.db, 48h window ending midnight UTC of game date)
   - Runs serialized RF model → `data/pregame_predictions.json`

2. **Each scan:** `M3EdgeStrategy` checks every upcoming game:
   - Skip if Kalshi price outside [0.40, 0.60] (only trade coin-flip games)
   - `edge = model_prob − kalshi_price`
   - SELL home if `edge < −0.05` (market overpricing home team)
   - BUY home if `edge > +0.05`

3. **Sizing:** Quarter-Kelly (25%), capped at 5% of bankroll / $100 hard max.

---

## File map

```
src/                          # Offline data pipeline
  fetch_games_api.py          # Pull NBA schedule/scores from nba_api
  fetch_reddit_bulk.py        # Bulk Reddit scrape (30 subreddits)
  sentiment.py                # Score posts with twitter-roberta-base-sentiment-latest
  feature_engineering.py      # Compute 41 rolling structured features
  user_features.py            # Aggregate sentiment by user → 12 features
  create_training_data.py     # Join everything → training_data_with_sentiment.csv

scripts/                      # Analysis & model scripts
  train_m3_model.py           # Train RF on all 953 games → models/
  compare_models.py           # M1 vs M2 vs M3: OOS metrics + trading sim
  extract_paper_results.py    # Thesis tables → results/paper_results.md
  backtest_strategies.py      # AntiBotClean and strategy comparisons
  fetch_all_kalshi_prices.py  # Fetch historical Kalshi prices (candlesticks)
  fetch_remaining_prices.py   # Fetch via trade history endpoint (fallback)
  backtest_real.py            # Backtest against live bot's collected odds (trading.db)
  analyze_bot_moves.py        # LLM bot move detector for Kalshi markets
  run_sim.py                  # Live paper-trading sim (RandomStrategy)

backend/                      # Live trading bot
  config.py                   # All tuneable parameters (PAPER_TRADING, Kelly, limits)
  api/main.py                 # FastAPI server + dashboard endpoints
  core/
    signals.py                # scan_and_trade() — orchestrates scans + order execution
    scheduler.py              # APScheduler job definitions
    edge.py                   # SharpVsKalshi edge finder (fallback strategy)
    risk.py                   # Kelly sizing + position limits + circuit breaker
    settlement.py             # Resolve finished games, book P&L, compute CLV
    strategy.py               # Strategy base class + SharpVsKalshi + RandomStrategy
  strategies/
    m3_edge.py                # M3EdgeStrategy (default) — RF model vs Kalshi price
    anti_bot.py               # AntiBotSentimentStrategy — sentiment+market agreement
    user_sentiment.py         # UserSentimentStrategy — raw sentiment diff
    order_book_anchor.py      # OrderBookAnchorStrategy — order book imbalance
  data/
    kalshi_client.py          # Kalshi REST API client
    reddit_collector.py       # Live Reddit fetch + score + store to DB
    pregame_features.py       # Compute M3 features for upcoming games
    markets.py                # Parse Kalshi tickers → team names
    nba_outcomes.py           # Fetch final scores for settlement

data/
  processed/
    games_api.csv             # 953 regular-season games (nba_api)
    features.csv              # 41 structured features per game
    training_data_with_sentiment.csv  # Full M3 training dataset (53 features)
    reddit_with_sentiment.jsonl       # Scored Reddit posts
  kalshi_historical_prices.csv  # Pre-game Kalshi prices for all 953 games
  kalshi_settled_games.json     # Settled game reference data
  trading.db                    # Live bot SQLite DB (games, odds, trades, reddit_posts)

models/                       # Serialized model (gitignored — run train_m3_model.py)
  m3_rf.pkl
  m3_scaler.pkl
  m3_feature_cols.json

results/                      # Generated outputs
  paper_results.md            # Thesis tables (eligible universe, Wilson CI, etc.)
  trade_log.csv               # Per-trade record for all OOS M1/M2/M3 predictions

RESULTS.md                    # Full methodology + results (authoritative reference)
```

---

## Rebuilding the dataset from scratch

```bash
# 1. Fetch NBA schedule and scores
python -m src.fetch_games_api

# 2. Bulk-scrape Reddit posts (30 subreddits, 48h pre-game windows)
python -m src.fetch_reddit_bulk

# 3. Score posts with sentiment model
python -m src.sentiment

# 4. Compute structured features
python -m src.feature_engineering

# 5. Build training dataset (join games + features + sentiment)
python -m src.create_training_data

# 6. Fetch historical Kalshi prices (needed for backtest/trading sim)
python scripts/fetch_all_kalshi_prices.py
```

---

## Configuration (`backend/config.py`)

| Parameter | Default | Description |
|---|---|---|
| `PAPER_TRADING` | `True` | Set `False` to place real Kalshi orders |
| `KELLY_FRACTION` | `0.25` | Quarter-Kelly position sizing |
| `MAX_POSITION_PCT` | `0.05` | Max 5% of bankroll per trade |
| `MAX_POSITION_DOLLARS` | `100` | Hard cap per trade |
| `DAILY_LOSS_LIMIT` | `200` | Stop trading if daily P&L ≤ −$200 |
| `MIN_EDGE` | `0.03` | Minimum edge for SharpVsKalshi fallback |
| `INITIAL_BANKROLL` | `1000.0` | Starting bankroll |

---

## Key numbers

| | Value |
|---|---|
| Training games | 953 NBA regular season (2025-10-21 → 2026-03-08) |
| Reddit posts | 28,070 scored posts across 30 subreddits |
| M3 OOS AUC | 0.696 (vs 0.689 struct-only baseline) |
| Kalshi market AUC | 0.724 (well-informed benchmark) |
| M3 SELL win rate | 56.3% (87 trades, coin-flip games only) |
| M3 total PnL | +$154 / Sharpe 0.66 (flat $50/bet, 7% fee) |
| M3 vs matched random | +$292 (random buy/sell on same 155 games avg −$138) |
| AntiBotClean win rate | 71.8% (p=0.0002), PnL ≈ −$20 (already priced in) |


### Development Notes
Portions of this codebase were developed with Claude Code.
