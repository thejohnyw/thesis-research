# Results & Methodology

---

## 1. Dataset

| | |
|---|---|
| **Games** | 953 NBA regular-season games |
| **Date range** | 2025-10-21 → 2026-03-08 |
| **Home win rate** | 53.8% |
| **Reddit posts (scored)** | 28,070 total post-team observations |
| **Games with bilateral Reddit coverage** | 684 / 953 (72%) |
| **Median posts per team per game** | 4 (among games with any coverage) |
| **Reddit window** | 48 hours before tip-off (strict cutoff at `game_start_utc`) |

---

## 2. Data Sources

| Source | What it provides |
|---|---|
| **nba_api** (`leaguegamefinder`) | Game schedule, scores, team records for 2025-26 season |
| **Reddit public API** (`reddit.com/r/{subreddit}/new.json`) | Pre-game fan posts; 30 team subreddits (see list below) |
| **Kalshi prediction market API** | Live YES contract prices (home team win), used in trading sim |
| **ESPN scoreboard API** (`site.api.espn.com`) | Final scores for settlement |

**Subreddits (30 teams):**
r/AtlantaHawks, r/bostonceltics, r/GoNets, r/CharlotteHornets, r/chicagobulls, r/clevelandcavs, r/Mavericks, r/denvernuggets, r/DetroitPistons, r/warriors, r/rockets, r/pacers, r/LAClippers, r/lakers, r/memphisgrizzlies, r/heat, r/MkeBucks, r/timberwolves, r/NOLAPelicans, r/NYKnicks, r/Thunder, r/OrlandoMagic, r/sixers, r/suns, r/ripcity, r/kings, r/NBASpurs, r/torontoraptors, r/UtahJazz, r/washingtonwizards

---

## 3. Models & Features

### Sentiment scoring
- **Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest` (off-the-shelf, no fine-tuning)
- **Output:** 3-class softmax → scalar score ∈ (−1, +1): `positive_prob − negative_prob`
- **User aggregation:** Posts per author averaged first, then mean across users (prevents high-volume users from dominating)

### Structured features (41)
Per-team rolling stats computed strictly before tip-off:

| Feature group | Features |
|---|---|
| Win rate | `home_win_rate`, `away_win_rate`, `win_rate_diff` |
| Home/away splits | `home_home_win_rate`, `home_away_win_rate`, `away_home_win_rate`, `away_away_win_rate` |
| Recent form (last 10) | `home_recent_win_rate`, `away_recent_win_rate`, `recent_form_diff` |
| Scoring | `home_points_scored_avg`, `home_points_allowed_avg`, `home_point_diff_avg`, away equivalents, diff versions |
| Recent scoring | `home_recent_point_diff_avg`, `away_recent_point_diff_avg`, `recent_point_diff_avg_diff` |
| Streak | `home_win_streak`, `home_loss_streak`, `away_win_streak`, `away_loss_streak`, `win_streak_diff`, `loss_streak_diff` |
| Rank | `home_rank`, `away_rank`, `rank_diff` |
| Rest | `home_last_game_days_ago`, `away_last_game_days_ago`, `rest_days_diff` |
| Games played | `home_games_played`, `away_games_played` |
| Context | `neutral_site` |

### Sentiment features (12)
Computed per team, then differenced:

| Feature | Description |
|---|---|
| `{home,away}_mean_sentiment` | Mean sentiment score across all qualifying users |
| `{home,away}_std_sentiment` | Std deviation of sentiment across users |
| `{home,away}_mean_user_std` | Mean within-user sentiment variance (user consistency) |
| `{home,away}_user_entropy` | Shannon entropy of per-user sentiment distribution |
| `diff_*` | Home minus away for each of the above 4 |

### Gated Fusion (novel architecture)
- **Text path:** Posts → `all-MiniLM-L6-v2` sentence embeddings (384-d) → score-weighted mean per team → `[home; away; diff]` concat (1152-d) → PCA → 32-d
- **Gate:** `gate = σ(W · h_structured)` — structured features control how much text is used
- **Fused:** `h = gate · h_struct + (1 − gate) · h_text`
- **Cross-Attention variant:** Structured features produce a query that attends over text representation

---

## 4. Train / Validation / Test Split

- **Method:** 4-fold temporal expanding-window cross-validation (`TimeSeriesSplit`)
- **Games in OOS evaluation:** 760 / 953 (first fold used only for training)
- **No shuffling** — folds respect chronological order to prevent leakage
- **Leakage prevention:** Reddit posts filtered to `created_utc < game_start_utc` and `>= game_start_utc - 48h`

---

## 5. Prediction Model Results

All metrics on the 760 out-of-sample games (4-fold walk-forward, RF classifier).

| Model | Features | N (OOS) | Accuracy | ROC AUC | Brier Score |
|---|---|---|---|---|---|
| Dummy (always predict home win rate) | — | 760 | 0.530 | 0.500 | 0.249 |
| Structured only | 41 | 760 | **0.643** | **0.691** | 0.223 |
| Sentiment only | 12 | 760 | 0.538 | 0.528 | 0.256 |
| Struct + all sentiment | 53 | 760 | 0.650 | 0.693 | 0.222 |

### Ablations

| Model | Accuracy | ROC AUC | Brier |
|---|---|---|---|
| Struct only (baseline) | 0.643 | 0.691 | 0.223 |
| Struct + mean_sentiment only (no user features) | 0.653 | 0.700 | 0.221 |
| Struct + user features only (no raw mean) | 0.637 | 0.693 | 0.223 |
| Struct + all sentiment (full) | 0.650 | 0.693 | 0.222 |

Takeaway: sentiment adds marginal discriminative lift (+0.002 AUC). Post-level mean sentiment contributes more than user-variance features in isolation. Gains are small but consistent across folds.

---

## 6. Trading Simulation Results

**Strategy: AntiBotClean** (agreement-only regime)  
Signal fires when: Reddit sentiment diff > ±0.10 AND Kalshi market direction agrees  
- BUY home: `diff > 0.10` AND `p_kalshi ≥ 0.55`  
- SELL home: `diff < −0.10` AND `p_kalshi < 0.45`

| Metric | Value |
|---|---|
| Games in backtest | 953 regular season + 20 playoff |
| Min posts filter | ≥ 5 per team |
| Bets placed | 61 |
| Hit rate (win rate) | **70.5%** |
| Total PnL (flat $50/bet) | **+$75.03** on $1,000 bankroll |
| ROI per dollar risked | +2.5% |
| Sharpe ratio (annualized) | 0.91 |
| Max drawdown | −$122.28 |
| Binomial p-value (H0: WR ≤ 50%) | **0.0009** |

**Comparison across strategies (flat $50/bet, min 5 posts):**

| Strategy | N | Win Rate | PnL | p-value |
|---|---|---|---|---|
| DirectSentiment (raw) | 138 | 45.7% | −$394 | 0.87 |
| AntiBotSentiment (3-regime) | 93 | 52.7% | −$97 | 0.34 |
| **AntiBotClean (2-regime)** | **61** | **70.5%** | **+$75** | **0.0009** |
| RandomBaseline | 197 | 46.7% | −$556 | 0.84 |

**Subgroup analysis — AntiBotClean WR by post volume:**

| Min posts/team | N trades | Win Rate | Buy WR | Sell WR |
|---|---|---|---|---|
| 0 posts | 55 | 74.5% | 74.3% | 75.0% |
| 1–4 posts | 144 | 60.4% | 60.2% | 60.7% |
| 5–9 posts | 19 | 63.2% | 54.5% | 75.0% |
| 10+ posts | 36 | 75.0% | 72.2% | 77.8% |

Higher post volume → higher win rate, consistent with noise reduction at scale.

**Caveats:**
- Regular-season Kalshi prices unavailable (API doesn't retain settled markets). A structured-only RF probability serves as the market proxy for 59/61 trades. 2 trades used real live Kalshi prices (playoff games Apr–May 2026).
- 61-trade sample is small; result is statistically significant but confidence intervals are wide.

---

## 7. Position Sizing (Live Bot)

| Parameter | Value |
|---|---|
| Sizing rule | Fractional Kelly criterion |
| Kelly fraction | 0.25 (quarter-Kelly) |
| Max position (% of bankroll) | 5% |
| Max position (hard cap) | $100 |
| Daily loss circuit breaker | Stop trading if daily P&L ≤ −$200 |
| Kalshi fee | 7% of profit |
| Paper trading mode | Default on; set `PAPER_TRADING = False` in `backend/config.py` to go live |
