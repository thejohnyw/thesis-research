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

### Three-model comparison
All models use a Random Forest classifier (150 trees, max_depth=6, min_samples_leaf=8, max_features=sqrt) with 4-fold walk-forward CV.

| Model | Label | Feature set |
|---|---|---|
| M1 | Baseline | 41 structured only |
| M2 | + Mean sentiment | 41 struct + 3 mean-sentiment cols |
| M3 | + Full distribution | 41 struct + 12 full sentiment distribution |

Kalshi pregame price (`expected_expiration_time − 3h` candlestick) is **not** included as a model feature to avoid contaminating the edge signal.

---

## 4. Train / Validation / Test Split

- **Method:** 4-fold temporal expanding-window cross-validation (`TimeSeriesSplit`)
- **Games in OOS evaluation:** 760 / 953 (first fold used only for training)
- **No shuffling** — folds respect chronological order to prevent leakage
- **Reddit window:** `[midnight_utc(game_date) − 48h, midnight_utc(game_date))` — the cutoff is midnight UTC on the game's calendar date (e.g. game on 2026-02-24 → cutoff = 2026-02-24 00:00 UTC ≈ 7 PM ET the night before). `cutoff_utc` in `games_api.csv` stores only the date string; `_game_utc()` in `src/create_training_data.py` always truncates to midnight UTC. **Implication:** game-day posts (injury reports, lineup tweets) are excluded — a conservative choice that guarantees no leakage but omits the freshest pre-game sentiment.

---

## 5. Prediction Model Results

All metrics on the 760 out-of-sample games (4-fold walk-forward, RF classifier).

| Model | Features | N (OOS) | Accuracy | ROC AUC | Brier Score |
|---|---|---|---|---|---|
| Dummy (majority class) | — | 760 | 0.530 | 0.500 | 0.249 |
| M1 — Baseline (struct only) | 41 | 760 | 0.636 | 0.689 | 0.224 |
| M2 — Standard (struct + mean sent) | 44 | 760 | **0.651** | 0.692 | 0.223 |
| M3 — Thesis (struct + full sent dist) | 53 | 760 | 0.643 | **0.696** | **0.221** |
| Kalshi market (hard threshold ≥ 0.5) | — | 760 | 0.654 | 0.724 | — |

Takeaway: AUC improves monotonically M1 → M2 → M3 (+0.007 total). Mean sentiment (M2) explains most of the gain; the full distribution's std and entropy features add a further +0.004 AUC. The Kalshi market (AUC 0.724) outperforms all three models, confirming the market is well-informed — the models' edge comes from identifying games where market pricing diverges from the sentiment signal.

---

## 6. Trading Simulation Results

### 6a. Model-Edge Strategy (primary thesis result)

**Signal:** Kalshi pregame price ∈ [0.40, 0.60] AND |model_prob − kalshi_price| > 0.05  
- BUY home: `model_prob − kalshi_price > +0.05`  
- SELL home: `model_prob − kalshi_price < −0.05`  
- Flat $50/bet, 7% Kalshi fee on profit, 4-fold walk-forward OOS (760 games)

| Model | N bets | Win Rate | PnL | Sharpe | p-value |
|---|---|---|---|---|---|
| M1 — Baseline | 167 | 44.3% | −$486 | −1.94 | 0.94 |
| M2 — Standard | 173 | 49.7% | −$48 | −0.18 | 0.56 |
| **M3 — Thesis** | **155** | **52.3%** | **+$154** | **0.66** | **0.315** |
| Random baseline (500-trial avg) | 230 | 50.2% | −$173 | −0.53 | 0.55 |

**Buy/Sell breakdown (M3):**

| Side | N | Win Rate | PnL |
|---|---|---|---|
| BUY home | 68 | 47.1% | −$104 |
| **SELL home** | **87** | **56.3%** | **+$258** |

**Interpretation:**  
M3's positive PnL (+$154) is driven entirely by the SELL side (56.3% WR, +$258). When M3 predicts a lower home-win probability than the Kalshi market, it is correct 56.3% of the time in coin-flip games. The BUY side (47.1% WR) does not generate a reliable edge, suggesting the sentiment signal's information content is asymmetric — it is better at detecting overpriced home teams than underpriced ones. The 0.315 p-value means results are directionally consistent but not statistically significant at conventional thresholds; the 155-trade sample is the binding constraint.

Wilson 95% CI on M3 overall win rate: [44.4%, 60.0%]

---

### 6b. AntiBotClean Strategy (directional-signal analysis)

**Signal:** Reddit raw sentiment diff > ±0.10 AND Kalshi market direction agrees  
- BUY home: `diff > 0.10` AND `p_kalshi ≥ 0.55`  
- SELL home: `diff < −0.10` AND `p_kalshi < 0.45`

| Metric | Value |
|---|---|
| Games in backtest | 953 regular season + 22 playoff |
| Min posts filter | ≥ 5 per team |
| Bets placed | 71 |
| Hit rate (win rate) | **71.8%** |
| Total PnL (flat $50/bet) | **−$14.61** |
| ROI per dollar risked | −0.4% |
| Sharpe ratio (annualized) | −0.15 |
| Max drawdown | −$114.86 |
| Binomial p-value (H0: WR ≤ 50%) | **0.0002** |

**Strategy comparison (flat $50/bet, min 5 posts, real Kalshi prices):**

| Strategy | N | Win Rate | PnL | p-value |
|---|---|---|---|---|
| DirectSentiment (raw) | 150 | 46.7% | −$432 | 0.82 |
| AntiBotSentiment (3-regime) | 108 | 53.7% | −$152 | 0.25 |
| **AntiBotClean (2-regime)** | **71** | **71.8%** | **−$15** | **0.0002** |
| RandomBaseline | 213 | 53.1% | +$167 | 0.21 |

**Interpretation:**  
AntiBotClean achieves a statistically significant 71.8% win rate (p=0.0002), confirming the combined sentiment+market-agreement signal has genuine directional predictive power. However, net PnL is near zero (−$14.61). When the strategy fires, the Kalshi market already prices the home team as a favorite (p ≥ 0.55), so winning pays out only $0.20–$0.40 per dollar wagered. After the 7% fee, small payoffs are insufficient to profit despite the high win rate. The sentiment signal is predictive but already priced into the market.

**Kalshi price source (both strategies):**
- All 953 regular-season games: real pre-game candlestick prices via Kalshi historical API (`expected_expiration_time − 3h` cutoff)
- 22 playoff games (Apr 28 – May 28, 2026): live bot prices collected before each game
- 100% of trades used real Kalshi prices — no RF proxy

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
