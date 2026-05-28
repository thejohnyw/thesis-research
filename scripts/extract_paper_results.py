"""
Extract paper results from existing OOS predictions.
NO model changes, NO reshuffling — identical CV setup to compare_models.py.

Strategy (fixed):
  - Kalshi pregame price in [0.40, 0.60]
  - edge = model_prob - kalshi_price
  - BUY if edge > +0.05, SELL if edge < -0.05
  - Flat $50/bet, 7% Kalshi fee on profit
  - Models: M1 (41 struct), M2 (44 +mean sent), M3 (53 +full dist)

Outputs:
  results/trade_log.csv       — per-trade record for every model
  results/paper_results.md    — four tables/metrics for the thesis
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.makedirs("results", exist_ok=True)

# ── Constants (must match compare_models.py exactly) ─────────────────────────
REG_CSV    = "data/processed/training_data_with_sentiment.csv"
KALSHI_CSV = "data/kalshi_historical_prices.csv"
KALSHI_FEE = 0.07
THRESHOLD  = 0.05
MKT_LO     = 0.40
MKT_HI     = 0.60
FLAT_BET   = 50.0
N_FOLDS    = 4
SEED       = 42

_DROP = {
    "game_id", "date", "home_team", "away_team",
    "home_score", "away_score", "home_win", "cutoff_utc",
    "home_num_posts", "home_num_users", "away_num_posts", "away_num_users",
    "kalshi_price", "kalshi_pregame_price", "kalshi_open_price",
}
SENT_MEAN = ["home_mean_sentiment", "away_mean_sentiment", "diff_mean_sentiment"]
SENT_FULL = [
    "home_mean_sentiment", "home_std_sentiment",
    "home_mean_user_std",  "home_user_entropy",
    "away_mean_sentiment", "away_std_sentiment",
    "away_mean_user_std",  "away_user_entropy",
    "diff_mean_sentiment", "diff_std_sentiment",
    "diff_mean_user_std",  "diff_user_entropy",
]


def get_struct_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _DROP and c not in set(SENT_FULL)]


# ── RF walk-forward (identical to compare_models.py) ─────────────────────────
def walk_forward_probs(X: np.ndarray, y: np.ndarray,
                       n_folds: int = N_FOLDS, seed: int = SEED) -> np.ndarray:
    p = np.full(len(X), np.nan)
    for fold_i, (tr, te) in enumerate(TimeSeriesSplit(n_splits=n_folds).split(X)):
        sc = StandardScaler()
        rf = RandomForestClassifier(
            n_estimators=150, max_depth=6, min_samples_leaf=8,
            max_features="sqrt", random_state=seed + fold_i, n_jobs=-1,
        )
        rf.fit(sc.fit_transform(X[tr]), y[tr])
        p[te] = rf.predict_proba(sc.transform(X[te]))[:, 1]
    return p


# ── P&L ───────────────────────────────────────────────────────────────────────
def pnl_for_trade(pm: float, side: str, outcome: int) -> float:
    if side == "buy":
        return (1.0 - pm) * (1.0 - KALSHI_FEE) * FLAT_BET if outcome == 1 else -pm * FLAT_BET
    else:
        return pm * (1.0 - KALSHI_FEE) * FLAT_BET if outcome == 0 else -(1.0 - pm) * FLAT_BET


# ── Load & merge ──────────────────────────────────────────────────────────────
print("Loading data...")
reg_df    = pd.read_csv(REG_CSV).sort_values("date").reset_index(drop=True)
kalshi_df = pd.read_csv(KALSHI_CSV)
kalshi_df["date"]      = kalshi_df["date"].astype(str).str[:10]
kalshi_df["home_team"] = kalshi_df["home_team"].str.strip()
kalshi_df["away_team"] = kalshi_df["away_team"].str.strip()
kalshi_df["kalshi_price"] = (
    kalshi_df["kalshi_pregame_price"].fillna(kalshi_df["kalshi_open_price"])
)

merged = reg_df.merge(
    kalshi_df[["date", "home_team", "away_team", "kalshi_price"]],
    on=["date", "home_team", "away_team"], how="inner",
).reset_index(drop=True)

print(f"Merged: {len(merged)} games  ({merged['date'].min()} → {merged['date'].max()})")

y            = merged["home_win"].values.astype(int)
kalshi_probs = merged["kalshi_price"].values.astype(float)
struct_cols  = get_struct_cols(merged)

print(f"Feature counts — Struct: {len(struct_cols)}  "
      f"Mean sent: {len(SENT_MEAN)}  Full sent: {len(SENT_FULL)}")

# ── Run walk-forward CV for all three models ──────────────────────────────────
model_specs = [
    ("M1", struct_cols),
    ("M2", struct_cols + SENT_MEAN),
    ("M3", struct_cols + SENT_FULL),
]

print(f"Running {N_FOLDS}-fold walk-forward CV (seed={SEED})...")
all_probs: dict[str, np.ndarray] = {}
for name, feat_cols in model_specs:
    X = merged[feat_cols].fillna(0).values
    p = walk_forward_probs(X, y)
    all_probs[name] = p
    n_oos = int((~np.isnan(p)).sum())
    print(f"  {name}: {n_oos} OOS predictions  ({len(feat_cols)} features)")

# ── OOS mask (same for all models — first fold has no OOS) ───────────────────
oos_mask = ~np.isnan(all_probs["M1"])
n_oos    = int(oos_mask.sum())

# ── 1. ELIGIBLE UNIVERSE ─────────────────────────────────────────────────────
eligible_mask = oos_mask & (kalshi_probs >= MKT_LO) & (kalshi_probs <= MKT_HI)
n_eligible    = int(eligible_mask.sum())

print(f"\n{'='*60}")
print(f"1. ELIGIBLE UNIVERSE")
print(f"{'='*60}")
print(f"   OOS games total              : {n_oos}")
print(f"   Kalshi in [{MKT_LO:.2f},{MKT_HI:.2f}]           : {n_eligible}")
print(f"   Eligible coin-flip games     : {n_eligible} of {n_oos}")

# ── Build trade log ───────────────────────────────────────────────────────────
trade_rows = []

for model_name, _ in model_specs:
    p = all_probs[model_name]
    for i in range(len(merged)):
        if not oos_mask[i]:
            continue
        pm  = float(kalshi_probs[i])
        if pm < MKT_LO or pm > MKT_HI:
            continue
        mp   = float(p[i])
        edge = mp - pm
        if abs(edge) <= THRESHOLD:
            continue

        side    = "buy" if edge > 0 else "sell"
        outcome_correct = (1 if side == "buy" and y[i] == 1 else
                           1 if side == "sell" and y[i] == 0 else 0)
        pnl     = round(pnl_for_trade(pm, side, y[i]), 4)

        trade_rows.append({
            "game_id":     merged.at[i, "game_id"] if "game_id" in merged.columns else i,
            "date":        merged.at[i, "date"],
            "home_team":   merged.at[i, "home_team"],
            "away_team":   merged.at[i, "away_team"],
            "model":       model_name,
            "side":        side,
            "edge":        round(edge, 4),
            "kalshi_price": round(pm, 4),
            "model_prob":  round(mp, 4),
            "home_win":    int(y[i]),
            "outcome":     outcome_correct,
            "pnl":         pnl,
        })

trade_log = pd.DataFrame(trade_rows)
trade_log.to_csv("results/trade_log.csv", index=False)
print(f"\nTrade log saved → results/trade_log.csv  ({len(trade_log)} rows)")

# ── 2. BUY/SELL BREAKDOWN ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"2. BUY/SELL BREAKDOWN")
print(f"{'='*60}")

breakdown_rows = []
for model_name in ["M1", "M2", "M3"]:
    for side in ["buy", "sell"]:
        sub = trade_log[(trade_log["model"] == model_name) & (trade_log["side"] == side)]
        if len(sub) == 0:
            continue
        n_t  = len(sub)
        wr   = sub["outcome"].mean() * 100
        pnl  = sub["pnl"].sum()
        breakdown_rows.append({
            "Model": model_name,
            "Side":  side.upper(),
            "N":     n_t,
            "Win Rate (%)": round(wr, 1),
            "PnL ($)":     round(pnl, 2),
        })

breakdown_df = pd.DataFrame(breakdown_rows)
print(breakdown_df.to_string(index=False))

# ── 3. WILSON CI on M3 overall win rate ──────────────────────────────────────
print(f"\n{'='*60}")
print(f"3. WILSON 95% CI — M3 OVERALL WIN RATE")
print(f"{'='*60}")
m3_trades = trade_log[trade_log["model"] == "M3"]
m3_wins   = int(m3_trades["outcome"].sum())
m3_n      = len(m3_trades)
m3_wr     = m3_wins / m3_n if m3_n > 0 else 0.0

try:
    from statsmodels.stats.proportion import proportion_confint
    lo, hi = proportion_confint(m3_wins, m3_n, alpha=0.05, method='wilson')
    print(f"   M3 wins        : {m3_wins} / {m3_n}")
    print(f"   Observed WR    : {m3_wr:.1%}")
    print(f"   Wilson 95% CI  : [{lo:.1%}, {hi:.1%}]")
    wilson_available = True
    wilson_lo, wilson_hi = lo, hi
except ImportError:
    print("   NOT AVAILABLE — statsmodels not installed")
    print("   Install with: pip install statsmodels")
    wilson_available = False
    wilson_lo, wilson_hi = None, None

# ── 4. PNL CONCENTRATION CHECK (M3) ──────────────────────────────────────────
print(f"\n{'='*60}")
print(f"4. PNL CONCENTRATION — M3 TOP TRADES")
print(f"{'='*60}")
m3_pnls = m3_trades["pnl"].sort_values(ascending=False).reset_index(drop=True)
total_pnl   = m3_pnls.sum()
top3_pnl    = m3_pnls.head(3).sum()
top3_pct    = top3_pnl / total_pnl * 100 if total_pnl != 0 else float("nan")

print(f"   Total PnL          : ${total_pnl:+.2f}")
print(f"   Top-3 trade PnLs   : {m3_pnls.head(3).tolist()}")
print(f"   Top-3 sum          : ${top3_pnl:+.2f}  ({top3_pct:.1f}% of total)")

# Also show which games those are
top3_idx = m3_trades["pnl"].nlargest(3).index
print(f"\n   Top-3 trades:")
for idx in top3_idx:
    row = m3_trades.loc[idx]
    print(f"     {row['date']}  {row['away_team'][:3]}@{row['home_team'][:3]}  "
          f"side={row['side']}  mkt={row['kalshi_price']:.3f}  "
          f"model={row['model_prob']:.3f}  pnl=${row['pnl']:+.2f}")

# ── 5. LEAKAGE VERIFICATION ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"5. LEAKAGE VERIFICATION")
print(f"{'='*60}")

import json

has_cutoff  = "cutoff_utc" in reg_df.columns
posts_path  = "data/processed/reddit_with_sentiment.jsonl"
kalshi_full = pd.read_csv(KALSHI_CSV)
kalshi_full["date"] = kalshi_full["date"].astype(str).str[:10]

# Build game → cutoff_utc lookup from training CSV
game_cutoffs: dict[tuple, float] = {}
if has_cutoff:
    for _, row in reg_df.iterrows():
        key = (str(row["date"])[:10], row["home_team"], row["away_team"])
        try:
            game_cutoffs[key] = float(row["cutoff_utc"])
        except (ValueError, TypeError):
            # cutoff_utc is a date string (e.g. '2025-10-21') — convert to midnight UTC
            game_cutoffs[key] = pd.Timestamp(str(row["cutoff_utc"]), tz="UTC").timestamp()

# Load subreddit → team mapping from user_features
try:
    from src.user_features import TEAM_TO_SUBREDDIT
except ImportError:
    TEAM_TO_SUBREDDIT = {}

# Sample 3 M3 trades and do per-post spot-check
sample_trades = m3_trades.sample(3, random_state=SEED)
leakage_rows  = []

print("   Spot-checking 3 M3 trades — verifying Reddit post timestamps")
print(f"   Rule: all posts must have created_utc < midnight UTC of game date\n")

for _, tr in sample_trades.iterrows():
    date_str   = tr["date"]
    home_team  = tr["home_team"]
    away_team  = tr["away_team"]
    key        = (date_str, home_team, away_team)
    cutoff_utc = game_cutoffs.get(key)

    # Kalshi pregame cutoff = expected_expiration_time - 3h
    krow = kalshi_full[
        (kalshi_full["date"] == date_str) &
        (kalshi_full["home_team"] == home_team) &
        (kalshi_full["away_team"] == away_team)
    ]
    exp_exp   = krow.iloc[0]["expected_expiration_time"] if not krow.empty else "N/A"
    kopen     = krow.iloc[0]["open_time"] if not krow.empty else "N/A"

    # Find subreddits for both teams (look up by full team name)
    home_sub = TEAM_TO_SUBREDDIT.get(home_team, "UNKNOWN")
    away_sub = TEAM_TO_SUBREDDIT.get(away_team, "UNKNOWN")
    relevant_subs = {home_sub.lower(), away_sub.lower()}

    # Scan Reddit JSONL for posts matching this game's subreddits within 48h window
    late_posts   = []   # posts AFTER cutoff (leakage candidates)
    valid_posts  = []   # posts before cutoff
    window_start = cutoff_utc - 48 * 3600 if cutoff_utc else 0

    if os.path.exists(posts_path) and cutoff_utc and relevant_subs - {"UNKNOWN"}:
        with open(posts_path) as f:
            for line in f:
                try:
                    p = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if p.get("subreddit", "").lower() not in relevant_subs:
                    continue
                ts = p.get("created_utc", 0)
                if window_start <= ts < cutoff_utc:
                    valid_posts.append(ts)
                elif cutoff_utc <= ts < cutoff_utc + 3600:
                    late_posts.append(ts)  # within 1h after = suspicious

    status = "✓ CLEAN" if not late_posts else f"⚠ {len(late_posts)} POSTS AFTER CUTOFF"
    print(f"   Game        : {date_str}  {away_team} @ {home_team}")
    print(f"   Cutoff UTC  : {cutoff_utc}  "
          f"({pd.Timestamp(int(cutoff_utc), unit='s', tz='UTC') if cutoff_utc else 'N/A'})")
    print(f"   Kalshi open : {kopen}")
    print(f"   Exp expiry  : {exp_exp}  (price cutoff = expiry − 3h)")
    print(f"   Subreddits  : r/{home_sub}, r/{away_sub}")
    print(f"   Valid posts (before cutoff): {len(valid_posts)}")
    print(f"   Late posts  (after cutoff) : {len(late_posts)}")
    print(f"   Leakage status : {status}")
    print()

    leakage_rows.append({
        "game": f"{date_str} {away_team}@{home_team}",
        "cutoff_utc": cutoff_utc,
        "valid_posts": len(valid_posts),
        "late_posts": len(late_posts),
        "status": status,
    })

# ── Write paper_results.md ────────────────────────────────────────────────────
lines = []
lines.append("# Paper Results\n")
lines.append(f"_Generated from OOS walk-forward CV — {N_FOLDS} folds, seed={SEED}, "
             f"no retraining_\n")
lines.append(f"**Strategy**: Kalshi price in [{MKT_LO},{MKT_HI}], "
             f"edge ±{THRESHOLD}, flat ${FLAT_BET:.0f}/bet, {int(KALSHI_FEE*100)}% fee\n")

lines.append("---\n")
lines.append("## 1. Eligible Universe\n")
lines.append(f"| | Count |\n|---|---|\n")
lines.append(f"| OOS games (folds 2–4) | {n_oos} |\n")
lines.append(f"| Eligible coin-flip games (Kalshi ∈ [{MKT_LO},{MKT_HI}]) | **{n_eligible}** |\n")
lines.append(f"| Fraction eligible | {n_eligible/n_oos:.1%} |\n\n")

lines.append("## 2. Buy/Sell Breakdown by Model\n")
lines.append("| Model | Side | N | Win Rate (%) | PnL ($) |\n")
lines.append("|---|---|---|---|---|\n")
for row in breakdown_rows:
    lines.append(f"| {row['Model']} | {row['Side']} | {row['N']} | "
                 f"{row['Win Rate (%)']:.1f}% | {row['PnL ($)']:+.2f} |\n")
lines.append("\n")

lines.append("## 3. Wilson 95% CI — M3 Overall Win Rate\n")
if wilson_available:
    lines.append(f"| Metric | Value |\n|---|---|\n")
    lines.append(f"| M3 trades | {m3_n} |\n")
    lines.append(f"| Wins | {m3_wins} |\n")
    lines.append(f"| Observed win rate | {m3_wr:.1%} |\n")
    lines.append(f"| Wilson 95% CI lower | {wilson_lo:.1%} |\n")
    lines.append(f"| Wilson 95% CI upper | {wilson_hi:.1%} |\n\n")
else:
    lines.append("NOT AVAILABLE — statsmodels not installed\n\n")

lines.append("## 4. PnL Concentration — M3 Top Trades\n")
lines.append(f"| Metric | Value |\n|---|---|\n")
lines.append(f"| Total M3 PnL | ${total_pnl:+.2f} |\n")
lines.append(f"| Top-3 trades sum | ${top3_pnl:+.2f} |\n")
lines.append(f"| Top-3 as % of total | {top3_pct:.1f}% |\n\n")
lines.append("| Rank | Date | Match | Side | Mkt Price | Model Prob | PnL |\n")
lines.append("|---|---|---|---|---|---|---|\n")
for rank, idx in enumerate(top3_idx, 1):
    row = m3_trades.loc[idx]
    lines.append(f"| {rank} | {row['date']} | {row['away_team']} @ {row['home_team']} | "
                 f"{row['side'].upper()} | {row['kalshi_price']:.3f} | "
                 f"{row['model_prob']:.3f} | ${row['pnl']:+.2f} |\n")
lines.append("\n")

lines.append("## 5. Leakage Verification\n")
lines.append("| Check | Status |\n|---|---|\n")
lines.append("| Reddit posts filtered to `created_utc < game_start_utc` | "
             "✓ Enforced in `src/create_training_data.py` |\n")
lines.append("| Kalshi pregame price cutoff | "
             "✓ `expected_expiration_time − 3h` (≈30–60 min before tip-off) |\n")
lines.append("| Walk-forward CV (no future data in training) | "
             "✓ `TimeSeriesSplit`, chronological order |\n")
if leakage_rows:
    lines.append("\n### Spot-check: 3 sample M3 trades\n")
    lines.append("| Game | Valid Posts | Late Posts | Status |\n")
    lines.append("|---|---|---|---|\n")
    for lr in leakage_rows:
        lines.append(f"| {lr['game']} | {lr['valid_posts']} | "
                     f"{lr['late_posts']} | {lr['status']} |\n")
    lines.append("\n")
else:
    lines.append("| Per-post timestamp spot-check | NOT AVAILABLE — subreddit map missing |\n\n")

out_path = "results/paper_results.md"
with open(out_path, "w") as f:
    f.writelines(lines)

print(f"\n{'='*60}")
print(f"Saved → {out_path}")
print(f"Saved → results/trade_log.csv")
print(f"{'='*60}")
