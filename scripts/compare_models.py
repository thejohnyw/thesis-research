"""
Three-model comparison using ONLY real pre-game Kalshi prices (no RF proxy).

  Model 1 — Baseline:  41 structured features only
  Model 2 — Standard:  Structured + mean sentiment (3 features, direction signal)
  Model 3 — Thesis:    Structured + full sentiment distribution (12 features)
  Random             :  Coin-flip control on same eligible games

Data:  953 regular-season games (Oct 21 2025 – Mar 8 2026)
       real Kalshi candlestick prices (expected_expiration_time − 3h cutoff)
CV:    4-fold walk-forward TimeSeriesSplit
Trade: bet when |model_prob − kalshi_pregame_price| > threshold
       and kalshi_pregame_price in [mkt_lo, mkt_hi]

Metrics: Brier score, ROC AUC, Accuracy (prediction quality)
         Sharpe ratio, Win Rate, PnL, N trades (trading alpha)

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --threshold 0.05 --mkt-lo 0.40 --mkt-hi 0.60
    python scripts/compare_models.py --flat-bet 50 --trades
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REG_CSV    = "data/processed/training_data_with_sentiment.csv"
KALSHI_CSV = "data/kalshi_historical_prices.csv"

KALSHI_FEE = 0.07

SEP  = "─" * 70
SEP2 = "═" * 70

# ── Feature column sets ───────────────────────────────────────────────────────

_DROP = {
    "game_id", "date", "home_team", "away_team",
    "home_score", "away_score", "home_win", "cutoff_utc",
    "home_num_posts", "home_num_users", "away_num_posts", "away_num_users",
    # Kalshi price is only for P&L calculation — must NOT be a model feature
    "kalshi_price", "kalshi_pregame_price", "kalshi_open_price",
}

SENT_MEAN = [
    "home_mean_sentiment", "away_mean_sentiment", "diff_mean_sentiment",
]

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


# ── RF helpers ────────────────────────────────────────────────────────────────

def _make_rf(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=150, max_depth=6, min_samples_leaf=8,
        max_features="sqrt", random_state=seed, n_jobs=-1,
    )


def walk_forward_probs(X: np.ndarray, y: np.ndarray,
                       n_folds: int = 4, seed: int = 42) -> np.ndarray:
    p = np.full(len(X), np.nan)
    for fold_i, (tr, te) in enumerate(TimeSeriesSplit(n_splits=n_folds).split(X)):
        sc = StandardScaler()
        rf = _make_rf(seed + fold_i)
        rf.fit(sc.fit_transform(X[tr]), y[tr])
        p[te] = rf.predict_proba(sc.transform(X[te]))[:, 1]
    return p


# ── P&L calculation ───────────────────────────────────────────────────────────

def pnl_per_dollar(p_market: float, side: str, outcome: int, fee: float) -> float:
    """
    BUY  (bet home wins): profit = (1-p)*(1-fee) if win, loss = -p
    SELL (bet home loses): profit = p*(1-fee) if win, loss = -(1-p)
    """
    if side == "buy":
        return (1.0 - p_market) * (1.0 - fee) if outcome == 1 else -p_market
    else:
        return p_market * (1.0 - fee) if outcome == 0 else -(1.0 - p_market)


def kelly_size(p_model: float, p_market: float, side: str, bankroll: float,
               fee: float, kelly_frac: float = 0.25,
               max_pct: float = 0.05, max_abs: float = 100.0) -> float:
    p_win    = p_model if side == "buy" else (1.0 - p_model)
    profit_r = (1.0 - p_market) * (1.0 - fee) if side == "buy" else p_market * (1.0 - fee)
    loss_r   = p_market if side == "buy" else (1.0 - p_market)
    if profit_r <= 0 or loss_r <= 0:
        return 0.0
    b          = profit_r / loss_r
    kelly_full = max(0.0, (b * p_win - (1.0 - p_win)) / b)
    size       = bankroll * kelly_full * kelly_frac
    return round(min(size, bankroll * max_pct, max_abs), 2)


def _binom_p(wins: int, n: int) -> float:
    try:
        return stats.binomtest(wins, n, 0.5, alternative="greater").pvalue
    except AttributeError:
        return stats.binom_test(wins, n, 0.5, alternative="greater")[1]


def _compute_stats(trades: list[dict], total_risked: float) -> dict:
    if not trades:
        return {"n": 0, "wins": 0, "wr": 0.0, "total_pnl": 0.0, "roi": 0.0,
                "sharpe": 0.0, "maxdd": 0.0, "binom_p": 1.0, "trades": []}
    pnls   = np.array([t["pnl"] for t in trades])
    curve  = np.cumsum(pnls)
    peak   = np.maximum.accumulate(curve)
    maxdd  = float((peak - curve).max())
    wins   = sum(1 for t in trades if t["won"])
    total  = float(pnls.sum())
    roi    = total / total_risked if total_risked else 0.0
    sharpe = float(pnls.mean() / pnls.std() * np.sqrt(252)) if pnls.std() > 0 else 0.0
    return {"n": len(trades), "wins": wins, "wr": wins / len(trades),
            "total_pnl": total, "roi": roi, "sharpe": sharpe,
            "maxdd": maxdd, "binom_p": _binom_p(wins, len(trades)), "trades": trades}


# ── Single-regime simulation (coin-flip range only) ──────────────────────────

def simulate(
    rows: list[dict],
    threshold: float,
    min_authors: int,
    bankroll: float,
    flat_bet: float = 50.0,
    kelly_frac: float = 0.0,      # >0 enables Kelly (flat_bet ignored)
    max_pct: float = 0.05,
    max_abs: float = 100.0,
    mkt_lo: float = 0.40,
    mkt_hi: float = 0.60,
    model_probs: np.ndarray | None = None,   # if None, random direction
    rng: random.Random | None = None,
) -> dict:
    """Bet when |model_prob - kalshi_prob| > threshold AND kalshi in [mkt_lo, mkt_hi]."""
    trades: list[dict] = []
    bk = bankroll
    total_risked = 0.0
    _rng = rng or random.Random(0)

    for idx, r in enumerate(rows):
        if r["home_num_users"] < min_authors or r["away_num_users"] < min_authors:
            continue
        pm = r["kalshi_prob"]
        if pm < mkt_lo or pm > mkt_hi:
            continue

        if model_probs is None:
            # Random baseline: coin-flip direction
            side = _rng.choice(["buy", "sell"])
        else:
            mp   = float(model_probs[idx])
            edge = mp - pm
            if abs(edge) <= threshold:
                continue
            side = "buy" if edge > 0 else "sell"

        if kelly_frac > 0:
            mp_for_kelly = float(model_probs[idx]) if model_probs is not None else (pm + 0.1)
            size = kelly_size(mp_for_kelly, pm, side, bk, KALSHI_FEE, kelly_frac, max_pct, max_abs)
        else:
            size = min(flat_bet, bk * 0.10)
        if size <= 0:
            continue

        pnl = round(pnl_per_dollar(pm, side, r["home_win"], KALSHI_FEE) * size, 4)
        total_risked += size
        bk = max(0.0, bk + pnl)
        trades.append({**r, "side": side, "size": size, "pnl": pnl, "won": pnl > 0})
        if bk <= 0:
            break

    return _compute_stats(trades, total_risked)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold",   type=float, default=0.05,
                    help="Min |model_prob - kalshi_price| edge to bet (default 0.05)")
    ap.add_argument("--mkt-lo",      type=float, default=0.40,
                    help="Lower Kalshi price bound (default 0.40)")
    ap.add_argument("--mkt-hi",      type=float, default=0.60,
                    help="Upper Kalshi price bound (default 0.60)")
    ap.add_argument("--min-authors", type=int,   default=0,
                    help="Min unique Reddit authors per team (default 0)")
    ap.add_argument("--flat-bet",    type=float, default=50.0,
                    help="Fixed $ per trade; use --kelly to switch to Kelly sizing")
    ap.add_argument("--kelly",       type=float, default=0.0,
                    help="Kelly fraction (0 = use flat-bet, default 0)")
    ap.add_argument("--bankroll",    type=float, default=1000.0)
    ap.add_argument("--folds",       type=int,   default=4)
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--rand-trials", type=int,   default=500,
                    help="Monte Carlo trials for random baseline (default 500)")
    ap.add_argument("--trades",      action="store_true")
    args = ap.parse_args()

    print(SEP2)
    print("  MODEL COMPARISON — 2025-26 NBA  (real Kalshi prices only)")
    print(f"  Market range : [{args.mkt_lo:.2f}, {args.mkt_hi:.2f}]   "
          f"Edge threshold : ±{args.threshold:.2f}")
    sizing = f"Kelly frac={args.kelly}" if args.kelly > 0 else f"Flat ${args.flat_bet:.0f}"
    print(f"  Sizing : {sizing}   Min authors/team : {args.min_authors}")
    print(f"  Bankroll : ${args.bankroll:.0f}   Random baseline : {args.rand_trials} trials")
    print(SEP2)

    # ── Load and merge ────────────────────────────────────────────────────────
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

    print(f"\nGames: {len(merged)}  ({merged['date'].min()} → {merged['date'].max()})")
    print(f"Home win rate: {merged['home_win'].mean():.1%}")

    struct_cols = get_struct_cols(merged)
    print(f"Feature counts — Struct: {len(struct_cols)}  "
          f"Mean sent: {len(SENT_MEAN)}  Full sent: {len(SENT_FULL)}")

    y            = merged["home_win"].values.astype(int)
    kalshi_probs = merged["kalshi_price"].values.astype(float)

    model_specs = [
        ("Model 1 — Baseline (struct only)",       struct_cols),
        ("Model 2 — Standard (struct+mean sent)",  struct_cols + SENT_MEAN),
        ("Model 3 — Thesis   (struct+full sent)",  struct_cols + SENT_FULL),
    ]

    # ── Walk-forward CV ───────────────────────────────────────────────────────
    print(f"\nRunning {args.folds}-fold walk-forward CV...")
    all_probs: dict[str, np.ndarray] = {}
    for name, feat_cols in model_specs:
        X = merged[feat_cols].fillna(0).values
        p = walk_forward_probs(X, y, n_folds=args.folds, seed=args.seed)
        all_probs[name] = p
        print(f"  {name[:42]:<42}: {int((~np.isnan(p)).sum())} OOS")

    # ── Build rows for simulation ─────────────────────────────────────────────
    rows_all: list[dict] = []
    for i, row in merged.iterrows():
        rows_all.append({
            "date":            str(row["date"])[:10],
            "home_team":       row["home_team"],
            "away_team":       row["away_team"],
            "home_win":        int(row["home_win"]),
            "kalshi_prob":     float(row["kalshi_price"]),
            "home_mean_sentiment": float(row.get("home_mean_sentiment", 0) or 0),
            "away_mean_sentiment": float(row.get("away_mean_sentiment", 0) or 0),
            "home_num_posts":  int(row.get("home_num_posts",  0) or 0),
            "away_num_posts":  int(row.get("away_num_posts",  0) or 0),
            "home_num_users":  int(row.get("home_num_users",  0) or 0),
            "away_num_users":  int(row.get("away_num_users",  0) or 0),
        })

    # OOS-only rows (rows where any model has a prediction)
    oos_mask = ~np.isnan(all_probs[model_specs[0][0]])

    sim_params = dict(
        min_authors=args.min_authors,
        bankroll=args.bankroll,
        flat_bet=args.flat_bet,
        kelly_frac=args.kelly,
        mkt_lo=args.mkt_lo,
        mkt_hi=args.mkt_hi,
    )

    # ── Prediction quality ────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print(f"  PREDICTION QUALITY (OOS, {args.folds}-fold walk-forward)")
    print(SEP2)
    print(f"  {'Model':<42}  {'N':>5}  {'Acc':>6}  {'AUC':>6}  {'Brier':>7}")
    print(SEP)

    y_oos  = y[oos_mask]
    kp_oos = kalshi_probs[oos_mask]
    print(f"  {'Kalshi market (reference)':<42}  {len(y_oos):>5}  "
          f"{np.mean((kp_oos >= 0.5).astype(int) == y_oos):>6.3f}  "
          f"{roc_auc_score(y_oos, kp_oos):>6.3f}  "
          f"{brier_score_loss(y_oos, kp_oos):>7.4f}")

    model_preds: dict[str, dict] = {}
    for name, _ in model_specs:
        p    = all_probs[name]
        mask = ~np.isnan(p)
        yy, pp = y[mask], p[mask]
        model_preds[name] = {
            "acc": float(np.mean((pp >= 0.5).astype(int) == yy)),
            "auc": float(roc_auc_score(yy, pp)),
            "brier": float(brier_score_loss(yy, pp)),
            "n": int(mask.sum()),
        }
        print(f"  {name:<42}  {int(mask.sum()):>5}  "
              f"{model_preds[name]['acc']:>6.3f}  "
              f"{model_preds[name]['auc']:>6.3f}  "
              f"{model_preds[name]['brier']:>7.4f}")

    # ── Trading simulation ────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print(f"  TRADING SIMULATION  Kalshi [{args.mkt_lo:.2f},{args.mkt_hi:.2f}]  "
          f"edge ±{args.threshold:.2f}  {sizing}")
    print(SEP2)
    print(f"  {'Strategy':<42}  {'N':>5}  {'WR':>6}  {'PnL':>9}  {'Sharpe':>7}  "
          f"{'MaxDD':>9}  {'p':>7}")
    print(SEP)

    sim_results: dict[str, dict] = {}

    # Model-driven simulations
    for name, _ in model_specs:
        p = all_probs[name]
        # Build per-row model_probs aligned to rows_all (NaN where not OOS)
        mp_array = p  # indexed same as merged
        rows_oos = [r for i, r in enumerate(rows_all) if oos_mask[i]]
        mp_oos   = p[oos_mask]

        sim = simulate(rows_oos, threshold=args.threshold,
                       model_probs=mp_oos, **sim_params)
        sim_results[name] = sim
        _print_sim_row(name, sim)

    # Random baseline — Monte Carlo over eligible games
    eligible_rows = [r for i, r in enumerate(rows_all)
                     if oos_mask[i]
                     and args.mkt_lo <= r["kalshi_prob"] <= args.mkt_hi
                     and r["home_num_users"] >= args.min_authors
                     and r["away_num_users"] >= args.min_authors]
    rand_pnls, rand_wrs, rand_sharpes = [], [], []
    for trial in range(args.rand_trials):
        rng_trial = random.Random(args.seed + trial)
        s = simulate(eligible_rows, threshold=0.0,
                     model_probs=None, rng=rng_trial, **sim_params)
        if s["n"] > 0:
            rand_pnls.append(s["total_pnl"])
            rand_wrs.append(s["wr"])
            rand_sharpes.append(s["sharpe"])
    if rand_pnls:
        rand_name = f"Random baseline ({args.rand_trials} trials avg)"
        rand_sim  = {
            "n": len(eligible_rows),
            "wr": float(np.mean(rand_wrs)),
            "total_pnl": float(np.mean(rand_pnls)),
            "sharpe": float(np.mean(rand_sharpes)),
            "maxdd": 0.0,
            "binom_p": float(np.mean([r >= 0.5 for r in rand_wrs])),
            "trades": [],
        }
        sim_results["random"] = rand_sim
        _print_sim_row(rand_name, rand_sim)

    print(SEP)

    # ── Detailed breakdown ────────────────────────────────────────────────────
    for name, _ in model_specs:
        sim  = sim_results[name]
        pred = model_preds[name]
        if sim["n"] == 0:
            continue
        buys  = [t for t in sim["trades"] if t["side"] == "buy"]
        sells = [t for t in sim["trades"] if t["side"] == "sell"]

        def _wr(lst): return sum(t["won"] for t in lst) / len(lst) if lst else 0.0
        def _p(lst):  return sum(t["pnl"] for t in lst)

        print(f"\n{SEP}")
        print(f"  {name}")
        print(SEP)
        print(f"  Prediction   : Acc={pred['acc']:.3f}  AUC={pred['auc']:.3f}  "
              f"Brier={pred['brier']:.4f}")
        print(f"  Trades       : {sim['n']}  ({sim['wins']}W / {sim['n']-sim['wins']}L)  "
              f"WR={sim['wr']:.1%}")
        print(f"  PnL          : ${sim['total_pnl']:+.2f}   ROI={sim['roi']:+.1%}   "
              f"Sharpe={sim['sharpe']:.2f}   MaxDD=-${sim['maxdd']:.2f}")
        print(f"  Binomial p   : {sim['binom_p']:.4f}")
        if buys:
            print(f"  BUY  home    : N={len(buys):3d}  WR={_wr(buys):.1%}  PnL=${_p(buys):+.2f}")
        if sells:
            print(f"  SELL home    : N={len(sells):3d}  WR={_wr(sells):.1%}  PnL=${_p(sells):+.2f}")

        if args.trades:
            print(f"\n  {'Date':<12} {'Match':<22} {'Side':<5} {'Edge':>6} "
                  f"{'Mkt':>5} {'$':>6} {'PnL':>8}")
            for t in sim["trades"]:
                match = f"{t['away_team'][:3]}@{t['home_team'][:3]}"
                edge  = t.get("model_prob", 0) - t["kalshi_prob"]
                print(f"  {t['date']:<12} {match:<22} {t['side']:<5} "
                      f"{edge:>+6.3f} {t['kalshi_prob']:>5.3f} ${t['size']:>5.2f} "
                      f"{t['pnl']:>+8.4f} {'✓' if t['won'] else '✗'}")

    print()


def _print_sim_row(name: str, sim: dict) -> None:
    if sim["n"] == 0:
        print(f"  {name:<42}  {'—':>5}  (no trades)")
        return
    pnl_s   = f"+${sim['total_pnl']:.2f}" if sim["total_pnl"] >= 0 else f"-${abs(sim['total_pnl']):.2f}"
    maxdd_s = f"-${sim['maxdd']:.2f}" if sim["maxdd"] > 0 else "—"
    print(f"  {name:<42}  {sim['n']:>5}  {sim['wr']:>6.1%}  "
          f"{pnl_s:>9}  {sim['sharpe']:>7.2f}  {maxdd_s:>9}  "
          f"{sim['binom_p']:>7.4f}")


if __name__ == "__main__":
    main()
