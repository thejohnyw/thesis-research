"""
Strategy comparison backtest — 2025-26 NBA season (regular + playoffs).

Data sources (in order of preference):
  1. REAL Kalshi prices + real reddit posts + real outcomes (playoff games
     where the live bots were running — Apr 28 → present).
     Reddit posts are time-gated by created_utc < game_start_utc (no leakage).

  2. RF-proxy Kalshi + pre-computed sentiment (regular season Oct–Mar).
     Kalshi historical data unavailable for those settled markets; the
     structured-only RandomForest is the best available substitute.
     Sentiment was already time-gated when the training CSV was built.

Strategies:
  DirectSentiment   — fire if |home_sent − away_sent| >= threshold
  AntiBotSentiment  — original 3-regime version (including Regime 2)
  AntiBotClean      — Regimes 1+3 only (Regime 2 removed after backtest)
  OrderBookAnchor   — same as AntiBotClean (no historical orderbook)
  RandomBaseline    — coin-flip control

Usage:
    python scripts/backtest_strategies.py
    python scripts/backtest_strategies.py --min-posts 5 --trades --verbose
    python scripts/backtest_strategies.py --threshold 0.08 --flat-bet 50
"""
from __future__ import annotations

import argparse, os, random, sqlite3, sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.user_features import SENTIMENT_FEATURE_COLS, aggregate_by_user
from backend.config import KALSHI_FEE, KELLY_FRACTION
from backend.data.markets import parse_kalshi_ticker

REG_SEASON_CSV = "data/processed/training_data_with_sentiment.csv"

# Live bot databases that carry real Kalshi prices + outcomes
LIVE_DBS = [
    "data/trading.db",           # Apr 28 – May 13 (first-round playoffs)
    "data/trading_sentiment.db", # May 13 – present (conf. semis + finals)
]

_EXCLUDE = {
    "game_id","date","home_team","away_team",
    "home_score","away_score","home_win","cutoff_utc",
    "home_num_posts","home_num_users","away_num_posts","away_num_users",
    "diff_mean_sentiment","diff_std_sentiment",
    "diff_mean_user_std","diff_user_entropy",
}
SEP  = "─" * 70
SEP2 = "═" * 70


# ── Sizing & P&L ─────────────────────────────────────────────────────────────

def kelly_size(p_win, p_market, side, bankroll, fee, kelly_frac,
               max_pct=0.05, max_abs=100.0):
    if side == "buy":
        profit_r = (1.0 - p_market) * (1.0 - fee)
        loss_r   = p_market
    else:
        profit_r = p_market * (1.0 - fee)
        loss_r   = 1.0 - p_market
    b = profit_r / loss_r if loss_r > 0 else 0.0
    if b <= 0:
        return 0.0
    kelly_full = max(0.0, (p_win * b - (1.0 - p_win)) / b)
    return round(min(bankroll * kelly_full * kelly_frac, bankroll * max_pct, max_abs), 2)


def pnl_per_dollar(p_market, side, outcome, fee):
    if side == "buy":
        return (1.0 - p_market) * (1.0 - fee) if outcome == 1 else -p_market
    else:
        return p_market * (1.0 - fee) if outcome == 0 else -(1.0 - p_market)


# ── Walk-forward RF market proxy (regular season only) ────────────────────────

def _make_rf(seed=42):
    return RandomForestClassifier(
        n_estimators=150, max_depth=6, min_samples_leaf=8,
        max_features="sqrt", random_state=seed, n_jobs=-1,
    )

def _fit_predict(X_tr, y_tr, X_te, seed):
    sc = StandardScaler()
    m  = _make_rf(seed)
    m.fit(sc.fit_transform(X_tr), y_tr)
    return m.predict_proba(sc.transform(X_te))[:, 1]

def build_rf_market_probs(df, n_folds, seed):
    struct_cols = [c for c in df.columns
                   if c not in _EXCLUDE and c not in SENTIMENT_FEATURE_COLS]
    X = df[struct_cols].fillna(0).values
    y = df["home_win"].values.astype(int)
    p = np.full(len(df), np.nan)
    for fold_i, (tr, te) in enumerate(TimeSeriesSplit(n_splits=n_folds).split(X)):
        p[te] = _fit_predict(X[tr], y[tr], X[te], seed + fold_i)
    return p


# ── Real Kalshi prices + outcomes from live DBs ───────────────────────────────

def _game_start_utc(scheduled_time: str) -> int | None:
    """Return scheduled_time as Unix timestamp, or None if unparseable."""
    if not scheduled_time:
        return None
    from datetime import datetime
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S+00:00", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(scheduled_time, fmt)
            from datetime import timezone
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            pass
    return None


def load_live_game_rows(min_posts: int, reddit_hours: int = 48) -> list[dict]:
    """
    Extract one record per physical game from the live bot databases.
    Each record has:
        date, home_team, away_team, home_win (outcome),
        kalshi_prob (first price observed for that game),
        home_mean_sentiment, away_mean_sentiment,
        home_num_posts, away_num_posts,
        source='live_kalshi'

    Dedup: keep only the home-team YES contract (where side_abbr == home_abbr).
    Time-gate: only reddit posts with created_utc < game_start_utc.
    """
    seen_games: dict[tuple, dict] = {}  # (home, away, date) -> row

    for db_path in LIVE_DBS:
        if not os.path.exists(db_path):
            continue
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Settled games only
        games = conn.execute("""
            SELECT id, external_id, home_team, away_team,
                   scheduled_time, outcome, kalshi_ticker
            FROM games
            WHERE outcome IS NOT NULL AND kalshi_ticker IS NOT NULL
        """).fetchall()

        for g in games:
            ticker = g["kalshi_ticker"] or ""
            parsed = parse_kalshi_ticker(ticker)
            if not parsed:
                continue

            # Keep only home-team YES contract (side_abbr == home_abbr)
            if parsed["side_abbr"] != parsed["home_abbr"]:
                continue

            home = g["home_team"]
            away = g["away_team"]
            sched = g["scheduled_time"] or ""
            date  = sched[:10]
            key   = (home, away, date)

            # Get earliest Kalshi price (before game started)
            start_ts = _game_start_utc(sched)
            if start_ts:
                ts_filter = f"AND timestamp < datetime({start_ts}, 'unixepoch')"
            else:
                ts_filter = ""

            price_row = conn.execute(f"""
                SELECT home_prob FROM odds_snapshots
                WHERE game_id = ? AND source = 'kalshi' {ts_filter}
                ORDER BY timestamp ASC LIMIT 1
            """, (g["id"],)).fetchone()

            if price_row is None:
                continue   # no pre-game price snapshot → skip

            kalshi_prob = float(price_row["home_prob"])

            # Time-gated reddit sentiment
            if start_ts:
                since_ts = start_ts - reddit_hours * 3600
                h_posts_raw = conn.execute("""
                    SELECT author, sentiment FROM reddit_posts
                    WHERE team = ? AND created_utc < ? AND created_utc >= ?
                      AND sentiment IS NOT NULL AND sentiment != 0.0
                """, (home, start_ts, since_ts)).fetchall()
                a_posts_raw = conn.execute("""
                    SELECT author, sentiment FROM reddit_posts
                    WHERE team = ? AND created_utc < ? AND created_utc >= ?
                      AND sentiment IS NOT NULL AND sentiment != 0.0
                """, (away, start_ts, since_ts)).fetchall()
            else:
                h_posts_raw = conn.execute("""
                    SELECT author, sentiment FROM reddit_posts
                    WHERE team = ? AND sentiment IS NOT NULL AND sentiment != 0.0
                """, (home,)).fetchall()
                a_posts_raw = conn.execute("""
                    SELECT author, sentiment FROM reddit_posts
                    WHERE team = ? AND sentiment IS NOT NULL AND sentiment != 0.0
                """, (away,)).fetchall()

            h_posts = [{"author": r["author"], "sentiment": r["sentiment"]} for r in h_posts_raw]
            a_posts = [{"author": r["author"], "sentiment": r["sentiment"]} for r in a_posts_raw]

            h_agg = aggregate_by_user(h_posts)
            a_agg = aggregate_by_user(a_posts)

            row = {
                "date":                date,
                "home_team":           home,
                "away_team":           away,
                "home_win":            int(g["outcome"]),
                "kalshi_prob":         kalshi_prob,
                "home_mean_sentiment": float(h_agg["mean_sentiment"]),
                "away_mean_sentiment": float(a_agg["mean_sentiment"]),
                "home_num_posts":      int(h_agg["num_posts"]),
                "away_num_posts":      int(a_agg["num_posts"]),
                "source":              "live_kalshi",
                "ticker":              ticker,
            }
            seen_games[key] = row   # later DB entry overwrites (more recent price)

        conn.close()

    rows = sorted(seen_games.values(), key=lambda r: r["date"])
    return rows


# ── Signal rules ─────────────────────────────────────────────────────────────

def signal_direct(diff, threshold, **_):
    if diff > threshold:  return "buy"
    if diff < -threshold: return "sell"
    return None

def signal_antibot(diff, p_market, threshold, agree=0.55, **_):
    if diff > threshold  and p_market >= agree:           return "buy"
    if diff < -threshold and p_market >= agree:           return "sell"
    if diff < -threshold and p_market < (1.0 - agree):   return "sell"
    return None

def signal_antibot_clean(diff, p_market, threshold, agree=0.55, **_):
    if diff > threshold  and p_market >= agree:           return "buy"
    if diff < -threshold and p_market < (1.0 - agree):   return "sell"
    return None

signal_orderbook = signal_antibot_clean


# ── Core backtest loop ────────────────────────────────────────────────────────

def run_strategy(name, signal_fn, rows, threshold, min_posts,
                 kelly_frac, bankroll, flat_bet, rng):
    trades, no_signal, blocked_posts = [], 0, 0
    bankroll_now = bankroll
    real_count = 0

    for row in rows:
        hn = int(row.get("home_num_posts", 0) or 0)
        an = int(row.get("away_num_posts", 0) or 0)
        if min_posts > 0 and (hn < min_posts or an < min_posts):
            blocked_posts += 1
            continue

        pm   = float(row["kalshi_prob"])
        diff = float(row.get("home_mean_sentiment", 0) or 0) - \
               float(row.get("away_mean_sentiment", 0) or 0)

        if name == "RandomBaseline":
            side = rng.choice(["buy", "sell"])
        elif name == "DirectSentiment":
            side = signal_fn(diff, threshold)
        else:
            side = signal_fn(diff, pm, threshold)

        if side is None:
            no_signal += 1
            continue

        if flat_bet > 0:
            size = min(flat_bet, bankroll_now * 0.10)
        else:
            p_win = (pm + min(abs(diff), 0.20) * 0.5) if side == "buy" \
                    else ((1 - pm) + min(abs(diff), 0.20) * 0.5)
            p_win = min(max(p_win, 0.01), 0.99)
            size  = kelly_size(p_win, pm, side, bankroll_now,
                               KALSHI_FEE, kelly_frac)

        if size < 1.0:
            no_signal += 1
            continue

        outcome = int(row["home_win"])
        pnl     = round(pnl_per_dollar(pm, side, outcome, KALSHI_FEE) * size, 4)
        is_real = (row.get("source") == "live_kalshi")

        trades.append({
            "date":       row["date"],
            "matchup":    f"{row['away_team']} @ {row['home_team']}",
            "side":       side,
            "diff":       round(diff, 4),
            "p_market":   round(pm, 4),
            "size":       size,
            "outcome":    outcome,
            "won":        pnl > 0,
            "pnl":        pnl,
            "real_price": is_real,
        })
        if is_real:
            real_count += 1
        bankroll_now = max(0.0, bankroll_now + pnl)
        if bankroll_now <= 0:
            break

    return {
        "name":          name,
        "trades":        trades,
        "bankroll0":     bankroll,
        "bankroll_f":    bankroll_now,
        "no_signal":     no_signal,
        "blocked_posts": blocked_posts,
        "n_evaluated":   len(rows) - blocked_posts,
        "real_count":    real_count,
    }


# ── Stats ─────────────────────────────────────────────────────────────────────

def compute_stats(r):
    t = r["trades"]
    n = len(t)
    if n == 0:
        return {**r, "n":0,"wins":0,"wr":0,"total_pnl":0,
                "roi":0,"sharpe":0,"max_dd":0,"avg_diff":0,
                "t_pval":1.0,"binom_pval":1.0}
    pnls   = np.array([x["pnl"]  for x in t])
    sizes  = np.array([x["size"] for x in t])
    wins   = sum(1 for x in t if x["won"])
    total  = float(pnls.sum())
    roi    = total / float(sizes.sum()) if sizes.sum() else 0
    sharpe = float(pnls.mean()/pnls.std()*np.sqrt(252)) if pnls.std()>0 else 0.0
    curve  = np.cumsum(pnls)
    max_dd = float((curve - np.maximum.accumulate(curve)).min())
    _, t_pval     = stats.ttest_1samp(pnls, 0)
    binom_pval    = stats.binomtest(wins, n, 0.5, alternative="greater").pvalue
    return {
        **r, "n":n, "wins":wins, "wr":wins/n,
        "total_pnl":total, "roi":roi, "sharpe":sharpe,
        "max_dd":max_dd, "avg_diff":float(np.mean([abs(x["diff"]) for x in t])),
        "t_pval":t_pval, "binom_pval":binom_pval,
    }


# ── Output ────────────────────────────────────────────────────────────────────

def print_comparison(results, n_reg, n_live):
    real_note = f"({n_reg} reg-season RF proxy + {n_live} playoffs real Kalshi)"
    print(f"\n{SEP2}")
    print(f"  STRATEGY COMPARISON — 2025-26 NBA  {real_note}")
    print(SEP2)
    print(f"  {'Strategy':<22} {'N':>5} {'WR':>7} {'PnL':>9} {'ROI':>7} "
          f"{'Sharpe':>7} {'MaxDD':>8} {'p(bin)':>8}")
    print(SEP)
    for s in results:
        if s["n"] == 0:
            print(f"  {s['name']:<22} {'—':>5}  (no trades)")
            continue
        pnl_s = f"+${s['total_pnl']:.2f}" if s["total_pnl"]>=0 else f"-${abs(s['total_pnl']):.2f}"
        real_str = f" [{s['real_count']} real]" if s['real_count'] else ""
        print(f"  {s['name']:<22} {s['n']:>5} {s['wr']:>7.1%} {pnl_s:>9} "
              f"{s['roi']:>+7.1%} {s['sharpe']:>7.2f} ${s['max_dd']:>7.2f} "
              f"{s['binom_pval']:>8.4f}{real_str}")
    print(SEP2)


def print_detail(s, show_trades=False, verbose=False):
    t = s["trades"]
    n = s["n"]
    real = s.get("real_count", 0)
    if n == 0:
        print(f"\n[{s['name']}] No trades (blocked={s['blocked_posts']}, no-sig={s['no_signal']})")
        return
    buys = sum(1 for x in t if x["side"]=="buy")
    print(f"\n{SEP}")
    print(f"  {s['name'].upper()}")
    print(SEP)
    print(f"  Games evaluated  : {s['n_evaluated']:,}  "
          f"(blocked posts: {s['blocked_posts']}  no-signal: {s['no_signal']})")
    print(f"  Trades placed    : {n}  ({buys} buy / {n-buys} sell)")
    print(f"  Real Kalshi bets : {real} / {n}  "
          f"({real/n:.0%} of trades used actual market prices)")
    print(f"  Win rate         : {s['wr']:.1%}  ({s['wins']}/{n})")
    print(f"  Total PnL        : ${s['total_pnl']:+.2f}")
    print(f"  Final bankroll   : ${s['bankroll0']:.0f} → ${s['bankroll_f']:.2f}"
          f"  ({(s['bankroll_f']-s['bankroll0'])/s['bankroll0']:+.1%})")
    print(f"  ROI (per $bet)   : {s['roi']:+.1%}")
    print(f"  Sharpe (ann.)    : {s['sharpe']:.2f}")
    print(f"  Max drawdown     : ${s['max_dd']:.2f}")
    print(f"  Avg |diff|       : {s['avg_diff']:+.4f}")
    print(f"  t-test p-value   : {s['t_pval']:.4f}")
    print(f"  Binomial p-val   : {s['binom_pval']:.4f}  (H0: WR ≤ 50%)")

    if verbose:
        monthly = defaultdict(list)
        for x in t:
            monthly[x["date"][:7]].append(x["pnl"])
        print("\n  Monthly breakdown:")
        for m in sorted(monthly):
            ps = monthly[m]
            wr = sum(1 for p in ps if p > 0) / len(ps)
            print(f"    {m}: {len(ps):>3} trades  PnL ${sum(ps):>+7.2f}  WR {wr:.0%}")

    if show_trades:
        print(f"\n  {'Src':<3} {'Date':<12} {'Matchup':<35} {'Side':<5} "
              f"{'Diff':>6} {'Mkt':>5} {'$':>6} {'PnL':>8}  W?")
        for x in t:
            mk = "✓" if x["won"] else "✗"
            src = "★" if x["real_price"] else "~"
            print(f"  {src:<3} {x['date']:<12} {x['matchup']:<34} {x['side']:<5} "
                  f"{x['diff']:>+6.3f} {x['p_market']:>5.3f} ${x['size']:>5.2f} "
                  f"{x['pnl']:>+8.4f} {mk}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Strategy backtest — real Kalshi prices where available")
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--min-posts", type=int,   default=0,
                    help="Min posts per team (default 0; try 5 for cleaner signal)")
    ap.add_argument("--kelly",     type=float, default=KELLY_FRACTION)
    ap.add_argument("--flat-bet",  type=float, default=50.0,
                    help="Fixed $ per trade (0 = fractional Kelly, default $50)")
    ap.add_argument("--bankroll",  type=float, default=1000.0)
    ap.add_argument("--folds",     type=int,   default=4)
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--reddit-hours", type=int, default=48,
                    help="Look-back window for reddit posts (live games, default 48h)")
    ap.add_argument("--verbose",   action="store_true")
    ap.add_argument("--trades",    action="store_true")
    args = ap.parse_args()

    print(SEP2)
    print("  STRATEGY BACKTEST — 2025-26 NBA (real Kalshi prices where available)")
    print(f"  Threshold  : ±{args.threshold:.2f}    Min posts : {args.min_posts}")
    print(f"  Flat bet   : {'$'+str(args.flat_bet) if args.flat_bet else 'Kelly'}")
    print(f"  Bankroll   : ${args.bankroll:.0f}   Reddit window : {args.reddit_hours}h")
    print(SEP2)

    # ── Phase 1: regular season → RF proxy ───────────────────────────────────
    print(f"\nPhase 1 — Regular season (RF proxy for Kalshi prices)...")
    reg_df = pd.read_csv(REG_SEASON_CSV).sort_values("date").reset_index(drop=True)
    print(f"  {len(reg_df)} games  ({reg_df['date'].min()} → {reg_df['date'].max()})")

    print(f"  Building walk-forward market probs ({args.folds} folds)...")
    p_rf = build_rf_market_probs(reg_df, args.folds, args.seed)
    n_oos = int((~np.isnan(p_rf)).sum())
    print(f"  OOS estimates: {n_oos} games")

    # Build reg-season row dicts
    reg_rows = []
    for i, row in reg_df.iterrows():
        pm = p_rf[i]
        if np.isnan(pm):
            continue
        reg_rows.append({
            "date":                str(row["date"])[:10],
            "home_team":           row["home_team"],
            "away_team":           row["away_team"],
            "home_win":            int(row["home_win"]),
            "kalshi_prob":         float(pm),
            "home_mean_sentiment": float(row.get("home_mean_sentiment", 0) or 0),
            "away_mean_sentiment": float(row.get("away_mean_sentiment", 0) or 0),
            "home_num_posts":      int(row.get("home_num_posts", 0) or 0),
            "away_num_posts":      int(row.get("away_num_posts", 0) or 0),
            "source":              "rf_proxy",
        })

    # ── Phase 2: playoffs → real Kalshi prices ────────────────────────────────
    print(f"\nPhase 2 — Playoffs (real Kalshi prices from live bot DBs)...")
    live_rows = load_live_game_rows(min_posts=0, reddit_hours=args.reddit_hours)
    print(f"  {len(live_rows)} unique settled playoff games with real prices")
    if live_rows:
        dates = sorted(set(r["date"] for r in live_rows))
        print(f"  Date range: {dates[0]} → {dates[-1]}")
        have_posts = sum(1 for r in live_rows
                        if r["home_num_posts"] >= 5 and r["away_num_posts"] >= 5)
        print(f"  Games with ≥5 posts/team: {have_posts}")

    # ── Combine ───────────────────────────────────────────────────────────────
    all_rows = reg_rows + live_rows
    all_rows.sort(key=lambda r: r["date"])
    print(f"\nCombined dataset: {len(all_rows)} games "
          f"({len(reg_rows)} RF proxy + {len(live_rows)} real Kalshi)")

    # ── Run strategies ────────────────────────────────────────────────────────
    strategy_configs = [
        ("DirectSentiment",  signal_direct),
        ("AntiBotSentiment", signal_antibot),
        ("AntiBotClean",     signal_antibot_clean),
        ("OrderBookAnchor",  signal_orderbook),
        ("RandomBaseline",   None),
    ]

    rng = random.Random(args.seed)
    results = []
    for name, fn in strategy_configs:
        print(f"\nSimulating {name}...")
        r = run_strategy(
            name=name, signal_fn=fn, rows=all_rows,
            threshold=args.threshold, min_posts=args.min_posts,
            kelly_frac=args.kelly, bankroll=args.bankroll,
            flat_bet=args.flat_bet, rng=rng,
        )
        results.append(compute_stats(r))

    print_comparison(results, n_reg=len(reg_rows), n_live=len(live_rows))
    for s in results:
        print_detail(s, show_trades=args.trades, verbose=args.verbose)
    print()


if __name__ == "__main__":
    main()
