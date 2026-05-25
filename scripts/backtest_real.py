"""
Backtest against real Kalshi market data stored in trading.db.

Unlike scripts/backtest.py (which simulates Kalshi prices from a logistic
model), this replays actual odds snapshots the live bot collected. Entry
price = first Kalshi snapshot for a game; closing price = last snapshot
(used for CLV). Sharp odds come from the first snapshot of each sharp book.

Requirements:
  - Run the live bot's collect_odds() for at least a few games first.
  - Games must have settled (outcome != NULL) to appear in backtest.

Usage:
    python scripts/backtest_real.py
    python scripts/backtest_real.py --strategy sharp --edge 0.03
    python scripts/backtest_real.py --strategy random --seed 42 --trade-prob 0.5
    python scripts/backtest_real.py --bankroll 500 --verbose
"""
import sys
import os
import argparse
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from backend.config import DATABASE_PATH, KALSHI_FEE, KELLY_FRACTION, SHARP_BOOKS
from backend.core.settlement import calculate_pnl, calculate_clv
from backend.core.strategy import Strategy, SharpVsKalshiStrategy, RandomStrategy


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_history(db_path: str) -> list[dict]:
    """
    Load all settled games that have at least one Kalshi odds snapshot.
    Returns one row per game with kalshi_entry, kalshi_close, and sharp_odds.
    """
    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    games = conn.execute("""
        SELECT g.id, g.home_team, g.away_team, g.outcome,
               COALESCE(g.scheduled_time, '') AS scheduled_time,
               g.kalshi_ticker
        FROM games g
        WHERE g.outcome IS NOT NULL
          AND EXISTS (
              SELECT 1 FROM odds_snapshots
              WHERE game_id = g.id AND source = 'kalshi'
          )
        ORDER BY g.scheduled_time
    """).fetchall()

    result = []
    for g in games:
        gid = g["id"]

        k_entry = conn.execute("""
            SELECT home_prob, timestamp FROM odds_snapshots
            WHERE game_id = ? AND source = 'kalshi'
            ORDER BY id ASC LIMIT 1
        """, (gid,)).fetchone()

        k_close = conn.execute("""
            SELECT home_prob FROM odds_snapshots
            WHERE game_id = ? AND source = 'kalshi'
            ORDER BY id DESC LIMIT 1
        """, (gid,)).fetchone()

        # First snapshot per sharp book
        sharp_rows = conn.execute("""
            SELECT source, home_prob FROM odds_snapshots
            WHERE game_id = ?
              AND source != 'kalshi'
              AND id IN (
                  SELECT MIN(id) FROM odds_snapshots
                  WHERE game_id = ? AND source != 'kalshi'
                  GROUP BY source
              )
        """, (gid, gid)).fetchall()

        result.append({
            "id": gid,
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "outcome": g["outcome"],
            "scheduled_time": g["scheduled_time"],
            "kalshi_ticker": g["kalshi_ticker"],
            "kalshi_entry": k_entry["home_prob"] if k_entry else None,
            "kalshi_close": k_close["home_prob"] if k_close else None,
            "entry_time": k_entry["timestamp"] if k_entry else None,
            "sharp_odds": {r["source"]: r["home_prob"] for r in sharp_rows},
        })

    conn.close()
    return result


def _best_sharp_prob(sharp_odds: dict) -> Optional[float]:
    """Average top two available sharp books in priority order."""
    probs = [sharp_odds[b] for b in SHARP_BOOKS if b in sharp_odds]
    if not probs:
        return None
    return sum(probs[:2]) / len(probs[:2])


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class BtTrade:
    date: str
    game_id: int
    matchup: str
    side: str
    kalshi_entry: float
    kalshi_close: float
    sharp_prob: Optional[float]
    edge: Optional[float]
    size: float
    kelly: float
    won: bool
    pnl: float
    clv: float


@dataclass
class BacktestResult:
    trades: list[BtTrade] = field(default_factory=list)
    initial_bankroll: float = 1000.0
    games_seen: int = 0
    games_skipped_no_odds: int = 0

    # ── aggregates ──

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        return sum(1 for t in self.trades if t.won) / len(self.trades) if self.trades else 0.0

    @property
    def roi(self) -> float:
        risked = sum(t.size for t in self.trades)
        return self.total_pnl / risked if risked else 0.0

    @property
    def sharpe(self) -> float:
        pnls = [t.pnl for t in self.trades]
        if len(pnls) < 2 or np.std(pnls) == 0:
            return 0.0
        return float(np.mean(pnls) / np.std(pnls) * np.sqrt(252))

    @property
    def avg_clv(self) -> float:
        clvs = [t.clv for t in self.trades]
        return float(np.mean(clvs)) if clvs else 0.0

    @property
    def max_drawdown(self) -> float:
        if not self.trades:
            return 0.0
        curve = np.cumsum([t.pnl for t in self.trades])
        return float((curve - np.maximum.accumulate(curve)).min())

    def summary(self, verbose: bool = True) -> str:
        if not self.trades:
            return (
                f"No trades. "
                f"({self.games_seen} settled games in DB, "
                f"{self.games_skipped_no_odds} missing Kalshi odds)"
            )

        final = self.initial_bankroll + self.total_pnl
        buys = sum(1 for t in self.trades if t.side == "buy")
        lines = [
            f"Games in DB:  {self.games_seen}  "
            f"({self.games_skipped_no_odds} skipped — no Kalshi odds)",
            f"Trades:       {self.n_trades} ({buys} buys / {self.n_trades - buys} sells)",
            f"Win rate:     {self.win_rate:.1%}",
            f"Total PnL:    ${self.total_pnl:+.2f}",
            f"Final:        ${self.initial_bankroll:.0f} → ${final:.2f}"
            f"  ({self.total_pnl / self.initial_bankroll:+.1%})",
            f"ROI:          {self.roi:+.1%}  (per dollar risked)",
            f"Sharpe:       {self.sharpe:.2f}",
            f"Max DD:       ${self.max_drawdown:.2f}",
            f"Avg CLV:      {self.avg_clv:+.4f}",
            f"Avg kelly:    {np.mean([t.kelly for t in self.trades]):.1%}",
            f"Avg size:     ${np.mean([t.size for t in self.trades]):.2f}",
        ]

        if verbose:
            monthly: dict[str, list[float]] = defaultdict(list)
            for t in self.trades:
                monthly[t.date[:7]].append(t.pnl)
            if monthly:
                lines.append("\nMonthly:")
                for m in sorted(monthly):
                    ps = monthly[m]
                    wr = sum(1 for p in ps if p > 0) / len(ps)
                    lines.append(
                        f"  {m}: {len(ps):>3} trades  "
                        f"PnL ${sum(ps):>+7.2f}  WR {wr:.0%}"
                    )

        return "\n".join(lines)


# ── Core backtest engine ──────────────────────────────────────────────────────

def run_backtest(
    strategy: Strategy,
    bankroll: float = 1000.0,
    db_path: str = DATABASE_PATH,
    daily_loss_limit: float = 200.0,
    max_position_pct: float = 0.05,
    max_position_dollars: float = 100.0,
) -> BacktestResult:
    """
    Replay historical Kalshi data from db_path through strategy.

    Position sizing mirrors the live bot: fractional Kelly with hard caps.
    Daily P&L circuit breaker stops sizing if daily loss exceeds the limit.
    """
    result = BacktestResult(initial_bankroll=bankroll)

    history = _load_history(db_path)
    result.games_seen = len(history)

    current_bankroll = bankroll
    daily_pnl = 0.0
    last_date = None

    for game in history:
        if game["kalshi_entry"] is None:
            result.games_skipped_no_odds += 1
            continue

        kalshi_entry: float = game["kalshi_entry"]
        kalshi_close: float = game["kalshi_close"] or kalshi_entry
        sharp_prob: Optional[float] = _best_sharp_prob(game["sharp_odds"])
        outcome: int = game["outcome"]
        date: str = (game["scheduled_time"] or "")[:10]
        matchup = f"{game['away_team']} @ {game['home_team']}"

        # Daily reset
        if date != last_date:
            daily_pnl = 0.0
            last_date = date

        # Circuit breaker
        if daily_pnl <= -daily_loss_limit:
            continue

        sig = strategy.signal(game, kalshi_entry, sharp_prob)
        if sig is None:
            continue

        # Edge for Kelly
        if sig.edge_override is not None:
            edge = sig.edge_override
        elif sharp_prob is not None:
            edge = abs(sharp_prob - kalshi_entry)
        else:
            edge = 0.01

        # Kelly sizing
        if sig.side == "buy":
            win_prob = min(0.99, kalshi_entry + edge)
            profit_r = (1 - kalshi_entry) * (1 - KALSHI_FEE)
            loss_r = kalshi_entry
        else:
            win_prob = min(0.99, 1 - (kalshi_entry - edge))
            profit_r = kalshi_entry * (1 - KALSHI_FEE)
            loss_r = 1 - kalshi_entry

        b = profit_r / loss_r if loss_r > 0 else 0
        if b <= 0:
            continue

        kelly_full = max(0, (win_prob * b - (1 - win_prob)) / b)
        kelly_adj = kelly_full * KELLY_FRACTION * sig.confidence
        size = current_bankroll * kelly_adj
        size = min(size, current_bankroll * max_position_pct, max_position_dollars)
        size = round(size, 2)

        if size < 1.0:
            continue

        # Settle
        pnl_per = calculate_pnl(kalshi_entry, sig.side, outcome)
        pnl = round(pnl_per * size, 4)
        clv = round(calculate_clv(kalshi_entry, kalshi_close, sig.side), 4)
        won = pnl > 0

        result.trades.append(BtTrade(
            date=date,
            game_id=game["id"],
            matchup=matchup,
            side=sig.side,
            kalshi_entry=kalshi_entry,
            kalshi_close=kalshi_close,
            sharp_prob=sharp_prob,
            edge=round(edge, 4),
            size=size,
            kelly=round(kelly_adj, 4),
            won=won,
            pnl=pnl,
            clv=clv,
        ))

        current_bankroll += pnl
        daily_pnl += pnl

        if current_bankroll <= 0:
            break

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Backtest against real Kalshi DB data")
    p.add_argument(
        "--strategy", choices=["sharp", "random", "sentiment"], default="sharp",
        help="Strategy to run (default: sharp)",
    )
    p.add_argument("--edge", type=float, default=0.03,
                   help="Min edge threshold (default: 0.03)")
    p.add_argument("--trade-prob", type=float, default=0.30,
                   help="Trade probability for RandomStrategy (default: 0.30)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for RandomStrategy")
    p.add_argument("--training-data", default="data/processed/training_data_with_sentiment.csv",
                   help="Path to training CSV for sentiment strategy")
    p.add_argument("--bankroll", type=float, default=1000.0,
                   help="Starting bankroll (default: $1000)")
    p.add_argument("--db", default=DATABASE_PATH,
                   help="Path to trading.db")
    p.add_argument("--verbose", action="store_true",
                   help="Show monthly breakdown")
    p.add_argument("--trades", action="store_true",
                   help="Print individual trade log")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.strategy == "sharp":
        strategy = SharpVsKalshiStrategy(min_edge=args.edge)
    elif args.strategy == "sentiment":
        from pathlib import Path
        import pandas as pd
        from backend.strategies.user_sentiment import UserSentimentStrategy
        td_path = Path(args.training_data)
        if not td_path.exists():
            print(f"Training data not found: {td_path}")
            print("Run the pipeline first:")
            print("  python -m src.sentiment")
            print("  python -m src.create_training_data")
            print("  python -m src.train_sentiment_model")
            return
        training_df = pd.read_csv(td_path)
        strategy = UserSentimentStrategy(min_edge=args.edge, training_df=training_df)
    else:
        strategy = RandomStrategy(trade_prob=args.trade_prob, seed=args.seed)

    print(f"Strategy:   {strategy}")
    print(f"Bankroll:   ${args.bankroll:.0f}")
    print(f"DB:         {args.db}")
    print()

    result = run_backtest(strategy, bankroll=args.bankroll, db_path=args.db)
    print(result.summary(verbose=args.verbose))

    if args.trades and result.trades:
        print("\nTrade log:")
        print(f"  {'Date':<12} {'Matchup':<35} {'Side':<5} "
              f"{'Entry':>6} {'Sharp':>6} {'Edge':>6} {'Size':>7} {'PnL':>8} {'CLV':>7}")
        for t in result.trades:
            sharp_str = f"{t.sharp_prob:.3f}" if t.sharp_prob else "  n/a"
            edge_str = f"{t.edge:+.3f}" if t.edge else "  n/a"
            print(
                f"  {t.date:<12} {t.matchup:<35} {t.side:<5} "
                f"{t.kalshi_entry:>6.3f} {sharp_str:>6} {edge_str:>6} "
                f"${t.size:>6.2f} {t.pnl:>+8.4f} {t.clv:>+7.4f}"
            )


if __name__ == "__main__":
    main()
