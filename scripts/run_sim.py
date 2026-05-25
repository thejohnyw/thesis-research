"""
Live NBA paper-trading sim — RandomStrategy against real Kalshi markets.

Each cycle:
  1. collect_odds()          — pull live Kalshi + sharp book prices into DB
  2. scan_and_trade(random)  — place random paper trades on open games
  3. settle_completed()      — resolve any finished games, book P&L
  4. print dashboard         — bankroll, open positions, recent trades

Usage:
    python scripts/run_sim.py
    python scripts/run_sim.py --interval 120 --trade-prob 0.6 --seed 7
    python scripts/run_sim.py --interval 30 --no-collect   # skip odds fetch
"""
import sys
import os
import time
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timezone

from backend.models.database import (
    init_db, get_bot_state, get_recent_trades, get_open_trades,
)
from backend.core.strategy import RandomStrategy
from backend.core.signals import collect_odds, scan_and_trade
from backend.core.settlement import settle_completed

logging.basicConfig(
    level=logging.WARNING,             # suppress library noise
    format="%(levelname)s %(name)s: %(message)s",
)

# ── formatting helpers ────────────────────────────────────────────────────────

SEP  = "─" * 62
SEP2 = "═" * 62

def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def pnl_str(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}${v:.2f}"

def _banner(text: str):
    pad = (62 - len(text) - 2) // 2
    print(f"\n{'═' * pad} {text} {'═' * (62 - pad - len(text) - 2)}")

def _print_state(cycle: int, strategy: RandomStrategy):
    state  = get_bot_state()
    open_t = get_open_trades()
    recent = get_recent_trades(limit=10)

    bankroll  = state["bankroll"]   if state else 0
    total_pnl = state["total_pnl"]  if state else 0
    daily_pnl = state["daily_pnl"]  if state else 0
    total_tr  = state["total_trades"]   if state else 0
    win_tr    = state["winning_trades"] if state else 0
    win_rate  = win_tr / total_tr if total_tr else 0

    settled   = [t for t in recent if t["status"] == "settled"]
    open_list = open_t  # already from get_open_trades()

    _banner(f"CYCLE {cycle}  {ts()}")

    # ── summary row ──
    print(f"  Bankroll   ${bankroll:>9.2f}   Total PnL  {pnl_str(total_pnl):>9}")
    print(f"  Daily PnL  {pnl_str(daily_pnl):>9}   Win rate   {win_rate:>8.0%}  ({win_tr}/{total_tr})")
    print(f"  Strategy   {strategy!r}")
    print(SEP)

    # ── open positions ──
    if open_list:
        print(f"  OPEN POSITIONS ({len(open_list)})")
        print(f"  {'Matchup':<32} {'Side':<5} {'Entry':>6} {'$Size':>7}")
        for t in open_list:
            matchup = f"{t['away_team']} @ {t['home_team']}"[:31]
            print(f"  {matchup:<32} {t['side']:<5} {t['entry_price']:>5.2f}  ${t['size']:>6.2f}")
    else:
        print("  No open positions.")

    print(SEP)

    # ── recent settled trades ──
    if settled:
        print(f"  RECENT SETTLED ({len(settled)} shown)")
        print(f"  {'Matchup':<32} {'Side':<5} {'Entry':>6} {'PnL':>8} {'CLV':>7}")
        for t in settled:
            matchup = f"{t['away_team']} @ {t['home_team']}"[:31]
            pnl_v   = t["pnl"]  or 0
            clv_v   = t["clv"]  or 0
            marker  = "✓" if pnl_v > 0 else "✗"
            print(
                f"  {marker} {matchup:<30} {t['side']:<5} "
                f"{t['entry_price']:>5.2f}  {pnl_str(pnl_v):>8}  {clv_v:>+6.4f}"
            )
    else:
        print("  No settled trades yet.")

    print(SEP2)


def _run_cycle(cycle: int, strategy: RandomStrategy, do_collect: bool) -> dict:
    """One full bot cycle. Returns dict of counts."""
    result = {"collected": 0, "traded": [], "settled": []}

    # 1. Collect
    if do_collect:
        print(f"[{ts()}] Collecting live odds...", end=" ", flush=True)
        try:
            n = collect_odds()
            result["collected"] = n
            print(f"{n} games")
        except Exception as e:
            print(f"FAILED — {e}")

    # 2. Trade
    print(f"[{ts()}] Scanning for trades...", end=" ", flush=True)
    try:
        trades = scan_and_trade(strategy)
        result["traded"] = trades
        if trades:
            print(f"{len(trades)} trade(s) placed:")
            for t in trades:
                print(f"       {t['mode']} {t['side'].upper():4} {t['game']}")
                print(f"             edge={t['edge']}  size={t['size']}  kelly={t['kelly']}")
                if t.get("bpi_warning"):
                    print(f"       ⚠ BPI WARNING: {t['bpi_warning']}")
        else:
            print("no signal")
    except Exception as e:
        print(f"FAILED — {e}")

    # 3. Settle
    print(f"[{ts()}] Checking settlements...", end=" ", flush=True)
    try:
        settled = settle_completed()
        result["settled"] = settled
        if settled:
            print(f"{len(settled)} settled:")
            for s in settled:
                print(f"       {s['game']}  →  PnL {pnl_str(s['pnl'])}  CLV {s['clv']:+.4f}")
        else:
            print("none")
    except Exception as e:
        print(f"FAILED — {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Live NBA paper-trading sim")
    parser.add_argument("--strategy",    choices=["random", "sharp", "sentiment", "antibot", "orderbook"],
                        default="random", help="Strategy to use (default: random)")
    parser.add_argument("--interval",    type=int,   default=60,   help="Seconds between cycles (default: 60)")
    parser.add_argument("--trade-prob",  type=float, default=0.40, help="Trade probability for random strategy")
    parser.add_argument("--edge",        type=float, default=0.03, help="Min edge for sharp/sentiment strategy")
    parser.add_argument("--seed",        type=int,   default=None, help="Random seed for reproducibility")
    parser.add_argument("--no-collect",  action="store_true",      help="Skip odds collection (use existing DB data)")
    parser.add_argument("--cycles",      type=int,   default=None, help="Stop after N cycles (default: run forever)")
    parser.add_argument("--db",          type=str,   default=None, help="Override database path (default: from config)")
    args = parser.parse_args()

    # Override DB path before any imports that read it
    if args.db:
        import backend.config as _cfg
        _cfg.DATABASE_PATH = args.db
        import backend.models.database as _dbmod
        _dbmod.DATABASE_PATH = args.db
        import backend.core.signals as _sig
        _sig_db = __import__("backend.models.database", fromlist=["DATABASE_PATH"])

    if args.strategy == "random":
        strategy = RandomStrategy(trade_prob=args.trade_prob, seed=args.seed)
    elif args.strategy == "sharp":
        from backend.core.strategy import SharpVsKalshiStrategy
        strategy = SharpVsKalshiStrategy(min_edge=args.edge)
    elif args.strategy == "sentiment":
        from backend.strategies.user_sentiment import UserSentimentStrategy
        strategy = UserSentimentStrategy(threshold=args.edge)  # live DB mode
    elif args.strategy == "antibot":
        from backend.strategies.anti_bot import AntiBotSentimentStrategy
        strategy = AntiBotSentimentStrategy(threshold=args.edge)  # live DB mode
    elif args.strategy == "orderbook":
        from backend.strategies.order_book_anchor import OrderBookAnchorStrategy
        strategy = OrderBookAnchorStrategy(threshold=args.edge)
    else:
        strategy = RandomStrategy(trade_prob=args.trade_prob, seed=args.seed)

    print(SEP2)
    print("  NBA PAPER TRADING SIM")
    print(f"  Strategy   : {strategy!r}")
    print(f"  Mode       : PAPER (no real orders)")
    print(f"  Interval   : {args.interval}s")
    print(f"  Collect    : {'yes' if not args.no_collect else 'no (using cached DB data)'}")
    print(SEP2)

    print("Initializing database...", end=" ", flush=True)
    init_db()
    print("ok")

    cycle = 0
    try:
        while True:
            cycle += 1
            if args.cycles and cycle > args.cycles:
                print("\nCycle limit reached. Done.")
                break

            print()
            _run_cycle(cycle, strategy, do_collect=not args.no_collect)
            _print_state(cycle, strategy)

            if args.cycles and cycle >= args.cycles:
                print("\nCycle limit reached. Done.")
                break

            print(f"\n  Next cycle in {args.interval}s — Ctrl+C to stop\n")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nStopped.")
        print(SEP)
        _print_state(cycle, strategy)
        print("\nAll trades above are paper only — no real money was used.")


if __name__ == "__main__":
    main()
