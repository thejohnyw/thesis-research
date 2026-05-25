"""
Settlement engine — resolve trades and compute P&L + CLV.

CLV (Closing Line Value): measures whether you got a better price than
the final market price. Positive CLV = you consistently beat the close,
which is the strongest indicator of long-term edge.
"""
import logging
from datetime import datetime, timedelta

from backend.config import KALSHI_FEE
from backend.data.nba_outcomes import get_completed_games, resolve_outcomes
from backend.models.database import (
    get_open_trades, settle_trade, get_latest_odds,
    update_bot_state, get_bot_state, set_game_outcome,
    get_upcoming_games,
)

log = logging.getLogger(__name__)


def calculate_pnl(entry_price: float, side: str, outcome: int) -> float:
    """
    Calculate P&L for a Kalshi binary contract.

    outcome: 1 = home wins, 0 = away wins
    side: 'buy' (bet home) or 'sell' (bet away)
    Returns P&L per dollar risked.
    """
    if side == "buy":
        if outcome == 1:  # home wins, YES settles at $1
            gross_profit = 1.0 - entry_price
            return gross_profit * (1 - KALSHI_FEE)
        else:
            return -entry_price
    else:  # sell
        if outcome == 0:  # away wins, YES settles at $0
            gross_profit = entry_price
            return gross_profit * (1 - KALSHI_FEE)
        else:
            return -(1.0 - entry_price)


def calculate_clv(entry_price: float, closing_price: float, side: str) -> float:
    """
    Closing Line Value: how much better was your entry vs the close.

    Positive CLV = you consistently beat the market.
    CLV = closing_prob - entry_prob (for buys)
    CLV = entry_prob - closing_prob (for sells)
    """
    if side == "buy":
        return closing_price - entry_price
    else:
        return entry_price - closing_price


def fetch_and_set_outcomes():
    """
    Fetch real NBA outcomes and update the games table.
    Checks today and yesterday to catch games that finished late.
    Returns number of games updated.
    """
    updated = 0
    unsettled_games = get_upcoming_games()  # outcome IS NULL
    if not unsettled_games:
        return 0

    # Check today and yesterday
    dates = [
        datetime.utcnow().strftime("%Y-%m-%d"),
        (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d"),
    ]

    all_completed = []
    for date in dates:
        all_completed.extend(get_completed_games(date))

    resolved = resolve_outcomes(unsettled_games, all_completed)

    for r in resolved:
        set_game_outcome(r["game_id"], r["outcome"])
        log.info(
            f"Outcome: {r['away_team']} @ {r['home_team']} → "
            f"{'Home' if r['outcome'] else 'Away'} wins "
            f"({r['home_score']}-{r['away_score']})"
        )
        updated += 1

    return updated


def settle_completed():
    """
    Check open trades for resolved games and settle them.
    First fetches real outcomes, then calculates P&L + CLV.
    Returns list of settled trade summaries.
    """
    # Step 1: fetch real outcomes from NBA API
    n_outcomes = fetch_and_set_outcomes()
    if n_outcomes:
        log.info(f"Updated {n_outcomes} game outcomes")

    # Step 2: settle trades with known outcomes
    open_trades = get_open_trades()
    settled = []

    for trade in open_trades:
        outcome = trade["game_outcome"]
        if outcome is None:
            continue

        pnl_per_dollar = calculate_pnl(trade["entry_price"], trade["side"], outcome)
        pnl = pnl_per_dollar * trade["size"]

        # Get closing odds for CLV
        closing_odds = get_latest_odds(trade["game_id"])
        closing_kalshi = closing_odds.get("kalshi", trade["entry_price"])
        clv = calculate_clv(trade["entry_price"], closing_kalshi, trade["side"])

        settle_trade(trade["id"], pnl=round(pnl, 4), clv=round(clv, 4), outcome=outcome)

        # Update bot state
        state = get_bot_state()
        update_bot_state(
            bankroll=state["bankroll"] + pnl,
            total_pnl=state["total_pnl"] + pnl,
            daily_pnl=state["daily_pnl"] + pnl,
            total_trades=state["total_trades"] + 1,
            winning_trades=state["winning_trades"] + (1 if pnl > 0 else 0),
        )

        settled.append({
            "trade_id": trade["id"],
            "game": f"{trade['away_team']} @ {trade['home_team']}",
            "side": trade["side"],
            "entry": trade["entry_price"],
            "pnl": round(pnl, 2),
            "clv": round(clv, 4),
            "outcome": "home" if outcome else "away",
        })

    return settled
