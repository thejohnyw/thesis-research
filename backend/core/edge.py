"""
Edge finder — compare sharp book odds to Kalshi prices.

Sharp books (Pinnacle, lowvig) are the closest proxy for true probability.
Kalshi is a prediction market with thinner liquidity and retail flow.
When they diverge beyond a threshold, there's a tradeable edge.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from backend.config import SHARP_BOOKS, MIN_EDGE
from backend.models.database import get_latest_odds_with_timestamps, get_upcoming_games

log = logging.getLogger(__name__)

MAX_ODDS_AGE_MINUTES = 120  # skip odds older than 2 hours


@dataclass
class Opportunity:
    game_id: int
    home_team: str
    away_team: str
    kalshi_prob: float
    sharp_prob: float
    edge: float
    confidence: float
    side: str           # 'buy' or 'sell'
    kalshi_ticker: str | None = None


def _is_stale(timestamp_str: str) -> bool:
    """Check if an odds snapshot is too old to trade on."""
    try:
        ts = datetime.fromisoformat(timestamp_str)
        age = datetime.utcnow() - ts
        return age > timedelta(minutes=MAX_ODDS_AGE_MINUTES)
    except (ValueError, TypeError):
        return True  # can't parse = treat as stale


def get_sharp_prob(odds: dict) -> float | None:
    """
    Get sharpest available book probability.
    Average top two sharp sources if available for better estimate.
    """
    sharp_probs = []
    for book in SHARP_BOOKS:
        if book in odds:
            sharp_probs.append(odds[book]["prob"])
    if not sharp_probs:
        return None
    if len(sharp_probs) >= 2:
        return sum(sharp_probs[:2]) / 2
    return sharp_probs[0]


def calculate_confidence(odds: dict) -> float:
    """
    Confidence based on book agreement.
    More books agreeing = higher confidence.
    """
    sharp_probs = [odds[b]["prob"] for b in SHARP_BOOKS if b in odds]
    if len(sharp_probs) < 2:
        return 0.5

    spread = max(sharp_probs) - min(sharp_probs)
    # Tight spread (<1%) = high confidence, wide (>5%) = low
    return max(0.3, min(1.0, 1.0 - spread * 10))


def find_edges(min_edge: float = None) -> list[Opportunity]:
    """
    Scan all upcoming games for Kalshi vs sharp book edges.
    Skips games with stale odds data.
    """
    if min_edge is None:
        min_edge = MIN_EDGE

    opportunities = []
    games = get_upcoming_games()

    for game in games:
        odds = get_latest_odds_with_timestamps(game["id"])

        kalshi_data = odds.get("kalshi")
        if kalshi_data is None:
            continue

        # Stale odds check
        if _is_stale(kalshi_data["timestamp"]):
            log.debug(f"Skipping {game['home_team']}: Kalshi odds stale ({kalshi_data['timestamp']})")
            continue

        # Check if any sharp book data is stale
        sharp_sources = [b for b in SHARP_BOOKS if b in odds]
        fresh_sources = [b for b in sharp_sources if not _is_stale(odds[b]["timestamp"])]
        if not fresh_sources:
            if sharp_sources:
                log.debug(f"Skipping {game['home_team']}: all sharp odds stale")
            continue

        kalshi_prob = kalshi_data["prob"]
        sharp_prob = get_sharp_prob(odds)

        if sharp_prob is None:
            continue

        edge = sharp_prob - kalshi_prob
        confidence = calculate_confidence(odds)

        if abs(edge) >= min_edge:
            opportunities.append(Opportunity(
                game_id=game["id"],
                home_team=game["home_team"],
                away_team=game["away_team"],
                kalshi_prob=kalshi_prob,
                sharp_prob=sharp_prob,
                edge=edge,
                confidence=confidence,
                side="buy" if edge > 0 else "sell",
                kalshi_ticker=game.get("kalshi_ticker"),
            ))

    return sorted(opportunities, key=lambda x: abs(x.edge) * x.confidence, reverse=True)
