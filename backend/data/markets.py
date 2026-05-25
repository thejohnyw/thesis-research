"""
Market matching — link Odds API games to Kalshi tickers.

Kalshi ticker format: KXNBAGAME-{DATE}{AWAY}{HOME}-{SIDE}
  e.g. KXNBAGAME-26APR07BOSMIL-BOS = Boston @ Milwaukee, Boston side
"""
import re
from datetime import datetime
from backend.config import TEAM_ABBREV, TEAM_TO_ABBREV

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def parse_kalshi_ticker(ticker: str) -> dict | None:
    """
    Parse KXNBAGAME-26APR07BOSMIL-BOS into components.
    The event portion encodes: date + away_abbr + home_abbr.
    """
    parts = ticker.rsplit("-", 1)
    if len(parts) != 2:
        return None
    event, side_abbr = parts
    if side_abbr not in TEAM_ABBREV:
        return None

    # Format: KXNBAGAME-{YYMONDD}{AWAY3}{HOME3}
    match = re.search(r'(\d{2})([A-Z]{3})(\d{2})([A-Z]{3})([A-Z]{3})$', event)
    if not match:
        return None

    yy, mon, dd, away_abbr, home_abbr = match.groups()
    if away_abbr not in TEAM_ABBREV or home_abbr not in TEAM_ABBREV:
        return None

    # Parse date from ticker
    game_date = None
    if mon in MONTH_MAP:
        game_date = datetime(2000 + int(yy), MONTH_MAP[mon], int(dd)).date()

    return {
        "event": event,
        "side_abbr": side_abbr,
        "away_abbr": away_abbr,
        "home_abbr": home_abbr,
        "away_team": TEAM_ABBREV[away_abbr],
        "home_team": TEAM_ABBREV[home_abbr],
        "game_date": game_date,
    }


def match_game_to_kalshi(home_team: str, away_team: str, kalshi_markets: list,
                         game_date=None):
    """
    Find Kalshi market for the home team side of a game.
    Matches on home team, away team, AND date to avoid collisions
    when the same teams play multiple games in a week.
    """
    home_abbr = TEAM_TO_ABBREV.get(home_team)
    away_abbr = TEAM_TO_ABBREV.get(away_team)
    if not home_abbr or not away_abbr:
        return None

    for m in kalshi_markets:
        parsed = parse_kalshi_ticker(m.ticker)
        if not parsed:
            continue
        # Match: home team side of the correct game
        if (parsed["home_abbr"] == home_abbr
                and parsed["away_abbr"] == away_abbr
                and parsed["side_abbr"] == home_abbr):
            # If we have a game date, enforce date match
            if game_date and parsed["game_date"]:
                if parsed["game_date"] != game_date:
                    continue
            return m
    return None
