"""
NBA game outcome fetching — resolves open trades by checking final scores.

Uses ESPN's free public scoreboard API (no key required).
Matches completed games to our database by team name + date.
"""
import httpx
import logging
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

# ESPN displayName → Odds API canonical name
TEAM_NAME_MAP = {
    "LA Clippers": "Los Angeles Clippers",
}


def _normalize_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def get_completed_games(date: str = None) -> list[dict]:
    """
    Fetch completed NBA games for a given date from ESPN.

    Args:
        date: YYYY-MM-DD string. Defaults to yesterday.

    Returns:
        List of dicts with home_team, away_team, home_score, away_score, home_won, date.
    """
    if not date:
        date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

    # ESPN expects YYYYMMDD format
    espn_date = date.replace("-", "")

    try:
        r = httpx.get(ESPN_SCOREBOARD, params={"dates": espn_date}, timeout=10)
        r.raise_for_status()
    except httpx.HTTPError as e:
        log.error(f"Failed to fetch ESPN scoreboard for {date}: {e}")
        return []

    games = []
    for event in r.json().get("events", []):
        comp = event["competitions"][0]
        status = comp["status"]["type"]["name"]

        if status != "STATUS_FINAL":
            continue

        teams = comp["competitors"]
        home = next(t for t in teams if t["homeAway"] == "home")
        away = next(t for t in teams if t["homeAway"] == "away")

        home_score = int(home.get("score", 0))
        away_score = int(away.get("score", 0))

        games.append({
            "home_team": _normalize_team(home["team"]["displayName"]),
            "away_team": _normalize_team(away["team"]["displayName"]),
            "home_score": home_score,
            "away_score": away_score,
            "home_won": home_score > away_score,
            "date": date,
        })

    log.info(f"ESPN: {len(games)} completed games for {date}")
    return games


def resolve_outcomes(games_db: list[dict], completed: list[dict]) -> list[dict]:
    """
    Match completed games to our database entries and return outcomes.

    Args:
        games_db: List of game dicts from our DB (with 'id', 'home_team', 'away_team', 'scheduled_time')
        completed: List of completed game dicts from ESPN (each has a 'date' field)

    Returns:
        List of dicts with game_id and outcome (1=home win, 0=away win)
    """
    resolved = []

    for db_game in games_db:
        # Extract the scheduled date (YYYY-MM-DD) from the DB game for date matching.
        # Accept ±1 day tolerance to handle UTC vs local tip-off time differences.
        sched_date = None
        sched_str = db_game.get("scheduled_time", "")
        if sched_str:
            try:
                sched_date = sched_str[:10]  # "YYYY-MM-DD"
            except Exception:
                pass

        for result in completed:
            if not (db_game["home_team"] == result["home_team"]
                    and db_game["away_team"] == result["away_team"]):
                continue

            # Require the ESPN result date to be within 1 day of the scheduled date.
            if sched_date:
                result_date = result.get("date", "")
                if result_date:
                    try:
                        d_sched  = datetime.strptime(sched_date,  "%Y-%m-%d")
                        d_result = datetime.strptime(result_date, "%Y-%m-%d")
                        if abs((d_result - d_sched).days) > 1:
                            continue   # wrong game (same teams, different date)
                    except ValueError:
                        pass  # can't parse — fall through and allow match

            resolved.append({
                "game_id": db_game["id"],
                "home_team": db_game["home_team"],
                "away_team": db_game["away_team"],
                "outcome": 1 if result["home_won"] else 0,
                "home_score": result["home_score"],
                "away_score": result["away_score"],
            })
            break

    return resolved
