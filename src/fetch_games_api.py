"""
Fetch NBA game data using nba_api.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _nba_season_str(season_end_year: int) -> str:
    start_year = season_end_year - 1
    return f"{start_year}-{str(season_end_year)[-2:]}"


def fetch_games_nba_api(season: int = 2025, include_playoffs: bool = True) -> pd.DataFrame:
    """
    Fetch one row per NBA game with home/away teams and scores.

    Args:
        season: Season end year (e.g., 2025 -> 2024-25)
        include_playoffs: Include playoffs in addition to regular season
    """
    try:
        from nba_api.stats.endpoints import leaguegamelog
    except ImportError as exc:
        raise ImportError("nba_api is required. Install with: pip install nba_api") from exc

    season_str = _nba_season_str(season)
    print(f"Fetching NBA games for season {season_str}...")

    frames: list[pd.DataFrame] = []
    regular = leaguegamelog.LeagueGameLog(
        player_or_team_abbreviation="T",
        season=season_str,
        season_type_all_star="Regular Season",
        timeout=60,
    ).get_data_frames()[0]
    frames.append(regular)

    if include_playoffs:
        playoffs = leaguegamelog.LeagueGameLog(
            player_or_team_abbreviation="T",
            season=season_str,
            season_type_all_star="Playoffs",
            timeout=60,
        ).get_data_frames()[0]
        if not playoffs.empty:
            frames.append(playoffs)

    team_game_logs = pd.concat(frames, ignore_index=True)
    if team_game_logs.empty:
        return pd.DataFrame()

    home_rows = team_game_logs[team_game_logs["MATCHUP"].str.contains(" vs. ", regex=False, na=False)].copy()
    away_rows = team_game_logs[team_game_logs["MATCHUP"].str.contains(" @ ", regex=False, na=False)].copy()

    home_rows = home_rows[["GAME_ID", "GAME_DATE", "TEAM_NAME", "PTS"]].rename(
        columns={
            "GAME_ID": "game_id",
            "GAME_DATE": "date",
            "TEAM_NAME": "home_team",
            "PTS": "home_score",
        }
    )
    away_rows = away_rows[["GAME_ID", "TEAM_NAME", "PTS"]].rename(
        columns={
            "GAME_ID": "game_id",
            "TEAM_NAME": "away_team",
            "PTS": "away_score",
        }
    )

    games = home_rows.merge(away_rows, on="game_id", how="inner")
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games = games.dropna(subset=["date"]).sort_values(["date", "game_id"]).drop_duplicates(subset=["game_id"])
    return games.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NBA game data from nba_api")
    parser.add_argument(
        "--season",
        type=int,
        default=2025,
        help="Season end year (default: 2025 for 2024-25)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/games_api.json",
        help="Path to save raw API response",
    )
    parser.add_argument(
        "--include_playoffs",
        action="store_true",
        help="Include playoff games in addition to regular season",
    )

    args = parser.parse_args()
    output_path = Path(args.output)

    df = fetch_games_nba_api(season=args.season, include_playoffs=args.include_playoffs)
    if df.empty:
        print("No data retrieved.")
        return

    # Save JSON payload
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8")
    print(f"Saved raw API response to {output_path}")

    # Save normalized CSV
    csv_output = Path(__file__).resolve().parents[1] / "data" / "raw" / f"games_api_{args.season}.csv"
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    output_df = df.rename(
        columns={
            "away_team": "away",
            "home_team": "home",
            "away_score": "away_score",
            "home_score": "home_score",
        }
    )[["game_id", "date", "away", "home", "away_score", "home_score"]]
    output_df.to_csv(csv_output, index=False)
    print(f"\nSaved {len(output_df)} games to {csv_output}")
    print(f"Columns: {list(output_df.columns)}")


if __name__ == "__main__":
    main()
