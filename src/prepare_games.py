from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from fetch_games_api import fetch_games_nba_api

# Default paths
RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "games_api_2025.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "games_api.csv"


def clean_team(s: str) -> str:
    s2 = (s or "")
    # Normalize non-breaking spaces and other odd whitespace.
    s2 = s2.replace("\xa0", " ").replace("\u00a0", " ").replace(" ", " ")
    s2 = re.sub(r"\s+", " ", s2).strip()
    # Some rows mark neutral site games like "v   Arizona".
    s2 = re.sub(r"^v\s+", "", s2, flags=re.IGNORECASE)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


def load_games_data(
    source_path: Path,
    use_api: bool = False,
    season: int = 2025,
    api_key: str | None = None,
    include_playoffs: bool = True,
) -> pd.DataFrame:
    """
    Load games data from either CSV file or API.
    
    Args:
        source_path: Path to CSV file (if use_api=False)
        use_api: If True, fetch from API instead of CSV
        season: Season end-year for API fetching (e.g., 2025 -> 2024-25)
        include_playoffs: Include playoff games for NBA API mode
        
    Returns:
        DataFrame with game data
    """
    if use_api:
        try:
            print(f"Fetching NBA games for season {season}...")
            df = fetch_games_nba_api(season=season, include_playoffs=include_playoffs)
            if df.empty:
                print("API fetch returned no data. Falling back to CSV...")
                df = pd.read_csv(source_path)
        except Exception as e:
            print(f"API fetch failed: {e}")
            print("Falling back to CSV file...")
            df = pd.read_csv(source_path)
    else:
        df = pd.read_csv(source_path)
    
    return df


def prepare_games_df(df: pd.DataFrame, min_year: int | None = 2005) -> pd.DataFrame:
    df = df.copy()

    def coalesce_column(target: str, aliases: list[str]) -> None:
        if target in df.columns:
            return
        for alias in aliases:
            if alias in df.columns:
                df[target] = df[alias]
                return

    # Standardize column names from different sources.
    coalesce_column("date", ["date", "game_date", "startDate", "start_date"])
    coalesce_column("away_team", ["away_team", "away", "awayTeam"])
    coalesce_column("home_team", ["home_team", "home", "homeTeam"])
    coalesce_column("away_score", ["away_score", "awayPoints", "away_pts", "visitor_score"])
    coalesce_column("home_score", ["home_score", "homePoints", "home_pts"])
    coalesce_column("away_rank", ["away_rank", "awayRank"])
    coalesce_column("home_rank", ["home_rank", "homeRank"])
    coalesce_column("neutral_site", ["neutral_site", "neutralSite"])

    # Normalize to tz-naive timestamps for consistent comparisons downstream.
    if "date" not in df.columns:
        raise ValueError("Missing date column (expected one of: date, game_date, startDate, start_date).")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)
    if min_year is not None:
        df = df[df["date"].dt.year >= min_year].copy()

    # Handle different column name conventions (CSV uses "away"/"home", API might use "away_team"/"home_team")
    if "away_team" in df.columns:
        df["away_team"] = df["away_team"].astype(str).map(clean_team)
    else:
        raise ValueError("Neither 'away' nor 'away_team' column found in data")

    if "home_team" in df.columns:
        df["home_team"] = df["home_team"].astype(str).map(clean_team)
    else:
        raise ValueError("Neither 'home' nor 'home_team' column found in data")

    # A simple guess: if the original home string contained a 'v', treat as neutral-site.
    neutral_source = df["home"] if "home" in df.columns else df["home_team"]
    df["neutral_site"] = neutral_source.astype(str).str.contains(r"\bv\b", regex=True)

    # Normalize scores to numeric
    if "home_score" not in df.columns or "away_score" not in df.columns:
        raise ValueError("Missing score columns (expected home_score/away_score or known aliases).")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    df["home_win"] = (df["home_score"] > df["away_score"]).astype("Int64")

    # Time-honest cutoff: midnight UTC on game day (conservative).
    # Later you can swap in real tipoff times and use tipoff - 1h.
    df["cutoff_utc"] = df["date"].dt.strftime("%Y-%m-%dT00:00:00Z")

    # API data may not provide ranks; ensure columns exist.
    if "away_rank" not in df.columns:
        df["away_rank"] = pd.NA
    if "home_rank" not in df.columns:
        df["home_rank"] = pd.NA

    keep = [
        "game_id",
        "date",
        "cutoff_utc",
        "away_team",
        "home_team",
        "away_score",
        "home_score",
        "home_win",
        "away_rank",
        "home_rank",
        "neutral_site",
    ]

    out = df[keep].sort_values(["date", "game_id"]).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare games data from CSV or API")
    parser.add_argument(
        "--use_api",
        action="store_true",
        help="Fetch data from API instead of CSV file",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to source CSV file fallback when API mode is disabled/fails",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2025,
        help="Season end-year for API fetching (default: 2025)",
    )
    parser.add_argument(
        "--regular_season_only",
        action="store_true",
        help="For NBA provider, exclude playoff games",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for processed games CSV (default: data/processed/games_api.csv)",
    )
    parser.add_argument(
        "--min_year",
        type=int,
        default=2005,
        help="Filter out games before this year (default: 2005)",
    )

    args = parser.parse_args()

    # Determine source path
    if args.source:
        raw_path = Path(args.source)
    else:
        raw_path = RAW

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = OUT

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data (from CSV or API)
    df = load_games_data(
        raw_path,
        use_api=args.use_api,
        season=args.season,
        include_playoffs=not args.regular_season_only,
    )
    out = prepare_games_df(df, min_year=args.min_year)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out):,} rows -> {out_path}")
    if args.use_api:
        print(f"  (Source: API for season {args.season})")
    else:
        print(f"  (Source: {raw_path})")


if __name__ == "__main__":
    main()
