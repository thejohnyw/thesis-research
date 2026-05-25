"""
Build the combined training dataset: structured features + user sentiment features.

Joins:
  data/processed/games_api.csv      — game_id, date, teams, outcome
  data/processed/features.csv       — 41 structured pre-game features
  data/processed/reddit_with_sentiment.jsonl  — Reddit posts with sentiment

Output:
  data/processed/training_data_with_sentiment.csv

The game timestamp used for the 48-hour Reddit window comes from
`cutoff_utc` in games_api.csv (the time data was collected, i.e., ~game start).
Falls back to midnight UTC of the game date when cutoff_utc is absent.

Usage:
    python -m src.create_training_data
    python -m src.create_training_data --hours 72 --min-posts 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.user_features import (
    load_posts, build_index, get_game_sentiment_features,
    SENTIMENT_FEATURE_COLS,
)

GAMES_PATH    = Path("data/processed/games_api.csv")
FEATURES_PATH = Path("data/processed/features.csv")
POSTS_PATH    = Path("data/processed/reddit_with_sentiment.jsonl")
OUTPUT_PATH   = Path("data/processed/training_data_with_sentiment.csv")


def _game_utc(row: pd.Series) -> float:
    """Unix timestamp for game date (midnight UTC). Uses 'date' column."""
    from datetime import datetime, timezone
    for col in ("date", "cutoff_utc"):
        val = row.get(col)
        if val and pd.notna(val):
            try:
                dt = datetime.strptime(str(val)[:10], "%Y-%m-%d")
                return dt.replace(tzinfo=timezone.utc).timestamp()
            except Exception:
                continue
    return 0.0


def build_dataset(hours: int = 48, min_posts: int = 0) -> pd.DataFrame:
    # ── Load data ──────────────────────────────────────────────────────────────
    games    = pd.read_csv(GAMES_PATH)
    features = pd.read_csv(FEATURES_PATH)

    if not POSTS_PATH.exists():
        raise FileNotFoundError(
            f"{POSTS_PATH} not found.\n"
            "Run first:  python -m src.sentiment"
        )
    posts = load_posts(POSTS_PATH)
    index = build_index(posts)

    print(f"Games:    {len(games)}")
    print(f"Features: {len(features)}")
    print(f"Posts:    {len(posts)} across {len(index)} subreddits")

    # Structured feature columns (everything except game_id, date, home_win)
    struct_cols = [c for c in features.columns if c not in ("game_id", "date", "home_win")]

    # ── Merge games + features ─────────────────────────────────────────────────
    df = games.merge(features[["game_id"] + struct_cols], on="game_id", how="inner")
    print(f"After merge: {len(df)} games with structured features")

    # ── Add sentiment features ─────────────────────────────────────────────────
    rows = []
    skipped_no_posts = 0

    for _, row in df.iterrows():
        game_utc = _game_utc(row)
        home = row["home_team"]
        away = row["away_team"]

        sent = get_game_sentiment_features(index, home, away, game_utc, hours=hours)

        total_posts = sent.get("home_num_posts", 0) + sent.get("away_num_posts", 0)
        if total_posts < min_posts:
            skipped_no_posts += 1

        combined = row.to_dict()
        combined.update(sent)
        rows.append(combined)

    result = pd.DataFrame(rows)
    print(f"Games with 0 Reddit posts: {skipped_no_posts} (kept, features zeroed)")

    # ── Summary ────────────────────────────────────────────────────────────────
    has_posts = (result["home_num_posts"] + result["away_num_posts"]) > 0
    print(f"Games with ≥1 post in window: {has_posts.sum()} / {len(result)}")
    mean_home = result.loc[has_posts, "home_mean_sentiment"].mean()
    mean_away = result.loc[has_posts, "away_mean_sentiment"].mean()
    print(f"Mean home sentiment (when posts exist): {mean_home:+.3f}")
    print(f"Mean away sentiment (when posts exist): {mean_away:+.3f}")

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours",     type=int,   default=48,
                    help="Reddit window before game (hours)")
    ap.add_argument("--min-posts", type=int,   default=0,
                    help="Min total posts to include a game (0 = keep all)")
    ap.add_argument("--output",    default=str(OUTPUT_PATH))
    args = ap.parse_args()

    df = build_dataset(hours=args.hours, min_posts=args.min_posts)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows → {out}")
    print(f"Columns: {len(df.columns)}  "
          f"(struct={len([c for c in df.columns if c in df.columns and c not in ['game_id','date','home_team','away_team','home_score','away_score','home_win','cutoff_utc']+list(SENTIMENT_FEATURE_COLS)+['home_num_users','home_num_posts','away_num_users','away_num_posts','diff_mean_sentiment','diff_std_sentiment','diff_mean_user_std','diff_user_entropy']])}"
          f"  sentiment={len(SENTIMENT_FEATURE_COLS)})")


if __name__ == "__main__":
    main()
