"""
Pre-game prediction job for the M3 live trading strategy.

Computes M3 model probabilities for all upcoming games and writes them to
data/pregame_predictions.json. Called daily at noon ET by the scheduler.

Structured features: computed from data/processed/games_api.csv (season
history) extended with the upcoming game row. Matches the training pipeline
exactly — cutoff = midnight UTC of the game's calendar date.

Sentiment features: read from trading.db reddit_posts (48h window ending at
midnight UTC of game date, same as training).

Usage:
    python -m backend.data.pregame_features
    python -m backend.data.pregame_features --date 2026-05-28
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.feature_engineering import compute_team_stats_at_game
from src.user_features import aggregate_by_user
from backend.models.database import get_conn, get_upcoming_games

log = logging.getLogger(__name__)

GAMES_CSV  = "data/processed/games_api.csv"
MODEL_PATH = "models/m3_rf.pkl"
SCALER_PATH = "models/m3_scaler.pkl"
FEAT_PATH  = "models/m3_feature_cols.json"
OUT_PATH   = "data/pregame_predictions.json"
SENT_HOURS = 48


def _load_model():
    with open(MODEL_PATH, "rb") as f:
        rf = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        sc = pickle.load(f)
    with open(FEAT_PATH) as f:
        feat_cols = json.load(f)
    return rf, sc, feat_cols


def _get_sentiment(team: str, cutoff_ts: float) -> dict:
    """Fetch scored posts from trading.db within the 48h window and aggregate."""
    window_start = int(cutoff_ts - SENT_HOURS * 3600)
    conn = get_conn()
    rows = conn.execute("""
        SELECT author, sentiment FROM reddit_posts
        WHERE team = ? AND created_utc >= ? AND created_utc <= ?
          AND sentiment IS NOT NULL
    """, (team, window_start, int(cutoff_ts))).fetchall()
    conn.close()
    posts = [{"author": r["author"] or "", "sentiment": r["sentiment"]} for r in rows]
    return aggregate_by_user(posts)


def compute_live_predictions(game_date: str | None = None) -> dict[str, float]:
    """
    Compute M3 probabilities for all upcoming games (optionally filtered by date).
    Returns {home|away|YYYY-MM-DD: model_prob}.
    """
    for path in (MODEL_PATH, SCALER_PATH, FEAT_PATH):
        if not os.path.exists(path):
            log.error(f"Missing model file: {path}. Run: python scripts/train_m3_model.py")
            return {}

    rf, sc, feat_cols = _load_model()

    history = pd.read_csv(GAMES_CSV)
    history["date"] = pd.to_datetime(history["date"]).dt.strftime("%Y-%m-%d")

    upcoming = get_upcoming_games()
    if game_date:
        upcoming = [g for g in upcoming if str(g.get("scheduled_time", ""))[:10] == game_date]

    if not upcoming:
        log.info("No upcoming games found.")
        return {}

    predictions: dict[str, float] = {}

    for game in upcoming:
        home  = game["home_team"]
        away  = game["away_team"]
        sched = str(game.get("scheduled_time", ""))[:10]
        if not sched:
            continue

        # Midnight UTC cutoff — matches _game_utc() in training pipeline
        cutoff_ts = pd.Timestamp(sched, tz="UTC").timestamp()

        # Append upcoming game to history so feature engineering sees it
        new_row = pd.DataFrame([{
            "game_id":    game["id"],
            "date":       sched,
            "home_team":  home,
            "away_team":  away,
            "home_score": np.nan,
            "away_score": np.nan,
            "home_win":   np.nan,
            "cutoff_utc": sched,
        }])
        games_df = pd.concat([history, new_row], ignore_index=True)
        games_df["date"] = pd.to_datetime(games_df["date"])
        games_df = games_df.sort_values(["date", "game_id"]).reset_index(drop=True)

        idxs = games_df.index[
            (games_df["home_team"] == home) &
            (games_df["away_team"] == away) &
            (games_df["date"] == pd.Timestamp(sched))
        ].tolist()
        if not idxs:
            log.warning(f"Could not locate {away}@{home} in games_df after concat")
            continue
        game_idx = idxs[-1]

        struct_feats = compute_team_stats_at_game(games_df, game_idx)

        home_sent = _get_sentiment(home, cutoff_ts)
        away_sent = _get_sentiment(away, cutoff_ts)

        sent_feats: dict[str, float] = {}
        for k, v in home_sent.items():
            sent_feats[f"home_{k}"] = v
        for k, v in away_sent.items():
            sent_feats[f"away_{k}"] = v
        for k in ["mean_sentiment", "std_sentiment", "mean_user_std", "user_entropy"]:
            sent_feats[f"diff_{k}"] = home_sent.get(k, 0.0) - away_sent.get(k, 0.0)

        all_feats = {**struct_feats, **sent_feats}
        x = np.array([float(all_feats.get(c, 0.0) or 0.0) for c in feat_cols]).reshape(1, -1)
        prob = float(rf.predict_proba(sc.transform(x))[0, 1])

        key = f"{home}|{away}|{sched}"
        predictions[key] = round(prob, 4)
        log.info(
            f"  {away[:3]}@{home[:3]}  {sched}  model={prob:.3f}  "
            f"home_sent={home_sent.get('mean_sentiment', 0):.3f}  "
            f"away_sent={away_sent.get('mean_sentiment', 0):.3f}  "
            f"home_posts={int(home_sent.get('num_posts', 0))}  "
            f"away_posts={int(away_sent.get('num_posts', 0))}"
        )

    return predictions


def run_and_save(game_date: str | None = None) -> None:
    preds = compute_live_predictions(game_date)
    if not preds:
        log.warning("No predictions generated.")
        return

    existing: dict = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            existing = json.load(f)
    existing.update(preds)

    with open(OUT_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    log.info(f"Saved {len(preds)} predictions → {OUT_PATH}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="Game date YYYY-MM-DD (default: all upcoming)")
    args = ap.parse_args()
    run_and_save(args.date)
