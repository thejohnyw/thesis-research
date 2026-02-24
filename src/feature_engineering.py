"""
Feature engineering pipeline for NBA game predictions.

Computes historical team statistics (win rates, points scored/allowed, recent form)
that are available before each game's cutoff time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def compute_team_stats_at_game(games_df: pd.DataFrame, game_idx: int, window_games: int = 10) -> dict:
    """
    Compute team statistics available at the time of a specific game.
    
    Only uses games that occurred before the current game's cutoff time.
    
    Args:
        games_df: DataFrame with all games, sorted by date
        game_idx: Index of the current game in games_df
        window_games: Number of recent games to consider for recent form
        
    Returns:
        Dictionary with features for the current game
    """
    current_game = games_df.iloc[game_idx]
    cutoff_date = pd.to_datetime(current_game["cutoff_utc"])
    
    # Ensure both dates are timezone-naive for comparison
    if cutoff_date.tz is not None:
        cutoff_date = cutoff_date.tz_localize(None)
    
    # Get all games before this cutoff
    past_games = games_df[games_df["date"] < cutoff_date].copy()
    
    if len(past_games) == 0:
        # First game of season - return default values
        return _default_features()
    
    away_team = current_game["away_team"]
    home_team = current_game["home_team"]
    
    features = {}
    
    # Home team features
    home_stats = _compute_team_stats(past_games, home_team, cutoff_date, window_games)
    for key, val in home_stats.items():
        features[f"home_{key}"] = val
    
    # Away team features
    away_stats = _compute_team_stats(past_games, away_team, cutoff_date, window_games)
    for key, val in away_stats.items():
        features[f"away_{key}"] = val
    
    # Relative features (home - away differences)
    features["win_rate_diff"] = home_stats["win_rate"] - away_stats["win_rate"]
    features["points_scored_diff"] = home_stats["points_scored_avg"] - away_stats["points_scored_avg"]
    features["points_allowed_diff"] = home_stats["points_allowed_avg"] - away_stats["points_allowed_avg"]
    features["point_diff_avg_diff"] = home_stats["point_diff_avg"] - away_stats["point_diff_avg"]
    features["recent_form_diff"] = home_stats["recent_win_rate"] - away_stats["recent_win_rate"]
    features["recent_point_diff_avg_diff"] = (
        home_stats["recent_point_diff_avg"] - away_stats["recent_point_diff_avg"]
    )
    features["win_streak_diff"] = home_stats["win_streak"] - away_stats["win_streak"]
    features["loss_streak_diff"] = home_stats["loss_streak"] - away_stats["loss_streak"]
    features["rest_days_diff"] = (
        home_stats["last_game_days_ago"] - away_stats["last_game_days_ago"]
    )
    
    # Ranking features
    features["home_rank"] = current_game.get("home_rank", np.nan)
    features["away_rank"] = current_game.get("away_rank", np.nan)
    features["rank_diff"] = (current_game.get("away_rank", np.nan) or np.nan) - (current_game.get("home_rank", np.nan) or np.nan)
    # Lower rank number is better, so negative means home is better ranked
    features["rank_diff"] = -features["rank_diff"] if not pd.isna(features["rank_diff"]) else np.nan
    
    # Site features
    features["neutral_site"] = 1 if current_game.get("neutral_site", False) else 0
    
    return features


def _compute_team_stats(
    past_games: pd.DataFrame,
    team: str,
    cutoff_date: pd.Timestamp,
    window_games: int,
) -> dict:
    """Compute statistics for a team from past games."""
    
    # Find all games where team played (home or away)
    team_games = past_games[
        (past_games["home_team"] == team) | (past_games["away_team"] == team)
    ].copy()
    
    if len(team_games) == 0:
        return _default_team_stats()
    
    team_games = team_games.sort_values("date", ascending=False)
    
    stats = {}
    
    # Overall win rate
    wins = 0
    for _, game in team_games.iterrows():
        if game["home_team"] == team and game["home_win"] == 1:
            wins += 1
        elif game["away_team"] == team and game["home_win"] == 0:
            wins += 1
    
    stats["win_rate"] = wins / len(team_games) if len(team_games) > 0 else 0.5
    stats["games_played"] = len(team_games)
    
    # Points scored and allowed
    points_scored = []
    points_allowed = []
    
    for _, game in team_games.iterrows():
        if game["home_team"] == team:
            points_scored.append(game["home_score"])
            points_allowed.append(game["away_score"])
        else:
            points_scored.append(game["away_score"])
            points_allowed.append(game["home_score"])
    
    stats["points_scored_avg"] = np.mean(points_scored) if points_scored else 70.0
    stats["points_allowed_avg"] = np.mean(points_allowed) if points_allowed else 70.0
    stats["point_diff_avg"] = stats["points_scored_avg"] - stats["points_allowed_avg"]
    
    # Recent form (last N games)
    recent_games = team_games.head(window_games)
    if len(recent_games) > 0:
        recent_wins = 0
        for _, game in recent_games.iterrows():
            if game["home_team"] == team and game["home_win"] == 1:
                recent_wins += 1
            elif game["away_team"] == team and game["home_win"] == 0:
                recent_wins += 1
        stats["recent_win_rate"] = recent_wins / len(recent_games)
    else:
        stats["recent_win_rate"] = stats["win_rate"]

    # Recent scoring form (last N games)
    if len(recent_games) > 0:
        recent_points_scored = []
        recent_points_allowed = []
        for _, game in recent_games.iterrows():
            if game["home_team"] == team:
                recent_points_scored.append(game["home_score"])
                recent_points_allowed.append(game["away_score"])
            else:
                recent_points_scored.append(game["away_score"])
                recent_points_allowed.append(game["home_score"])
        stats["recent_points_scored_avg"] = np.mean(recent_points_scored)
        stats["recent_points_allowed_avg"] = np.mean(recent_points_allowed)
        stats["recent_point_diff_avg"] = stats["recent_points_scored_avg"] - stats["recent_points_allowed_avg"]
    else:
        stats["recent_points_scored_avg"] = stats["points_scored_avg"]
        stats["recent_points_allowed_avg"] = stats["points_allowed_avg"]
        stats["recent_point_diff_avg"] = stats["point_diff_avg"]
    
    # Home win rate vs away win rate
    home_games = team_games[team_games["home_team"] == team]
    away_games = team_games[team_games["away_team"] == team]
    
    home_wins = home_games["home_win"].sum() if len(home_games) > 0 else 0
    home_win_rate = home_wins / len(home_games) if len(home_games) > 0 else 0.5
    
    away_wins = (away_games["home_win"] == 0).sum() if len(away_games) > 0 else 0
    away_win_rate = away_wins / len(away_games) if len(away_games) > 0 else 0.5
    
    stats["home_win_rate"] = home_win_rate
    stats["away_win_rate"] = away_win_rate

    # Streaks (consecutive wins/losses, most recent backwards)
    win_streak = 0
    loss_streak = 0
    for _, game in team_games.iterrows():
        is_win = (
            (game["home_team"] == team and game["home_win"] == 1)
            or (game["away_team"] == team and game["home_win"] == 0)
        )
        if is_win:
            if loss_streak > 0:
                break
            win_streak += 1
        else:
            if win_streak > 0:
                break
            loss_streak += 1
    stats["win_streak"] = win_streak
    stats["loss_streak"] = loss_streak

    # Rest days since last game
    last_game_date = pd.to_datetime(team_games["date"].iloc[0])
    rest_days = (cutoff_date - last_game_date).days
    stats["last_game_days_ago"] = max(rest_days, 0)
    
    return stats


def _default_team_stats() -> dict:
    """Return default team stats for teams with no prior games."""
    return {
        "win_rate": 0.5,
        "games_played": 0,
        "points_scored_avg": 70.0,
        "points_allowed_avg": 70.0,
        "point_diff_avg": 0.0,
        "recent_win_rate": 0.5,
        "recent_points_scored_avg": 70.0,
        "recent_points_allowed_avg": 70.0,
        "recent_point_diff_avg": 0.0,
        "home_win_rate": 0.5,
        "away_win_rate": 0.5,
        "win_streak": 0,
        "loss_streak": 0,
        "last_game_days_ago": 7,
    }


def _default_features() -> dict:
    """Return default features when no past games exist."""
    default_team = _default_team_stats()
    return {
        f"home_{k}": v for k, v in default_team.items()
    } | {
        f"away_{k}": v for k, v in default_team.items()
    } | {
        "win_rate_diff": 0.0,
        "points_scored_diff": 0.0,
        "points_allowed_diff": 0.0,
        "point_diff_avg_diff": 0.0,
        "recent_form_diff": 0.0,
        "recent_point_diff_avg_diff": 0.0,
        "win_streak_diff": 0.0,
        "loss_streak_diff": 0.0,
        "rest_days_diff": 0.0,
        "home_rank": np.nan,
        "away_rank": np.nan,
        "rank_diff": np.nan,
        "neutral_site": 0,
    }


def build_feature_matrix(games_df: pd.DataFrame, window_games: int = 10) -> pd.DataFrame:
    """
    Build feature matrix for all games using time-honest feature engineering.
    
    For each game, features are computed using only data available before
    that game's cutoff time.
    
    Args:
        games_df: DataFrame with game data, must have 'date', 'cutoff_utc' columns
        window_games: Number of recent games to consider for recent form
        
    Returns:
        DataFrame with features for each game
    """
    games_df = games_df.copy()
    games_df["date"] = pd.to_datetime(games_df["date"])
    games_df = games_df.sort_values(["date", "game_id"]).reset_index(drop=True)
    
    # Ensure home_win is available for feature engineering (but may be NaN for future games)
    if "home_win" not in games_df.columns:
        games_df["home_win"] = np.where(
            games_df["home_score"] > games_df["away_score"], 1,
            np.where(games_df["home_score"] < games_df["away_score"], 0, np.nan)
        )
    
    feature_rows = []
    
    for idx in range(len(games_df)):
        features = compute_team_stats_at_game(games_df, idx, window_games)
        features["game_id"] = games_df.iloc[idx]["game_id"]
        features["date"] = games_df.iloc[idx]["date"]
        if "home_win" in games_df.columns:
            features["home_win"] = games_df.iloc[idx]["home_win"]
        feature_rows.append(features)
    
    features_df = pd.DataFrame(feature_rows)
    
    # Reorder columns for readability
    feature_cols = [c for c in features_df.columns if c not in ["game_id", "date", "home_win"]]
    ordered_cols = ["game_id", "date", "home_win"] + sorted(feature_cols)
    ordered_cols = [c for c in ordered_cols if c in features_df.columns]
    
    return features_df[ordered_cols]


def main() -> None:
    games_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "games_api.csv"
    output_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "features.csv"
    
    print(f"Loading games from {games_path}")
    games_df = pd.read_csv(games_path, parse_dates=["date"])
    
    print(f"Building features for {len(games_df)} games...")
    features_df = build_feature_matrix(games_df, window_games=10)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)
    print(f"Wrote features to {output_path}")
    print(f"\nFeature shape: {features_df.shape}")
    print(f"\nFeature columns:")
    print(features_df.columns.tolist())
    print(f"\nFirst few rows:")
    print(features_df.head())


if __name__ == "__main__":
    main()
