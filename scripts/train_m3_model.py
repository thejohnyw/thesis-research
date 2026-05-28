"""
Train the M3 RF model on all 953 training games and serialize it for live use.

Outputs:
  models/m3_rf.pkl            — fitted RandomForestClassifier
  models/m3_scaler.pkl        — fitted StandardScaler
  models/m3_feature_cols.json — ordered list of 53 feature names

Hyperparameters identical to compare_models.py (validated in OOS backtest).
Must be re-run whenever training data is updated.

Usage:
    python scripts/train_m3_model.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TRAINING_CSV = "data/processed/training_data_with_sentiment.csv"
KALSHI_CSV   = "data/kalshi_historical_prices.csv"
OUT_DIR      = "models"
SEED         = 42

_DROP = {
    "game_id", "date", "home_team", "away_team",
    "home_score", "away_score", "home_win", "cutoff_utc",
    "home_num_posts", "home_num_users", "away_num_posts", "away_num_users",
    "kalshi_price", "kalshi_pregame_price", "kalshi_open_price",
}
SENT_FULL = [
    "home_mean_sentiment", "home_std_sentiment",
    "home_mean_user_std",  "home_user_entropy",
    "away_mean_sentiment", "away_std_sentiment",
    "away_mean_user_std",  "away_user_entropy",
    "diff_mean_sentiment", "diff_std_sentiment",
    "diff_mean_user_std",  "diff_user_entropy",
]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(TRAINING_CSV).sort_values("date").reset_index(drop=True)

    kalshi = pd.read_csv(KALSHI_CSV)
    kalshi["date"] = kalshi["date"].astype(str).str[:10]
    kalshi["kalshi_price"] = kalshi["kalshi_pregame_price"].fillna(kalshi["kalshi_open_price"])
    merged = df.merge(
        kalshi[["date", "home_team", "away_team", "kalshi_price"]],
        on=["date", "home_team", "away_team"],
        how="inner",
    ).reset_index(drop=True)

    y = merged["home_win"].values.astype(int)

    struct_cols = [c for c in merged.columns if c not in _DROP and c not in set(SENT_FULL)]
    feat_cols   = struct_cols + SENT_FULL

    print(f"Training on {len(merged)} games  ({len(feat_cols)} features)")
    print(f"  Struct: {len(struct_cols)}  Sentiment: {len(SENT_FULL)}")

    X = merged[feat_cols].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(
        n_estimators=150, max_depth=6, min_samples_leaf=8,
        max_features="sqrt", random_state=SEED, n_jobs=-1,
    )
    rf.fit(X_scaled, y)

    rf_path   = os.path.join(OUT_DIR, "m3_rf.pkl")
    sc_path   = os.path.join(OUT_DIR, "m3_scaler.pkl")
    feat_path = os.path.join(OUT_DIR, "m3_feature_cols.json")

    with open(rf_path, "wb") as f:
        pickle.dump(rf, f)
    with open(sc_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(feat_path, "w") as f:
        json.dump(feat_cols, f, indent=2)

    print(f"\nSaved:")
    print(f"  {rf_path}")
    print(f"  {sc_path}")
    print(f"  {feat_path}")

    train_acc = (rf.predict(X_scaled) == y).mean()
    print(f"\nIn-sample accuracy : {train_acc:.3f}")
    print("OOS AUC from compare_models.py : 0.696 (M3, 4-fold walk-forward)")


if __name__ == "__main__":
    main()
