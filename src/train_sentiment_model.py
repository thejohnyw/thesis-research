"""
Train a RandomForest on structured + user-sentiment features.

Evaluates two models with temporal cross-validation (no data leakage):
  1. Structured only  — baseline
  2. Structured + sentiment — does Reddit signal help?

Saves the combined model + scaler to models/user_sentiment_model.pkl.

Usage:
    python -m src.train_sentiment_model
    python -m src.train_sentiment_model --folds 5 --trees 200
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.user_features import SENTIMENT_FEATURE_COLS

DATA_PATH  = Path("data/processed/training_data_with_sentiment.csv")
MODEL_PATH = Path("models/user_sentiment_model.pkl")

# Structured feature columns as they appear in features.csv
_EXCLUDE = {"game_id", "date", "home_team", "away_team", "home_score",
            "away_score", "home_win", "cutoff_utc",
            "home_num_posts", "home_num_users", "away_num_posts", "away_num_users",
            "diff_mean_sentiment", "diff_std_sentiment",
            "diff_mean_user_std", "diff_user_entropy"}


def get_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    struct = [c for c in df.columns if c not in _EXCLUDE
              and c not in SENTIMENT_FEATURE_COLS]
    text   = [c for c in SENTIMENT_FEATURE_COLS if c in df.columns]
    return struct, text


def _make_model(n_estimators: int, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        min_samples_leaf=8,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )


def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    n_estimators: int,
    seed: int,
    label: str,
) -> dict[str, list[float]]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics: dict[str, list[float]] = {"acc": [], "auc": [], "logloss": []}

    for fold, (tr, te) in enumerate(tscv.split(X)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])

        m = _make_model(n_estimators, seed + fold)
        m.fit(Xtr, y[tr])

        prob = m.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)

        metrics["acc"].append(accuracy_score(y[te], pred))
        metrics["auc"].append(roc_auc_score(y[te], prob))
        metrics["logloss"].append(log_loss(y[te], prob))

    print(f"\n{label}")
    print(f"  Accuracy : {np.mean(metrics['acc']):.3f} ± {np.std(metrics['acc']):.3f}")
    print(f"  AUC      : {np.mean(metrics['auc']):.3f} ± {np.std(metrics['auc']):.3f}")
    print(f"  Log Loss : {np.mean(metrics['logloss']):.3f} ± {np.std(metrics['logloss']):.3f}")
    return metrics


def train_and_save(
    n_splits: int = 4,
    n_estimators: int = 150,
    seed: int = 42,
) -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found.\n"
            "Run first:\n"
            "  python -m src.sentiment\n"
            "  python -m src.create_training_data"
        )

    df = pd.read_csv(DATA_PATH).sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df)} games  ({df['date'].min()} → {df['date'].max()})")

    struct_cols, text_cols = get_columns(df)
    print(f"Structured features : {len(struct_cols)}")
    print(f"Sentiment features  : {len(text_cols)}")

    y          = df["home_win"].values
    X_struct   = df[struct_cols].fillna(0).values
    X_text     = df[text_cols].fillna(0).values
    X_combined = np.hstack([X_struct, X_text])

    print("\n" + "═" * 55)
    print("TEMPORAL CV RESULTS")
    print("═" * 55)

    metrics_struct   = evaluate(X_struct,   y, n_splits, n_estimators, seed, "Structured only")
    metrics_combined = evaluate(X_combined, y, n_splits, n_estimators, seed, "Structured + Sentiment")

    # Delta
    delta_auc = np.mean(metrics_combined["auc"]) - np.mean(metrics_struct["auc"])
    delta_acc = np.mean(metrics_combined["acc"]) - np.mean(metrics_struct["acc"])
    print(f"\nSentiment delta:  AUC {delta_auc:+.3f}   Acc {delta_acc:+.3f}")
    if delta_auc > 0.005:
        print("  → Sentiment helps — use combined model")
    elif delta_auc < -0.005:
        print("  → Sentiment hurts — use structured only (common for efficient markets)")
    else:
        print("  → No significant difference")

    # ── Feature importance (from one full-data fit) ────────────────────────────
    scaler_full = StandardScaler()
    X_full = scaler_full.fit_transform(X_combined)
    model_full = _make_model(n_estimators, seed)
    model_full.fit(X_full, y)

    all_cols = struct_cols + text_cols
    importance = sorted(
        zip(all_cols, model_full.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )

    print("\nTop 20 Features:")
    for feat, imp in importance[:20]:
        tag = "[S]" if feat in struct_cols else "[T]"
        print(f"  {tag} {feat:<42} {imp:.4f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model":       model_full,
            "scaler":      scaler_full,
            "struct_cols": struct_cols,
            "text_cols":   text_cols,
            "all_cols":    all_cols,
            "metrics":     {
                "structured":   metrics_struct,
                "combined":     metrics_combined,
            },
        }, f)

    print(f"\nModel saved → {MODEL_PATH}")
    print(f"  Features: {len(all_cols)} total  ({len(struct_cols)} struct + {len(text_cols)} sentiment)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--seed",  type=int, default=42)
    args = ap.parse_args()
    train_and_save(args.folds, args.trees, args.seed)


if __name__ == "__main__":
    main()
