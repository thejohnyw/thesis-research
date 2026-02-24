"""
Baseline 2: Text-only model for NBA game prediction.

Uses only text embedding features (no structured stats) and evaluates with
time-based backtesting splits.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backtest import TimeBasedSplit, validate_time_split


def build_text_only_matrix(
    features_df: pd.DataFrame,
    text_embeddings_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a text-only matrix with one row per game.
    """
    base = features_df[["game_id", "date", "home_win"]].copy()
    base["date"] = pd.to_datetime(base["date"])

    text_df = text_embeddings_df.copy().drop_duplicates(subset=["game_id"], keep="last")
    out = base.merge(text_df, on="game_id", how="left")

    text_cols = [c for c in out.columns if c.startswith("text_emb_")]
    if not text_cols:
        raise ValueError("No text embedding columns found (expected text_emb_*).")

    out[text_cols] = out[text_cols].fillna(0.0)
    if "text_article_count" in out.columns:
        out["text_article_count"] = out["text_article_count"].fillna(0).astype(int)
    return out


class TextOnlyModel:
    def __init__(
        self,
        model_type: str = "logistic",
        random_state: int = 42,
        max_iter: int = 700,
        C: float = 0.3,
        alpha: float = 1e-4,
        hidden_layer_sizes: tuple[int, ...] = (128, 64),
        pca_components: int | None = None,
    ):
        if model_type not in {"logistic", "mlp"}:
            raise ValueError("model_type must be one of: logistic, mlp")
        self.model_type = model_type
        self.pca_components = pca_components
        if model_type == "logistic":
            classifier = LogisticRegression(
                random_state=random_state,
                C=C,
                max_iter=max_iter,
                class_weight="balanced",
            )
        else:
            classifier = MLPClassifier(
                random_state=random_state,
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=alpha,
                max_iter=max_iter,
                early_stopping=True,
                n_iter_no_change=10,
            )

        steps: list[tuple[str, object]] = [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
        if pca_components is not None and pca_components > 0:
            steps.append(("pca", PCA(n_components=pca_components, random_state=random_state)))
        steps.append(("classifier", classifier))
        self.pipeline = Pipeline(steps)
        self.feature_names_: Optional[list[str]] = None
        self.is_fitted = False

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        return sorted([c for c in df.columns if c.startswith("text_emb_") and pd.api.types.is_numeric_dtype(df[c])])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TextOnlyModel":
        mask = ~y.isna()
        X_train = X[mask].copy()
        y_train = y[mask].copy()
        self.feature_names_ = self.get_feature_columns(X_train)
        if "pca" in self.pipeline.named_steps:
            max_components = min(len(X_train), len(self.feature_names_))
            requested = int(self.pipeline.named_steps["pca"].n_components)
            if requested > max_components:
                self.pipeline.set_params(pca=PCA(n_components=max_components, random_state=42))
        self.pipeline.fit(X_train[self.feature_names_].values, y_train.values)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.feature_names_ is None:
            raise ValueError("Model is not fitted.")
        return self.pipeline.predict_proba(X[self.feature_names_].values)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _compute_calibration_error(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    err = 0.0
    for i in range(n_bins):
        in_bin = (y_pred_proba > bins[i]) & (y_pred_proba <= bins[i + 1])
        frac = in_bin.mean()
        if frac == 0:
            continue
        err += abs(y_pred_proba[in_bin].mean() - y_true[in_bin].mean()) * frac
    return float(err)


def _evaluate(model: TextOnlyModel, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    mask = ~y_test.isna()
    X_eval = X_test[mask].copy()
    y_eval = y_test[mask].copy()
    if len(X_eval) == 0:
        return {
            "accuracy": np.nan,
            "log_loss": np.nan,
            "mse": np.nan,
            "roc_auc": np.nan,
            "calibration_error": np.nan,
            "n_samples": 0,
        }
    pred = model.predict(X_eval)
    proba = model.predict_proba(X_eval)[:, 1]
    try:
        auc = roc_auc_score(y_eval, proba)
    except ValueError:
        auc = np.nan
    return {
        "accuracy": accuracy_score(y_eval, pred),
        "log_loss": log_loss(y_eval, proba),
        "mse": mean_squared_error(y_eval, proba),
        "roc_auc": auc,
        "calibration_error": _compute_calibration_error(y_eval.values, proba),
        "n_samples": len(X_eval),
    }


def backtest_baseline2(
    text_df: pd.DataFrame,
    n_splits: int = 5,
    test_size: float = 0.2,
    model_params: Optional[dict] = None,
    min_train_games: int = 100,
) -> pd.DataFrame:
    if model_params is None:
        model_params = {}
    splitter = TimeBasedSplit(date_col="date", n_splits=n_splits, test_size=test_size, min_train_games=min_train_games)
    splits = splitter.split(text_df)
    rows: list[dict] = []

    for idx, split in enumerate(splits):
        if not validate_time_split(text_df, split):
            continue
        X_train = text_df.iloc[split.train_indices].copy()
        X_test = text_df.iloc[split.test_indices].copy()
        y_train = X_train["home_win"]
        y_test = X_test["home_win"]

        print(f"\nSplit {idx+1}/{len(splits)}:")
        print(f"  Train: {len(X_train)} games ({split.train_start_date.date()} to {split.train_end_date.date()})")
        print(f"  Test:  {len(X_test)} games ({split.test_start_date.date()} to {split.test_end_date.date()})")

        model = TextOnlyModel(**model_params)
        model.fit(X_train, y_train)
        metrics = _evaluate(model, X_test, y_test)
        metrics["split"] = idx + 1
        metrics["train_start"] = split.train_start_date
        metrics["train_end"] = split.train_end_date
        metrics["test_start"] = split.test_start_date
        metrics["test_end"] = split.test_end_date
        metrics["n_train"] = len(X_train)
        metrics["n_test"] = len(X_test)
        rows.append(metrics)

        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Log Loss: {metrics['log_loss']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"  Calibration Error: {metrics['calibration_error']:.4f}")

    if not rows:
        raise ValueError("No valid splits evaluated.")
    return pd.DataFrame(rows)
