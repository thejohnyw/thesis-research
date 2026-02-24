"""
Baseline 3: Fusion model for structured features + pre-game NBA text embeddings.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
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


def _normalize_team_name(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def _build_join_keys(df: pd.DataFrame, away_col: str, home_col: str, date_col: str) -> pd.DataFrame:
    keyed = df.copy()
    keyed[date_col] = pd.to_datetime(keyed[date_col], errors="coerce").dt.date
    keyed["_away_norm"] = keyed[away_col].map(_normalize_team_name)
    keyed["_home_norm"] = keyed[home_col].map(_normalize_team_name)
    return keyed


def build_text_embeddings_from_games(
    games_df: pd.DataFrame,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Ablation helper: build one text embedding vector per game from synthetic
    matchup text only (no external news).

    Returns columns:
    - game_id
    - text_emb_000 ... text_emb_NNN
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for text embeddings. "
            "Install with: pip install sentence-transformers"
        ) from exc

    required = {"game_id", "away_team", "home_team", "date"}
    missing = sorted(required - set(games_df.columns))
    if missing:
        raise ValueError(f"games_df missing required columns for text embeddings: {missing}")

    game_ids = games_df["game_id"].astype(int).tolist()
    dates = pd.to_datetime(games_df["date"], errors="coerce")
    texts = [
        f"NBA game: {away} at {home}. Date: {dt.date() if not pd.isna(dt) else 'unknown'}."
        for away, home, dt in zip(games_df["away_team"], games_df["home_team"], dates)
    ]

    if not game_ids:
        raise ValueError("No rows found for text embeddings.")

    model = SentenceTransformer(model_name, local_files_only=True)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=float)

    emb_cols = [f"text_emb_{i:03d}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    emb_df.insert(0, "game_id", game_ids)
    return emb_df


def build_text_embeddings_from_articles_jsonl(
    articles_jsonl: Path,
    games_df: Optional[pd.DataFrame] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    max_articles: int = 20,
    diagnostics_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build one text embedding per game from an articles JSONL file.
    Accepts output from fetch_google_news.py or fetch_nba_text.py (same format).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for text embeddings. "
            "Install with: pip install sentence-transformers"
        ) from exc

    if not articles_jsonl.exists():
        raise FileNotFoundError(f"Articles JSONL not found: {articles_jsonl}")

    records: list[dict] = []
    skipped_lines = 0

    with articles_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped_lines += 1
                continue

            game_id = int(record.get("game_id", -1))
            articles = record.get("articles", []) or []
            snippets: list[str] = []
            for article in articles[:max_articles]:
                body = str(article.get("body_text", "") or "").strip()
                if body:
                    snippets.append(body)
                    continue
                summary = str(article.get("summary", "") or "").strip()
                if summary:
                    snippets.append(summary)
                    continue
                title = str(article.get("title", "") or "").strip()
                domain = str(article.get("domain", "") or "").strip()
                if title and domain:
                    snippets.append(f"{title}. Source: {domain}.")
                elif title:
                    snippets.append(title)
            records.append(
                {
                    "record_game_id": game_id,
                    "away_team": str(record.get("away_team", "") or ""),
                    "home_team": str(record.get("home_team", "") or ""),
                    "game_date": pd.to_datetime(record.get("game_date"), errors="coerce"),
                    "text_article_count": len(articles),
                    "text_input": " ".join(snippets),
                }
            )

    if not records:
        raise ValueError("No rows found in articles JSONL.")

    text_df = pd.DataFrame(records)
    text_df = text_df.dropna(subset=["game_date"]).copy()
    text_df["game_date"] = pd.to_datetime(text_df["game_date"]).dt.date
    text_df = text_df.sort_values(["text_article_count", "record_game_id"], ascending=[False, True])
    text_df = text_df.drop_duplicates(subset=["record_game_id", "game_date", "away_team", "home_team"], keep="first")

    if skipped_lines:
        print(f"Warning: skipped {skipped_lines} malformed JSONL lines in {articles_jsonl}")

    model = SentenceTransformer(model_name, local_files_only=True)
    embeddings = model.encode(
        text_df["text_input"].fillna("").tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=float)

    emb_cols = [f"text_emb_{i:03d}" for i in range(embeddings.shape[1])]
    text_emb_df = pd.concat(
        [
            text_df[["record_game_id", "away_team", "home_team", "game_date", "text_article_count", "text_input"]].reset_index(
                drop=True
            ),
            pd.DataFrame(embeddings, columns=emb_cols),
        ],
        axis=1,
    )

    if games_df is None:
        emb_df = text_emb_df.rename(columns={"record_game_id": "game_id"})[["game_id", "text_article_count", *emb_cols]]
        return emb_df

    required_game_cols = {"game_id", "away_team", "home_team", "date"}
    missing = sorted(required_game_cols - set(games_df.columns))
    if missing:
        raise ValueError(f"games_df missing required columns for alignment: {missing}")

    games_keyed = _build_join_keys(games_df, away_col="away_team", home_col="home_team", date_col="date")
    games_keyed = games_keyed[["game_id", "date", "_away_norm", "_home_norm"]].copy()
    games_keyed["date"] = pd.to_datetime(games_keyed["date"]).dt.date

    text_keyed = _build_join_keys(text_emb_df, away_col="away_team", home_col="home_team", date_col="game_date")
    text_keyed = text_keyed.rename(columns={"game_date": "date"})

    # 1) direct game_id match
    direct = games_keyed.merge(
        text_keyed[["record_game_id", "text_article_count", "text_input", *emb_cols]],
        left_on="game_id",
        right_on="record_game_id",
        how="left",
    )
    direct["match_type"] = np.where(direct["record_game_id"].notna(), "game_id", "")

    # 2) fallback: team/date key for unmatched rows
    unmatched_mask = direct["record_game_id"].isna()
    if unmatched_mask.any():
        key_match = direct.loc[unmatched_mask, ["game_id", "date", "_away_norm", "_home_norm"]].merge(
            text_keyed[["record_game_id", "date", "_away_norm", "_home_norm", "text_article_count", "text_input", *emb_cols]],
            on=["date", "_away_norm", "_home_norm"],
            how="left",
        )
        key_match = key_match.sort_values(["game_id", "text_article_count"], ascending=[True, False])
        key_match = key_match.drop_duplicates(subset=["game_id"], keep="first")
        for col in ["record_game_id", "text_article_count", "text_input", *emb_cols]:
            direct.loc[unmatched_mask, col] = direct.loc[unmatched_mask, "game_id"].map(key_match.set_index("game_id")[col])
        direct.loc[unmatched_mask & direct["record_game_id"].notna(), "match_type"] = "team_date"

    matched = int(direct["record_game_id"].notna().sum())
    coverage = matched / max(1, len(direct))
    with_articles = int((direct["text_article_count"].fillna(0) > 0).sum())
    print(
        f"Article alignment coverage: matched {matched}/{len(direct)} games ({coverage:.1%}), "
        f"games with >=1 article: {with_articles}/{len(direct)}"
    )

    if diagnostics_csv is not None:
        diagnostics_csv.parent.mkdir(parents=True, exist_ok=True)
        diag_cols = ["game_id", "date", "record_game_id", "match_type", "text_article_count", "text_input"]
        diagnostics = direct[diag_cols].copy()
        diagnostics["text_preview"] = diagnostics["text_input"].fillna("").str.slice(0, 220)
        diagnostics = diagnostics.drop(columns=["text_input"])
        diagnostics.to_csv(diagnostics_csv, index=False)
        print(f"Saved alignment diagnostics -> {diagnostics_csv}")

    emb_df = direct[["game_id", "text_article_count", *emb_cols]].copy()
    emb_df["text_article_count"] = emb_df["text_article_count"].fillna(0).astype(int)
    emb_df[emb_cols] = emb_df[emb_cols].fillna(0.0)
    return emb_df


def build_fusion_matrix(
    features_df: pd.DataFrame,
    text_embeddings_df: pd.DataFrame,
    structured_dim: int = 32,
) -> pd.DataFrame:
    """
    Create fusion matrix:
    - structured embedding (PCA over structured numeric features)
    - text embedding (BERT from NBA news text)
    """
    base = features_df.copy()
    base["date"] = pd.to_datetime(base["date"])

    # Structured feature block
    exclude_cols = {"game_id", "date", "home_win", "away_team", "home_team"}
    structured_cols = [c for c in base.columns if c not in exclude_cols]
    X_struct = base[structured_cols]

    struct_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    X_struct_scaled = struct_pipe.fit_transform(X_struct)

    n_components = min(structured_dim, X_struct_scaled.shape[1], max(1, X_struct_scaled.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=42)
    X_struct_emb = pca.fit_transform(X_struct_scaled)
    struct_cols = [f"struct_emb_{i:03d}" for i in range(X_struct_emb.shape[1])]
    struct_df = pd.DataFrame(X_struct_emb, columns=struct_cols)

    fused = pd.concat(
        [base[["game_id", "date", "home_win"]].reset_index(drop=True), struct_df.reset_index(drop=True)],
        axis=1,
    )

    text_df = text_embeddings_df.copy()
    text_df = text_df.drop_duplicates(subset=["game_id"], keep="last")
    fused = fused.merge(text_df, on="game_id", how="left")

    text_cols = [c for c in fused.columns if c.startswith("text_emb_")]
    if text_cols:
        fused[text_cols] = fused[text_cols].fillna(0.0)
    if "text_article_count" in fused.columns:
        fused["text_article_count"] = fused["text_article_count"].fillna(0).astype(int)
    return fused


class FusionModel:
    """Classifier over fused structured+text embeddings."""

    def __init__(
        self,
        model_type: str = "logistic",
        random_state: int = 42,
        max_iter: int = 500,
        C: float = 1.0,
        alpha: float = 1e-4,
        hidden_layer_sizes: tuple[int, ...] = (128, 64),
    ):
        if model_type not in {"mlp", "logistic"}:
            raise ValueError("model_type must be one of: mlp, logistic")
        self.model_type = model_type
        self.random_state = random_state
        self.max_iter = max_iter
        self.C = C
        self.alpha = alpha
        self.hidden_layer_sizes = hidden_layer_sizes

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

        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )
        self.feature_names_: Optional[list[str]] = None
        self.is_fitted = False

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        exclude = {"game_id", "date", "home_win"}
        return sorted([c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FusionModel":
        mask = ~y.isna()
        X_train = X[mask].copy()
        y_train = y[mask].copy()
        if len(X_train) == 0:
            raise ValueError("No valid training rows.")
        self.feature_names_ = self.get_feature_columns(X_train)
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
    bin_edges = np.linspace(0, 1, n_bins + 1)
    error = 0.0
    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        in_bin = (y_pred_proba > lower) & (y_pred_proba <= upper)
        frac = in_bin.mean()
        if frac == 0:
            continue
        observed = y_true[in_bin].mean()
        predicted = y_pred_proba[in_bin].mean()
        error += abs(predicted - observed) * frac
    return float(error)


def _evaluate(model: FusionModel, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
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
        roc_auc = roc_auc_score(y_eval, proba)
    except ValueError:
        roc_auc = np.nan
    return {
        "accuracy": accuracy_score(y_eval, pred),
        "log_loss": log_loss(y_eval, proba),
        "mse": mean_squared_error(y_eval, proba),
        "roc_auc": roc_auc,
        "calibration_error": _compute_calibration_error(y_eval.values, proba),
        "n_samples": len(X_eval),
    }


def backtest_baseline3(
    fusion_df: pd.DataFrame,
    n_splits: int = 5,
    test_size: float = 0.2,
    model_params: Optional[dict] = None,
    min_train_games: int = 100,
) -> pd.DataFrame:
    if model_params is None:
        model_params = {}

    splitter = TimeBasedSplit(date_col="date", n_splits=n_splits, test_size=test_size, min_train_games=min_train_games)
    splits = splitter.split(fusion_df)
    results: list[dict] = []

    for split_idx, split in enumerate(splits):
        if not validate_time_split(fusion_df, split):
            print(f"Warning: Split {split_idx+1} is not time-valid; skipping.")
            continue

        X_train = fusion_df.iloc[split.train_indices].copy()
        X_test = fusion_df.iloc[split.test_indices].copy()
        y_train = X_train["home_win"]
        y_test = X_test["home_win"]

        print(f"\nSplit {split_idx+1}/{len(splits)}:")
        print(f"  Train: {len(X_train)} games ({split.train_start_date.date()} to {split.train_end_date.date()})")
        print(f"  Test:  {len(X_test)} games ({split.test_start_date.date()} to {split.test_end_date.date()})")

        model = FusionModel(**model_params)
        model.fit(X_train, y_train)
        metrics = _evaluate(model, X_test, y_test)
        metrics["split"] = split_idx + 1
        metrics["train_start"] = split.train_start_date
        metrics["train_end"] = split.train_end_date
        metrics["test_start"] = split.test_start_date
        metrics["test_end"] = split.test_end_date
        metrics["n_train"] = len(X_train)
        metrics["n_test"] = len(X_test)
        results.append(metrics)

        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Log Loss: {metrics['log_loss']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"  Calibration Error: {metrics['calibration_error']:.4f}")

    if not results:
        raise ValueError("No valid backtest results.")
    return pd.DataFrame(results)
