"""
Train/evaluate Baseline 3 fusion model:
- Structured embedding from engineered features
- Frozen BERT text embedding from pre-game NBA news text
- Classifier: logistic regression (default); MLP supported as ablation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from baseline3 import (
    backtest_baseline3,
    build_fusion_matrix,
    build_text_embeddings_from_games,
    build_text_embeddings_from_articles_jsonl,
)


def parse_hidden_sizes(raw: str) -> tuple[int, ...]:
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    if not vals:
        return (128, 64)
    return tuple(int(v) for v in vals)


def check_text_coverage(
    features_df: pd.DataFrame,
    text_embeddings_df: pd.DataFrame,
    warn_only: bool = False,
) -> None:
    merged = features_df[["game_id", "date", "home_win"]].merge(
        text_embeddings_df[["game_id", "text_article_count"]],
        on="game_id",
        how="left",
    )
    merged["text_article_count"] = merged["text_article_count"].fillna(0)
    outcome = merged[~merged["home_win"].isna()].copy()
    with_text = outcome[outcome["text_article_count"] > 0].copy()
    if with_text.empty:
        msg = "No games with text embeddings found after join."
        if warn_only:
            print(f"Warning: {msg}")
            return
        raise ValueError(msg)
    latest_game = pd.to_datetime(outcome["date"]).max()
    latest_text = pd.to_datetime(with_text["date"]).max()
    coverage = len(with_text) / max(1, len(outcome))
    lag_days = int((latest_game - latest_text).days)
    print(
        f"Text coverage check: {len(with_text)}/{len(outcome)} games with text ({coverage:.1%}), "
        f"latest text date={latest_text.date()}, latest game date={latest_game.date()}, lag_days={lag_days}"
    )
    if lag_days > 21:
        msg = "Text embeddings are stale relative to game table. Re-run fetch_google_news.py and rebuild embeddings."
        if warn_only:
            print(f"Warning: {msg}")
        else:
            raise ValueError(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Baseline 3 (structured + text fusion)")
    parser.add_argument("--features_csv", type=str, default="data/processed/features.csv")
    parser.add_argument("--games_csv", type=str, default="data/processed/games_api.csv")
    parser.add_argument("--articles_jsonl", type=str, default="data/raw/nba_gdelt_articles_enriched.jsonl")
    parser.add_argument(
        "--text_source",
        choices=["articles", "matchup"],
        default="articles",
        help="Text source: articles (Google News JSONL) or matchup (ablation)",
    )
    parser.add_argument("--text_embeddings_csv", type=str, default="data/processed/nba_text_embeddings.csv")
    parser.add_argument("--diagnostics_csv", type=str, default="data/processed/article_alignment_preview.csv")
    parser.add_argument("--fusion_csv", type=str, default="data/processed/fusion_features.csv")
    parser.add_argument("--results_csv", type=str, default="data/processed/baseline3_results.csv")
    parser.add_argument("--rebuild_text_embeddings", action="store_true")
    parser.add_argument("--text_model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--text_batch_size", type=int, default=64)
    parser.add_argument("--text_max_articles", type=int, default=20)
    parser.add_argument("--structured_dim", type=int, default=32)
    parser.add_argument(
        "--model_type",
        choices=["logistic", "mlp"],
        default="logistic",
        help="Classifier on fused features: logistic (main baseline) or mlp (ablation)",
    )
    parser.add_argument("--hidden_sizes", type=str, default="128,64")
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--require_text", action="store_true")
    parser.add_argument("--allow_missing_text", action="store_true")
    parser.add_argument(
        "--first_x",
        type=int,
        default=0,
        help="Limit to first N games (chronological) for fast experiments. 0 = all games.",
    )
    args = parser.parse_args()
    if not args.require_text and not args.allow_missing_text:
        args.require_text = True

    project_root = Path(__file__).resolve().parents[1]
    features_path = project_root / args.features_csv
    games_path = project_root / args.games_csv
    articles_path = project_root / args.articles_jsonl

    # Adjust output paths when using first_x
    suffix = f"_first{args.first_x}" if args.first_x > 0 else ""
    text_emb_path = project_root / args.text_embeddings_csv.replace(".csv", f"{suffix}.csv")
    diag_path = project_root / args.diagnostics_csv.replace(".csv", f"{suffix}.csv")
    fusion_path = project_root / args.fusion_csv.replace(".csv", f"{suffix}.csv")
    results_path = project_root / args.results_csv.replace(".csv", f"{suffix}.csv")

    if not features_path.exists():
        raise FileNotFoundError(f"Missing features CSV: {features_path}")
    if not games_path.exists():
        raise FileNotFoundError(f"Missing games CSV: {games_path}")

    print(f"Loading structured features from {features_path}")
    features_df = pd.read_csv(features_path, parse_dates=["date"])
    games_df = pd.read_csv(games_path, parse_dates=["date"])

    # Apply first_x limit (chronological)
    if args.first_x > 0:
        games_df = games_df.sort_values("date").head(args.first_x).copy()
        features_df = features_df[features_df["game_id"].isin(games_df["game_id"])].copy()
        print(f"first_x={args.first_x}: using {len(games_df)} games, {len(features_df)} feature rows")
    else:
        print(f"Structured rows: {len(features_df)}")

    if args.text_source == "matchup":
        print("Note: using matchup text source is an ablation mode.")
    if args.model_type == "mlp":
        print("Note: using MLP classifier is an ablation mode.")

    if args.rebuild_text_embeddings or not text_emb_path.exists():
        if args.text_source == "articles":
            print(f"Building text embeddings from articles JSONL: {articles_path}")
            text_embeddings_df = build_text_embeddings_from_articles_jsonl(
                articles_jsonl=articles_path,
                games_df=games_df,
                model_name=args.text_model_name,
                batch_size=args.text_batch_size,
                max_articles=args.text_max_articles,
                diagnostics_csv=diag_path,
            )
        else:
            print("Building matchup text embeddings from NBA games table")
            text_embeddings_df = build_text_embeddings_from_games(
                games_df=games_df,
                model_name=args.text_model_name,
                batch_size=args.text_batch_size,
            )
        text_emb_path.parent.mkdir(parents=True, exist_ok=True)
        text_embeddings_df.to_csv(text_emb_path, index=False)
        print(f"Saved text embeddings -> {text_emb_path}")
    else:
        print(f"Loading cached text embeddings from {text_emb_path}")
        text_embeddings_df = pd.read_csv(text_emb_path)
        overlap = int(games_df["game_id"].astype(int).isin(text_embeddings_df["game_id"].astype(int)).sum())
        if args.text_source == "articles" and overlap == 0:
            print("Cached text embeddings do not align to current games; rebuilding.")
            text_embeddings_df = build_text_embeddings_from_articles_jsonl(
                articles_jsonl=articles_path,
                games_df=games_df,
                model_name=args.text_model_name,
                batch_size=args.text_batch_size,
                max_articles=args.text_max_articles,
                diagnostics_csv=diag_path,
            )
            text_embeddings_df.to_csv(text_emb_path, index=False)
            print(f"Saved rebuilt text embeddings -> {text_emb_path}")

    check_text_coverage(features_df, text_embeddings_df, warn_only=args.first_x > 0)

    print("Building fusion matrix")
    fusion_df = build_fusion_matrix(
        features_df=features_df,
        text_embeddings_df=text_embeddings_df,
        structured_dim=args.structured_dim,
    )
    fusion_path.parent.mkdir(parents=True, exist_ok=True)
    fusion_df.to_csv(fusion_path, index=False)
    print(f"Saved fusion features -> {fusion_path}")
    print(f"Fusion shape: {fusion_df.shape}")

    fusion_with_outcomes = fusion_df[~fusion_df["home_win"].isna()].copy()
    if args.allow_missing_text:
        args.require_text = False
    if args.require_text and "text_article_count" in fusion_with_outcomes.columns:
        before = len(fusion_with_outcomes)
        fusion_with_outcomes = fusion_with_outcomes[fusion_with_outcomes["text_article_count"] > 0].copy()
        print(f"Filtered to games with matched text: {len(fusion_with_outcomes)}/{before}")

    min_rows = 20 if args.first_x > 0 else 100
    if len(fusion_with_outcomes) < min_rows:
        raise ValueError(f"Too few outcome rows for training: {len(fusion_with_outcomes)} (min {min_rows})")

    model_params = {
        "model_type": args.model_type,
        "max_iter": args.max_iter,
        "alpha": args.alpha,
        "C": args.C,
        "hidden_layer_sizes": parse_hidden_sizes(args.hidden_sizes),
    }

    min_train_games = max(5, len(fusion_with_outcomes) // 4) if args.first_x > 0 else 100
    print("Running time-based backtest for Baseline 3")
    results_df = backtest_baseline3(
        fusion_with_outcomes,
        n_splits=args.n_splits,
        test_size=args.test_size,
        model_params=model_params,
        min_train_games=min_train_games,
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"Saved results -> {results_path}")

    metric_cols = ["accuracy", "log_loss", "mse", "roc_auc", "calibration_error"]
    print("\nAverage metrics across splits:")
    for col in metric_cols:
        if col in results_df.columns:
            print(f"  {col}: {results_df[col].mean():.4f} +- {results_df[col].std():.4f}")


if __name__ == "__main__":
    main()
