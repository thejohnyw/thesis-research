"""
Main training script for Baseline 1.

Orchestrates the full pipeline:
1. Load games data
2. Build features (time-honest)
3. Run backtesting evaluation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtest import TimeBasedSplit
from baseline1 import backtest_baseline1
from feature_engineering import build_feature_matrix
from prepare_games import RAW, load_games_data, prepare_games_df


def should_refresh_games(games_path: Path, mode: str) -> bool:
    if mode == "never":
        return False
    if not games_path.exists():
        return True
    if mode == "always":
        return True
    try:
        games_df = pd.read_csv(games_path, parse_dates=["date"])
        max_date = pd.to_datetime(games_df["date"], errors="coerce").max()
        if pd.isna(max_date):
            return True
        today = pd.Timestamp.utcnow().normalize()
        return max_date.normalize() < today
    except Exception:
        return True


def refresh_games_data(
    games_path: Path,
    raw_path: Path,
    use_api: bool,
    season: int,
    api_key: str | None,
    min_year: int | None,
    include_playoffs: bool,
) -> None:
    df = load_games_data(
        raw_path,
        use_api=use_api,
        season=season,
        api_key=api_key,
        include_playoffs=include_playoffs,
    )
    out = prepare_games_df(df, min_year=min_year)
    games_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(games_path, index=False)
    print(f"Refreshed games table with {len(out):,} rows -> {games_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Baseline 1 model")
    parser.add_argument(
        "--games_csv",
        type=str,
        default="data/processed/games_api.csv",
        help="Path to games CSV file",
    )
    parser.add_argument(
        "--features_csv",
        type=str,
        default="data/processed/features.csv",
        help="Path to save/load features CSV",
    )
    parser.add_argument(
        "--auto_refresh_games",
        type=str,
        choices=["never", "if_stale", "always"],
        default="always",
        help="Refresh games table before training (default: always)",
    )
    parser.add_argument(
        "--refresh_source",
        type=str,
        choices=["api", "csv"],
        default="api",
        help="Source for refreshing games data (default: api)",
    )
    parser.add_argument(
        "--regular_season_only",
        action="store_true",
        help="Exclude playoff games during NBA API refresh",
    )
    parser.add_argument(
        "--raw_games_csv",
        type=str,
        default=None,
        help="Path to raw games CSV fallback if API refresh fails",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2025,
        help="Season year for API fetching (default: 2025)",
    )
    parser.add_argument(
        "--min_year",
        type=int,
        default=2005,
        help="Filter out games before this year when building games table (default: 2005)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key (or set CBB_API_KEY / COLLEGE_BASKETBALL_API_KEY / BEARER_TOKEN)",
    )
    parser.add_argument(
        "--rebuild_features",
        action="store_true",
        help="Rebuild features even if features CSV exists",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of time-based splits for backtesting",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data for each test set",
    )
    parser.add_argument(
        "--window_games",
        type=int,
        default=10,
        help="Number of recent games for recent form features",
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        default="data/processed/baseline1_results.csv",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Regularization parameter for logistic regression",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Maximum iterations for logistic regression",
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parents[1]
    games_path = project_root / args.games_csv
    features_path = project_root / args.features_csv
    results_path = project_root / args.results_csv
    raw_games_path = Path(args.raw_games_csv) if args.raw_games_csv else RAW

    # Ensure output directories exist
    features_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 0: Refresh games if requested
    if should_refresh_games(games_path, args.auto_refresh_games):
        print("="*60)
        print("Step 0: Refreshing games data")
        print("="*60)
        use_api = args.refresh_source == "api"
        try:
            refresh_games_data(
                games_path=games_path,
                raw_path=raw_games_path,
                use_api=use_api,
                season=args.season,
                api_key=args.api_key,
                min_year=args.min_year,
                include_playoffs=not args.regular_season_only,
            )
        except Exception as e:
            print(f"Warning: games refresh failed ({e}). Using existing games file if available.")

    # Step 1: Build or load features
    if not features_path.exists() or args.rebuild_features:
        print("="*60)
        print("Step 1: Building features")
        print("="*60)
        
        if not games_path.exists():
            raise FileNotFoundError(f"Games file not found: {games_path}")
        
        print(f"Loading games from {games_path}")
        games_df = pd.read_csv(games_path, parse_dates=["date"])
        
        print(f"Building features for {len(games_df)} games...")
        print(f"Using window_games={args.window_games} for recent form")
        features_df = build_feature_matrix(games_df, window_games=args.window_games)
        
        print(f"Saving features to {features_path}")
        features_df.to_csv(features_path, index=False)
        print(f"Feature shape: {features_df.shape}")
    else:
        print("="*60)
        print("Step 1: Loading existing features")
        print("="*60)
        print(f"Loading features from {features_path}")
        features_df = pd.read_csv(features_path, parse_dates=["date"])
        print(f"Loaded {len(features_df)} games with features")
    
    # Step 2: Filter to games with outcomes
    print("\n" + "="*60)
    print("Step 2: Filtering games with outcomes")
    print("="*60)
    
    games_with_outcomes = features_df[~features_df["home_win"].isna()].copy()
    print(f"Games with outcomes: {len(games_with_outcomes)}")
    print(f"Date range: {games_with_outcomes['date'].min()} to {games_with_outcomes['date'].max()}")
    
    if len(games_with_outcomes) < 100:
        raise ValueError(f"Too few games with outcomes ({len(games_with_outcomes)}). Need at least 100.")
    
    # Step 3: Run backtesting evaluation
    print("\n" + "="*60)
    print("Step 3: Running backtesting evaluation")
    print("="*60)
    
    # Map legacy C (inverse reg strength) to MLP alpha (L2 regularization).
    alpha = 1.0 / args.C if args.C and args.C > 0 else 1e-4
    model_params = {
        'alpha': alpha,
        'max_iter': args.max_iter,
    }
    
    results_df = backtest_baseline1(
        games_with_outcomes,
        n_splits=args.n_splits,
        test_size=args.test_size,
        model_params=model_params,
    )
    
    # Step 4: Save results
    print("\n" + "="*60)
    print("Step 4: Saving results")
    print("="*60)
    
    results_df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")
    
    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)


if __name__ == "__main__":
    main()
