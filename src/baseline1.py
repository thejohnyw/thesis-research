"""
Baseline 1: Logistic Regression on Structured Historical Features

A simple logistic regression model trained on pre-game structured historical data:
- Team records and stats
- Recent performance metrics
- Ranking differences
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backtest import TimeBasedSplit, TimeSplit, validate_time_split


class Baseline1Model:
    """
    Baseline 1: MLP on structured features.
    """
    
    def __init__(
        self,
        random_state: int = 42,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        alpha: float = 1e-4,
        max_iter: int = 500,
    ):
        """
        Args:
            random_state: Random seed for reproducibility
            hidden_layer_sizes: MLP hidden layer sizes
            alpha: L2 regularization term for MLP
            max_iter: Maximum iterations for MLP
        """
        self.random_state = random_state
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        
        # Build pipeline: imputation -> scaling -> MLP
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier(
                random_state=random_state,
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=alpha,
                max_iter=max_iter,
                early_stopping=True,
                n_iter_no_change=10,
            )),
        ])
        
        self.feature_names_ = None
        self.is_fitted = False
    
    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Get list of feature columns to use for training.
        
        Excludes metadata columns (game_id, date, home_win) and focuses on
        structured historical features.
        """
        exclude_cols = {'game_id', 'date', 'home_win', 'away_team', 'home_team'}
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        return sorted(feature_cols)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Baseline1Model:
        """
        Train the model on feature matrix X and target y.
        
        Args:
            X: DataFrame with feature columns
            y: Series with binary target (1 = home win, 0 = away win)
            
        Returns:
            Self
        """
        # Handle missing values in target (future games)
        mask = ~y.isna()
        X_train = X[mask].copy()
        y_train = y[mask].copy()
        
        if len(X_train) == 0:
            raise ValueError("No valid training samples (all targets are NaN)")
        
        # Get feature columns
        feature_cols = self.get_feature_columns(X_train)
        self.feature_names_ = feature_cols
        
        # Extract feature matrix
        X_features = X_train[feature_cols].values
        
        # Fit pipeline
        self.pipeline.fit(X_features, y_train.values)
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for home win.
        
        Args:
            X: DataFrame with feature columns
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for [away_win, home_win]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.feature_names_ is None:
            raise ValueError("Model has no feature names. Fit the model first.")
        
        # Extract features
        X_features = X[self.feature_names_].values
        
        # Predict probabilities
        proba = self.pipeline.predict_proba(X_features)
        
        return proba
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict binary outcomes (1 = home win, 0 = away win).
        
        Args:
            X: DataFrame with feature columns
            
        Returns:
            Array of binary predictions
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


def evaluate_model(
    model: Baseline1Model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Evaluate model performance on test set.
    
    Returns dictionary with metrics:
    - accuracy: Classification accuracy
    - log_loss: Log loss (probabilistic metric)
    - mse: Mean squared error on probabilities
    - roc_auc: ROC AUC score
    - calibration_error: Mean calibration error
    """
    # Handle NaN targets (future games)
    mask = ~y_test.isna()
    X_eval = X_test[mask].copy()
    y_eval = y_test[mask].copy()
    
    if len(X_eval) == 0:
        return {
            'accuracy': np.nan,
            'log_loss': np.nan,
            'mse': np.nan,
            'roc_auc': np.nan,
            'calibration_error': np.nan,
            'n_samples': 0,
        }
    
    # Get predictions
    y_pred_binary = model.predict(X_eval)
    y_pred_proba = model.predict_proba(X_eval)[:, 1]  # Probability of home win
    
    # Compute metrics
    accuracy = accuracy_score(y_eval, y_pred_binary)
    logloss = log_loss(y_eval, y_pred_proba)
    mse = mean_squared_error(y_eval, y_pred_proba)
    
    # ROC AUC (may fail if only one class in test set)
    try:
        roc_auc = roc_auc_score(y_eval, y_pred_proba)
    except ValueError:
        roc_auc = np.nan
    
    # Calibration error (simple binning approach)
    calibration_error = compute_calibration_error(y_eval.values, y_pred_proba, n_bins=10)
    
    return {
        'accuracy': accuracy,
        'log_loss': logloss,
        'mse': mse,
        'roc_auc': roc_auc,
        'calibration_error': calibration_error,
        'n_samples': len(X_eval),
    }


def compute_calibration_error(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute calibration error using binning.
    
    Groups predictions into bins and measures how well predicted probabilities
    match observed frequencies.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    calibration_error = 0.0
    n_samples = len(y_true)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Mean predicted probability in bin
            accuracy_in_bin = y_true[in_bin].mean() if in_bin.sum() > 0 else 0.0
            avg_prob_in_bin = y_pred_proba[in_bin].mean() if in_bin.sum() > 0 else 0.0
            
            # Calibration error contribution
            calibration_error += np.abs(avg_prob_in_bin - accuracy_in_bin) * prop_in_bin
    
    return calibration_error


def backtest_baseline1(
    features_df: pd.DataFrame,
    n_splits: int = 5,
    test_size: float = 0.2,
    model_params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Run backtesting evaluation for Baseline 1.
    
    Args:
        features_df: DataFrame with features and target
        n_splits: Number of time-based splits
        test_size: Fraction of data for each test set
        model_params: Optional parameters for model (C, max_iter, etc.)
        
    Returns:
        DataFrame with evaluation metrics for each split
    """
    if model_params is None:
        model_params = {}
    
    # Create time-based splits
    splitter = TimeBasedSplit(date_col="date", n_splits=n_splits, test_size=test_size)
    splits = splitter.split(features_df)
    
    results = []
    
    for split_idx, split in enumerate(splits):
        # Validate split
        if not validate_time_split(features_df, split):
            print(f"Warning: Split {split_idx+1} does not maintain temporal ordering")
            continue
        
        # Split data
        X_train = features_df.iloc[split.train_indices].copy()
        X_test = features_df.iloc[split.test_indices].copy()
        y_train = X_train["home_win"]
        y_test = X_test["home_win"]
        
        print(f"\nSplit {split_idx+1}/{len(splits)}:")
        print(f"  Train: {len(X_train)} games ({split.train_start_date.date()} to {split.train_end_date.date()})")
        print(f"  Test:  {len(X_test)} games ({split.test_start_date.date()} to {split.test_end_date.date()})")
        
        # Train model
        model = Baseline1Model(**model_params)
        try:
            model.fit(X_train, y_train)
            
            # Evaluate
            metrics = evaluate_model(model, X_test, y_test)
            metrics['split'] = split_idx + 1
            metrics['train_start'] = split.train_start_date
            metrics['train_end'] = split.train_end_date
            metrics['test_start'] = split.test_start_date
            metrics['test_end'] = split.test_end_date
            metrics['n_train'] = len(X_train)
            metrics['n_test'] = len(X_test)
            
            results.append(metrics)
            
            # Print metrics
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Log Loss: {metrics['log_loss']:.4f}")
            print(f"  MSE: {metrics['mse']:.4f}")
            print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"  Calibration Error: {metrics['calibration_error']:.4f}")
            
        except Exception as e:
            print(f"  Error in split {split_idx+1}: {e}")
            continue
    
    if not results:
        raise ValueError("No valid splits could be evaluated")
    
    results_df = pd.DataFrame(results)
    
    # Compute average metrics
    print("\n" + "="*60)
    print("Average Metrics Across Splits:")
    print("="*60)
    metric_cols = ['accuracy', 'log_loss', 'mse', 'roc_auc', 'calibration_error']
    for col in metric_cols:
        if col in results_df.columns:
            mean_val = results_df[col].mean()
            std_val = results_df[col].std()
            print(f"  {col:20s}: {mean_val:.4f} ± {std_val:.4f}")
    
    return results_df


def main() -> None:
    """Main entry point for Baseline 1 evaluation."""
    from pathlib import Path
    
    features_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "features.csv"
    
    if not features_path.exists():
        print(f"Features file not found: {features_path}")
        print("Run feature_engineering.py first to generate features.")
        return
    
    print(f"Loading features from {features_path}")
    features_df = pd.read_csv(features_path, parse_dates=["date"])
    
    print(f"Loaded {len(features_df)} games with features")
    print(f"Date range: {features_df['date'].min()} to {features_df['date'].max()}")
    
    # Filter to games with outcomes (remove future games if any)
    games_with_outcomes = features_df[~features_df["home_win"].isna()].copy()
    print(f"Games with outcomes: {len(games_with_outcomes)}")
    
    # Run backtesting
    results_df = backtest_baseline1(
        games_with_outcomes,
        n_splits=5,
        test_size=0.2,
        model_params={'C': 1.0, 'max_iter': 1000},
    )
    
    # Save results
    results_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "baseline1_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
