"""
Backtesting engine with time-based train/test splits.

Ensures that training data only includes games that occurred before test games,
maintaining temporal ordering for realistic evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


@dataclass
class TimeSplit:
    """Represents a single time-based train/test split."""
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp


class TimeSeriesSplit(BaseCrossValidator):
    """
    Time-series cross-validator for time-based backtesting.
    
    Splits data chronologically, ensuring all training data occurs before test data.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[float] = None,
        gap: int = 0,
        max_train_size: Optional[int] = None,
    ):
        """
        Args:
            n_splits: Number of splits. If None, uses expanding window approach
            test_size: Fraction of data to use for test (default: 1/n_splits)
            gap: Number of samples to exclude between train and test (default: 0)
            max_train_size: Maximum number of samples in training set (for rolling window)
        """
        self.n_splits = n_splits
        self.test_size = test_size or (1.0 / n_splits if n_splits else 0.2)
        self.gap = gap
        self.max_train_size = max_train_size
    
    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.
        
        Args:
            X: Feature matrix (DataFrame or array)
            y: Target vector (optional)
            groups: Group labels (optional, not used)
            
        Yields:
            (train_indices, test_indices) tuples
        """
        if isinstance(X, pd.DataFrame):
            dates = X.index if hasattr(X.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(X.index) else X['date']
        else:
            raise ValueError("X must be a DataFrame with date information")
        
        n_samples = len(X)
        test_samples = int(n_samples * self.test_size)
        
        if self.n_splits is None:
            # Expanding window: each test set grows, training set includes all previous
            for i in range(1, n_samples // test_samples + 1):
                test_start = min(i * test_samples, n_samples)
                test_end = min(test_start + test_samples, n_samples)
                
                if test_end <= test_start:
                    break
                
                train_end = test_start - self.gap
                if train_end <= 0:
                    continue
                
                train_start = 0
                if self.max_train_size:
                    train_start = max(0, train_end - self.max_train_size)
                
                train_indices = np.arange(train_start, train_end)
                test_indices = np.arange(test_start, test_end)
                
                if len(train_indices) > 0 and len(test_indices) > 0:
                    yield train_indices, test_indices
        else:
            # Fixed number of splits with expanding or rolling window
            for i in range(self.n_splits):
                test_start = int(n_samples * (1 - (self.n_splits - i) * self.test_size))
                test_end = int(n_samples * (1 - (self.n_splits - i - 1) * self.test_size))
                
                test_start = max(0, test_start)
                test_end = min(n_samples, test_end)
                
                if test_end <= test_start:
                    continue
                
                train_end = test_start - self.gap
                if train_end <= 0:
                    continue
                
                train_start = 0
                if self.max_train_size:
                    train_start = max(0, train_end - self.max_train_size)
                
                train_indices = np.arange(train_start, train_end)
                test_indices = np.arange(test_start, test_end)
                
                if len(train_indices) > 0 and len(test_indices) > 0:
                    yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations."""
        return self.n_splits or 5


class TimeBasedSplit:
    """
    Simpler time-based splitter that works directly with date columns.
    """
    
    def __init__(
        self,
        date_col: str = "date",
        n_splits: int = 5,
        test_size: float = 0.2,
        min_train_games: int = 100,
    ):
        """
        Args:
            date_col: Name of the date column
            n_splits: Number of splits to generate
            test_size: Fraction of data for each test set
            min_train_games: Minimum number of games required in training set
        """
        self.date_col = date_col
        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train_games = min_train_games
    
    def split(self, df: pd.DataFrame) -> list[TimeSplit]:
        """
        Generate time-based splits from a DataFrame.
        
        Args:
            df: DataFrame with games, must have date_col column
            
        Returns:
            List of TimeSplit objects
        """
        df = df.copy()
        df = df.sort_values(self.date_col).reset_index(drop=True)
        
        if self.date_col not in df.columns:
            raise ValueError(f"DataFrame must have '{self.date_col}' column")
        
        dates = pd.to_datetime(df[self.date_col])
        n_samples = len(df)
        test_samples = int(n_samples * self.test_size)
        
        splits = []
        
        for i in range(self.n_splits):
            # Calculate split boundaries
            test_start_idx = int(n_samples * (1 - (self.n_splits - i) * self.test_size))
            test_end_idx = int(n_samples * (1 - (self.n_splits - i - 1) * self.test_size))
            
            test_start_idx = max(0, test_start_idx)
            test_end_idx = min(n_samples, test_end_idx)
            
            if test_end_idx <= test_start_idx:
                continue
            
            train_end_idx = test_start_idx
            
            # Check minimum training size
            if train_end_idx < self.min_train_games:
                continue
            
            train_indices = np.arange(0, train_end_idx)
            test_indices = np.arange(test_start_idx, test_end_idx)
            
            train_start_date = dates.iloc[0]
            train_end_date = dates.iloc[train_end_idx - 1] if train_end_idx > 0 else dates.iloc[0]
            test_start_date = dates.iloc[test_start_idx]
            test_end_date = dates.iloc[test_end_idx - 1]
            
            split = TimeSplit(
                train_indices=train_indices,
                test_indices=test_indices,
                train_start_date=train_start_date,
                train_end_date=train_end_date,
                test_start_date=test_start_date,
                test_end_date=test_end_date,
            )
            
            splits.append(split)
        
        return splits


def validate_time_split(df: pd.DataFrame, split: TimeSplit, date_col: str = "date") -> bool:
    """
    Validate that a time split maintains temporal ordering.
    
    Returns True if all training dates are before or equal to test dates,
    and training indices are strictly before test indices.
    """
    dates = pd.to_datetime(df[date_col])
    
    train_dates = dates.iloc[split.train_indices]
    test_dates = dates.iloc[split.test_indices]
    
    if len(train_dates) == 0 or len(test_dates) == 0:
        return False
    
    # Check that max training index is less than min test index
    # This ensures temporal ordering even if dates are the same
    max_train_idx = split.train_indices.max() if len(split.train_indices) > 0 else -1
    min_test_idx = split.test_indices.min() if len(split.test_indices) > 0 else len(df)
    
    # Also check dates as a secondary check
    max_train_date = train_dates.max()
    min_test_date = test_dates.min()
    
    return max_train_idx < min_test_idx and max_train_date <= min_test_date


def main() -> None:
    """Example usage of backtesting splits."""
    from pathlib import Path
    
    features_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "features.csv"
    
    if not features_path.exists():
        print(f"Features file not found: {features_path}")
        print("Run feature_engineering.py first to generate features.")
        return
    
    df = pd.read_csv(features_path, parse_dates=["date"])
    
    print(f"Loaded {len(df)} games with features")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    splitter = TimeBasedSplit(date_col="date", n_splits=5, test_size=0.2)
    splits = splitter.split(df)
    
    print(f"\nGenerated {len(splits)} time-based splits:")
    for i, split in enumerate(splits):
        print(f"\nSplit {i+1}:")
        print(f"  Train: {len(split.train_indices)} games ({split.train_start_date.date()} to {split.train_end_date.date()})")
        print(f"  Test:  {len(split.test_indices)} games ({split.test_start_date.date()} to {split.test_end_date.date()})")
        
        is_valid = validate_time_split(df, split)
        print(f"  Valid: {is_valid}")


if __name__ == "__main__":
    main()

