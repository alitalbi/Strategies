"""
Data cleaning and preparation
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class FuturesDataCleaner:
    """Clean and prepare futures data for analysis"""

    def __init__(self, config):
        self.config = config
        self.cleaning_log = []

    def clean_data(self, df: pd.DataFrame, anomalies: Dict) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute full cleaning pipeline

        Args:
            df: Raw dataframe
            anomalies: Anomaly detection results

        Returns:
            Cleaned dataframe and cleaning report
        """
        logger.info("Starting data cleaning pipeline...")

        original_rows = len(df)
        df_clean = df.copy()

        # Execute cleaning steps in order
        df_clean = self._remove_duplicates(df_clean)
        df_clean = self._handle_missing_values(df_clean)
        df_clean = self._remove_impossible_values(df_clean)
        df_clean = self._handle_outliers(df_clean)
        df_clean = self._handle_price_spikes(df_clean, anomalies)
        df_clean = self._sort_and_reset(df_clean)
        df_clean = self._validate_cleaned_data(df_clean)

        # Generate cleaning report
        report = self._generate_cleaning_report(original_rows, len(df_clean))

        logger.info(
            f"Cleaning complete: {original_rows} → {len(df_clean)} rows ({len(df_clean) / original_rows * 100:.1f}% retained)")

        return df_clean, report

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate timestamps, keeping last occurrence"""
        before = len(df)

        # Keep last occurrence (assumes corrections come later)
        df = df.drop_duplicates(subset=['timestamp'], keep='last')

        removed = before - len(df)
        if removed > 0:
            self.cleaning_log.append(f"Removed {removed} duplicate timestamps")
            logger.info(f"✓ Removed {removed} duplicates")

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values appropriately"""
        before_nulls = df.isnull().sum().sum()

        # For timestamp: cannot impute, must drop
        if df['timestamp'].isnull().any():
            null_time = df['timestamp'].isnull().sum()
            df = df.dropna(subset=['timestamp'])
            self.cleaning_log.append(f"Dropped {null_time} rows with missing timestamps")
            logger.info(f"✓ Dropped {null_time} rows with missing timestamps")

        # For price: forward fill limited number of periods
        if df['price'].isnull().any():
            null_price = df['price'].isnull().sum()

            # Forward fill up to limit
            df['price'] = df['price'].fillna(method='ffill', limit=self.config.FORWARD_FILL_LIMIT)

            # If still null, drop
            still_null = df['price'].isnull().sum()
            if still_null > 0:
                df = df.dropna(subset=['price'])
                self.cleaning_log.append(f"Forward filled {null_price - still_null} prices, dropped {still_null}")
                logger.info(f"✓ Forward filled {null_price - still_null} prices, dropped remaining {still_null}")

        after_nulls = df.isnull().sum().sum()
        if after_nulls > 0:
            logger.warning(f"Warning: {after_nulls} null values remain in other columns")

        return df

    def _remove_impossible_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove values that are impossible/unrealistic"""
        before = len(df)

        # Remove non-positive prices
        df = df[df['price'] > 0]

        # Remove prices outside realistic range
        df = df[
            (df['price'] >= self.config.MIN_PRICE) &
            (df['price'] <= self.config.MAX_PRICE)
            ]

        removed = before - len(df)
        if removed > 0:
            self.cleaning_log.append(f"Removed {removed} impossible values")
            logger.info(f"✓ Removed {removed} impossible price values")

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle statistical outliers

        Strategy: Remove only extreme outliers (>4 sigma) as some may be real market events
        """
        before = len(df)

        # Calculate Z-scores
        price_mean = df['price'].mean()
        price_std = df['price'].std()
        z_scores = np.abs((df['price'] - price_mean) / price_std)

        # Keep only within threshold
        df = df[z_scores <= self.config.ZSCORE_THRESHOLD].copy()

        removed = before - len(df)
        if removed > 0:
            self.cleaning_log.append(f"Removed {removed} outliers beyond {self.config.ZSCORE_THRESHOLD} sigma")
            logger.info(f"✓ Removed {removed} outliers (>{self.config.ZSCORE_THRESHOLD} sigma)")

        return df

    def _handle_price_spikes(self, df: pd.DataFrame, anomalies: Dict) -> pd.DataFrame:
        """
        Handle extreme price spikes

        Strategy: Keep most spikes as they may be real, but flag for review
        """
        df = df.copy()

        # Calculate price changes
        df['price_change_pct'] = df['price'].pct_change() * 100

        # Flag but don't remove (these could be real FOMC reactions)
        df['potential_spike'] = np.abs(df['price_change_pct']) > self.config.MAX_SINGLE_MOVE_PCT

        spike_count = df['potential_spike'].sum()
        if spike_count > 0:
            logger.info(f"✓ Flagged {spike_count} potential spikes for review (not removed)")
            self.cleaning_log.append(
                f"Flagged {spike_count} large moves (>{self.config.MAX_SINGLE_MOVE_PCT}%) for review")

        return df

    def _sort_and_reset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort by timestamp and reset index"""
        df = df.sort_values('timestamp').reset_index(drop=True)
        logger.info("✓ Sorted data by timestamp")
        return df

    def _validate_cleaned_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final validation checks"""

        # Check no nulls in critical columns
        assert df['timestamp'].notnull().all(), "Null timestamps remain!"
        assert df['price'].notnull().all(), "Null prices remain!"

        # Check no duplicates
        assert not df['timestamp'].duplicated().any(), "Duplicate timestamps remain!"

        # Check prices are positive
        assert (df['price'] > 0).all(), "Non-positive prices remain!"

        # Check proper sorting
        assert df['timestamp'].is_monotonic_increasing, "Data not properly sorted!"

        logger.info("✓ All validation checks passed")

        return df

    def _generate_cleaning_report(self, original_rows: int, final_rows: int) -> Dict:
        """Generate comprehensive cleaning report"""

        report = {
            'original_rows': original_rows,
            'final_rows': final_rows,
            'rows_removed': original_rows - final_rows,
            'retention_rate': final_rows / original_rows * 100,
            'cleaning_steps': self.cleaning_log.copy(),
        }

        return report

    def print_report(self, report: Dict):
        """Print cleaning report"""
        print("\n" + "=" * 60)
        print("DATA CLEANING REPORT")
        print("=" * 60)
        print(f"Original rows:     {report['original_rows']:,}")
        print(f"Final rows:        {report['final_rows']:,}")
        print(f"Rows removed:      {report['rows_removed']:,}")
        print(f"Retention rate:    {report['retention_rate']:.2f}%")
        print("\nCleaning steps performed:")
        for i, step in enumerate(report['cleaning_steps'], 1):
            print(f"  {i}. {step}")
        print("=" * 60 + "\n")