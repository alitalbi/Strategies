"""
Data loading utilities with validation
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class FuturesDataLoader:
    """Handle loading and initial validation of futures data"""

    def __init__(self, config):
        self.config = config

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load futures data from CSV

        Args:
            filepath: Path to CSV file

        Returns:
            Raw dataframe
        """
        logger.info(f"Loading data from: {filepath}")

        try:
            df = pd.read_csv(filepath)
        except UnicodeDecodeError:
            # Try different encoding
            df = pd.read_csv(filepath, encoding='latin-1')

        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        return df

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names and identify key columns

        This handles different naming conventions in input data
        """
        df = df.copy()

        # Convert all columns to lowercase
        df.columns = df.columns.str.lower().str.strip()

        # Map common variations to standard names
        column_mapping = {
            # Timestamp variations
            'date': 'timestamp',
            'datetime': 'timestamp',
            'time': 'timestamp',
            'trade_time': 'timestamp',

            # Price variations
            'close': 'price',
            'last': 'price',
            'last_price': 'price',
            'settlement': 'price',
        }

        df = df.rename(columns=column_mapping)

        # Verify we have required columns
        required = ['timestamp', 'price']
        missing = [col for col in required if col not in df.columns]

        if missing:
            logger.error(f"Missing required columns: {missing}")
            logger.info(f"Available columns: {df.columns.tolist()}")
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("Column standardization complete")
        return df

    def parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert timestamp column to datetime and handle timezone
        """
        df = df.copy()

        # Convert to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Check for failed conversions
        null_timestamps = df['timestamp'].isnull().sum()
        if null_timestamps > 0:
            logger.warning(f"Failed to parse {null_timestamps} timestamps")

        # Sort by time
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Add timezone if not present (assume UTC for futures)
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')

        logger.info(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def initial_validation(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Basic validation checks on raw data

        Returns:
            Validated dataframe and validation report
        """
        report = {
            'total_rows': len(df),
            'date_range': (df['timestamp'].min(), df['timestamp'].max()),
            'null_counts': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated(subset=['timestamp']).sum(),
            'price_range': (df['price'].min(), df['price'].max()),
        }

        logger.info("Initial validation complete")
        logger.info(f"Date range: {report['date_range'][0]} to {report['date_range'][1]}")
        logger.info(f"Price range: ${report['price_range'][0]:.2f} - ${report['price_range'][1]:.2f}")

        return df, report