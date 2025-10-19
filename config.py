"""
Configuration file for ADIA Futures Analysis Pipeline
All magic numbers and parameters defined here for easy adjustment
"""

from dataclasses import dataclass
from typing import List

def get_endpoint(ticker_request):
    endpoint_data = f"https://raw.githubusercontent.com/alitalbi/storage_data_fy/refs/heads/master/{ticker_request}.csv"
    return endpoint_data
@dataclass
class DataConfig:
    """Data loading and validation parameters"""
    # File paths
    INPUT_FILE: str = "https://raw.githubusercontent.com/alitalbi/storage_data_fy/refs/heads/master/GC=F.csv"
    OUTPUT_DB: str = 'output/futures.db'
    OUTPUT_CSV: str = 'cleaned_futures.csv'

    # Expected columns (adjust based on your actual data)
    TIMESTAMP_COL: str = 'timestamp'  # or 'date', 'datetime', etc.
    PRICE_COL: str = 'price'  # or 'close', 'last', etc.

    # Data validation thresholds
    MIN_PRICE: float = 90.0  # Realistic min for 5Y futures
    MAX_PRICE: float = 140.0  # Realistic max for 5Y futures
    MIN_ROWS_REQUIRED: int = 100  # Minimum data points needed


@dataclass
class CleaningConfig:
    """Data cleaning parameters"""
    # Outlier detection
    ZSCORE_THRESHOLD: float = 4.0  # Remove beyond 4 std devs

    # Time gap detection
    MAX_GAP_MINUTES: int = 60  # Flag gaps > 1 hour

    # Price spike detection
    MAX_SINGLE_MOVE_PCT: float = 2.0  # Flag moves > 2% in one period

    # Missing data handling
    FORWARD_FILL_LIMIT: int = 5  # Max periods to forward fill


@dataclass
class FeatureConfig:
    """Feature engineering parameters"""
    # Moving averages
    MA_WINDOWS: List[int] = None

    # Volatility windows
    VOL_WINDOWS: List[int] = None

    # RSI period
    RSI_PERIOD: int = 14

    def __post_init__(self):
        if self.MA_WINDOWS is None:
            self.MA_WINDOWS = [5, 20, 50]
        if self.VOL_WINDOWS is None:
            self.VOL_WINDOWS = [10, 20, 30]


@dataclass
class SignalConfig:
    """Trading signal parameters"""
    # Mean reversion signal
    LOOKBACK_PERIOD: int = 20
    ENTRY_ZSCORE: float = 1.5
    EXIT_ZSCORE: float = 0.5

    # Position sizing
    MAX_POSITION_SIZE: float = 1.0


# Create instances
DATA_CONFIG = DataConfig()
CLEANING_CONFIG = CleaningConfig()
FEATURE_CONFIG = FeatureConfig()
SIGNAL_CONFIG = SignalConfig()