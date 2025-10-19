"""
Feature engineering for futures trading signals
"""

import pandas as pd
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create trading features and signals from price data"""

    def __init__(self, config):
        self.config = config

    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features for analysis

        Args:
            df: Cleaned price dataframe

        Returns:
            Dataframe with engineered features
        """
        logger.info("Starting feature engineering...")

        df = df.copy()

        # Basic transformations
        df = self._add_returns(df)
        df = self._add_log_returns(df)

        # Moving averages
        df = self._add_moving_averages(df)

        # Volatility features
        df = self._add_volatility_features(df)

        # Momentum indicators
        df = self._add_momentum_indicators(df)

        # Technical indicators
        df = self._add_rsi(df)
        df = self._add_bollinger_bands(df)

        # Price position features
        df = self._add_price_position_features(df)

        # Time-based features
        df = self._add_time_features(df)

        # Trading signals
        df = self._generate_signals(df)

        # Drop NaN rows created by rolling windows
        initial_rows = len(df)
        df = df.dropna()
        dropped = initial_rows - len(df)

        logger.info(
            f"Feature engineering complete. Created {len([c for c in df.columns if c not in ['timestamp', 'price']])} features")
        logger.info(f"Dropped {dropped} rows due to rolling window initialization")

        return df

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate simple returns"""
        df['returns'] = df['price'].pct_change()
        return df

    def _add_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate log returns (better for statistical properties)"""
        df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate multiple moving averages

        Rationale: MAs smooth noise and help identify trend direction
        """
        for window in self.config.MA_WINDOWS:
            df[f'ma_{window}'] = df['price'].rolling(window=window).mean()

            # Price relative to MA (normalized)
            df[f'price_to_ma_{window}'] = (df['price'] - df[f'ma_{window}']) / df[f'ma_{window}']

        logger.info(f"✓ Added {len(self.config.MA_WINDOWS)} moving averages")
        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling volatility measures

        Rationale: Vol regime affects signal reliability - mean reversion works better in low vol
        """
        for window in self.config.VOL_WINDOWS:
            # Standard deviation of returns
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()

            # Annualized volatility (assuming daily data, ~252 trading days)
            df[f'volatility_{window}_annual'] = df[f'volatility_{window}'] * np.sqrt(252)

        # Volatility of volatility (vol clustering indicator)
        df['vol_of_vol'] = df['volatility_20'].rolling(window=20).std()

        logger.info(f"✓ Added {len(self.config.VOL_WINDOWS)} volatility measures")
        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum features

        Rationale: Capture rate of price change over different timeframes
        """
        # Simple momentum (price change)
        for period in [3, 5, 10, 20]:
            df[f'momentum_{period}'] = df['price'] - df['price'].shift(period)
            df[f'momentum_{period}_pct'] = df['momentum_{period}'] / df['price'].shift(period) * 100

        # Rate of change
        df['roc_5'] = (df['price'] - df['price'].shift(5)) / df['price'].shift(5) * 100
        df['roc_20'] = (df['price'] - df['price'].shift(20)) / df['price'].shift(20) * 100

        logger.info("✓ Added momentum indicators")
        return df

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Relative Strength Index

        Rationale: Classic overbought/oversold indicator. RSI > 70 = overbought, < 30 = oversold
        """
        period = self.config.RSI_PERIOD

        # Calculate price changes
        delta = df['price'].diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # RSI-based signals
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)

        logger.info(f"✓ Added RSI ({period} period)")
        return df

    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands

        Rationale: Identify when price is at extreme relative to recent range
        """
        window = 20
        num_std = 2

        # Middle band (SMA)
        df['bb_middle'] = df['price'].rolling(window=window).mean()

        # Standard deviation
        rolling_std = df['price'].rolling(window=window).std()

        # Upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (rolling_std * num_std)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * num_std)

        # Bandwidth (volatility measure)
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # %B (price position within bands)
        df['bb_percent'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        logger.info("✓ Added Bollinger Bands")
        return df

    def _add_price_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate where price sits relative to recent ranges

        Rationale: Mean reversion trades based on extremes in range
        """
        for window in [10, 20, 50]:
            # Highest and lowest prices in window
            df[f'high_{window}'] = df['price'].rolling(window=window).max()
            df[f'low_{window}'] = df['price'].rolling(window=window).min()

            # Price position (0 = at low, 1 = at high)
            range_size = df[f'high_{window}'] - df[f'low_{window}']
            df[f'price_position_{window}'] = (df['price'] - df[f'low_{window}']) / range_size

            # Distance from high/low (in % terms)
            df[f'dist_from_high_{window}'] = (df[f'high_{window}'] - df['price']) / df['price'] * 100
            df[f'dist_from_low_{window}'] = (df['price'] - df[f'low_{window}']) / df['price'] * 100

        logger.info("✓ Added price position features")
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time-based features

        Rationale: Market behavior varies by day of week, time of month, etc.
        """
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0 = Monday
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter

        # Binary flags for specific patterns
        df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
        df['is_quarter_end'] = ((df['month'] % 3 == 0) & (df['day_of_month'] >= 28)).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)

        logger.info("✓ Added time-based features")
        return df

    def _generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate primary trading signal

        SIGNAL STRATEGY: Mean Reversion based on Z-score

        Rationale:
        - 5Y Treasury futures exhibit mean-reverting behavior over short horizons
        - Price deviations from moving average tend to correct
        - Z-score normalizes for changing volatility regimes

        Entry rules:
        - LONG when Z-score < -1.5 (price significantly below MA = oversold)
        - SHORT when Z-score > +1.5 (price significantly above MA = overbought)
        - EXIT when Z-score crosses back through 0 (returned to mean)
        """
        lookback = self.config.LOOKBACK_PERIOD

        # Calculate deviation from moving average
        df['deviation'] = df['price'] - df[f'ma_{lookback}']

        # Calculate rolling standard deviation of prices
        df['rolling_std'] = df['price'].rolling(window=lookback).std()

        # Z-score: how many standard deviations away from mean
        df['z_score'] = df['deviation'] / df['rolling_std']

        # Generate signal (-1 = short, 0 = neutral, +1 = long)
        entry_threshold = self.config.ENTRY_ZSCORE
        exit_threshold = self.config.EXIT_ZSCORE

        df['signal_raw'] = 0
        df.loc[df['z_score'] < -entry_threshold, 'signal_raw'] = 1  # Long signal
        df.loc[df['z_score'] > entry_threshold, 'signal_raw'] = -1  # Short signal

        # Signal strength (continuous version)
        df['signal_strength'] = -df['z_score'] / entry_threshold  # Inverted for mean reversion
        df['signal_strength'] = df['signal_strength'].clip(-2, 2)  # Cap at +/-2

        # Additional confirmation: only trade if volatility is not extreme
        vol_percentile = df['volatility_20'].rolling(window=100).quantile(0.75)
        df['vol_regime_ok'] = (df['volatility_20'] < vol_percentile).astype(int)

        # Final signal: only trade in normal vol regime
        df['signal'] = df['signal_raw'] * df['vol_regime_ok']

        # MA cross signal (secondary - for comparison)
        df['ma_cross_signal'] = 0
        df.loc[df[f'ma_{self.config.MA_WINDOWS[0]}'] > df[f'ma_{self.config.MA_WINDOWS[1]}'], 'ma_cross_signal'] = 1
        df.loc[df[f'ma_{self.config.MA_WINDOWS[0]}'] < df[f'ma_{self.config.MA_WINDOWS[1]}'], 'ma_cross_signal'] = -1

        # Signal change detection (for entry/exit timing)
        df['signal_change'] = df['signal'].diff()
        df['entry_signal'] = (df['signal_change'] != 0).astype(int)

        logger.info("✓ Generated trading signals")
        logger.info(f"   Signal distribution: {df['signal'].value_counts().to_dict()}")

        return df

    def get_feature_importance_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create summary of features for documentation

        Returns DataFrame with feature descriptions
        """
        features = []

        # Get all feature columns (exclude timestamp, price, and intermediate calculations)
        feature_cols = [col for col in df.columns
                        if col not in ['timestamp', 'price', 'deviation', 'rolling_std']]

        for col in feature_cols:
            features.append({
                'feature': col,
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'null_count': df[col].isnull().sum(),
            })

        return pd.DataFrame(features)