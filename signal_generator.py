"""
Signal Generation Module
Generates mean reversion, momentum, and combined signals
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Generate trading signals from features

    Signals:
    1. Mean Reversion (primary)
    2. Momentum (comparison)
    3. Combined (regression-weighted ensemble)
    """

    def __init__(self, config):
        self.config = config
        self.signals = {}

    def generate_all_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all signal types

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with signals added
        """
        logger.info("Generating trading signals...")

        df = df.copy()

        # Generate individual signals
        df = self._generate_mean_reversion_signal(df)
        df = self._generate_momentum_signal(df)

        # Combined signal will be added later after regression
        df['signal_combined'] = 0  # Placeholder

        logger.info("✓ Signal generation complete")
        return df

    def _generate_mean_reversion_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mean Reversion Signal based on Z-score

        Strategy:
        - LONG (1) when price significantly below MA (oversold)
        - SHORT (-1) when price significantly above MA (overbought)
        - NEUTRAL (0) otherwise

        Rationale: 5Y Treasury futures mean-revert due to:
        - Slow-moving macro fundamentals
        - Dealer hedging activity
        - High liquidity enabling quick price discovery
        """
        logger.info("Generating Mean Reversion Signal...")

        lookback = self.config.LOOKBACK_PERIOD
        entry_threshold = self.config.ENTRY_ZSCORE

        # Ensure we have required features
        if f'ma_{lookback}' not in df.columns:
            df[f'ma_{lookback}'] = df['price'].rolling(window=lookback).mean()

        if 'z_score' not in df.columns:
            deviation = df['price'] - df[f'ma_{lookback}']
            rolling_std = df['price'].rolling(window=lookback).std()
            df['z_score'] = deviation / rolling_std

        # Generate signal
        df['signal_mean_reversion'] = 0
        df.loc[df['z_score'] < -entry_threshold, 'signal_mean_reversion'] = 1  # Long (oversold)
        df.loc[df['z_score'] > entry_threshold, 'signal_mean_reversion'] = -1  # Short (overbought)

        # Signal strength (continuous)
        df['signal_mr_strength'] = -df['z_score'] / entry_threshold
        df['signal_mr_strength'] = df['signal_mr_strength'].clip(-2, 2)

        # Volatility filter (only trade in reasonable vol regime)
        if 'volatility_20' in df.columns:
            vol_threshold = df['volatility_20'].quantile(0.75)
            df['vol_regime_ok'] = (df['volatility_20'] < vol_threshold).astype(int)
            df['signal_mean_reversion'] = df['signal_mean_reversion'] * df['vol_regime_ok']

        signal_dist = df['signal_mean_reversion'].value_counts()
        logger.info(f"  Mean Reversion Signal: {signal_dist.to_dict()}")

        return df

    def _generate_momentum_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Momentum Signal based on moving average crossover

        Strategy:
        - LONG (1) when fast MA > slow MA (uptrend)
        - SHORT (-1) when fast MA < slow MA (downtrend)
        - NEUTRAL (0) when no clear trend

        Rationale: While futures mean-revert short-term, they can trend
        over weeks/months during Fed policy shifts
        """
        logger.info("Generating Momentum Signal...")

        fast_ma = 5
        slow_ma = 20

        # Ensure we have MAs
        if f'ma_{fast_ma}' not in df.columns:
            df[f'ma_{fast_ma}'] = df['price'].rolling(window=fast_ma).mean()
        if f'ma_{slow_ma}' not in df.columns:
            df[f'ma_{slow_ma}'] = df['price'].rolling(window=slow_ma).mean()

        # Basic crossover signal
        df['ma_diff'] = df[f'ma_{fast_ma}'] - df[f'ma_{slow_ma}']
        df['signal_momentum'] = 0
        df.loc[df['ma_diff'] > 0, 'signal_momentum'] = 1  # Uptrend
        df.loc[df['ma_diff'] < 0, 'signal_momentum'] = -1  # Downtrend

        # Signal strength based on magnitude of difference
        df['signal_mom_strength'] = df['ma_diff'] / df[f'ma_{slow_ma}'] * 100
        df['signal_mom_strength'] = df['signal_mom_strength'].clip(-2, 2)

        # Add momentum confirmation (ROC)
        if 'roc_20' in df.columns:
            # Only signal if ROC confirms
            df.loc[(df['signal_momentum'] == 1) & (df['roc_20'] < 0), 'signal_momentum'] = 0
            df.loc[(df['signal_momentum'] == -1) & (df['roc_20'] > 0), 'signal_momentum'] = 0

        signal_dist = df['signal_momentum'].value_counts()
        logger.info(f"  Momentum Signal: {signal_dist.to_dict()}")

        return df

    def generate_combined_signal(self, df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
        """
        Generate combined signal using learned weights

        Args:
            df: DataFrame with individual signals
            weights: Dictionary with 'mean_reversion' and 'momentum' weights

        Returns:
            DataFrame with combined signal
        """
        logger.info(f"Generating Combined Signal with weights: {weights}")

        df = df.copy()

        # Weighted combination
        w_mr = weights.get('mean_reversion', 0.5)
        w_mom = weights.get('momentum', 0.5)

        # Use signal strengths for combination
        if 'signal_mr_strength' in df.columns and 'signal_mom_strength' in df.columns:
            df['signal_combined_raw'] = (w_mr * df['signal_mr_strength'] +
                                         w_mom * df['signal_mom_strength'])
        else:
            df['signal_combined_raw'] = (w_mr * df['signal_mean_reversion'] +
                                         w_mom * df['signal_momentum'])

        # Convert to discrete signal
        threshold = 0.5
        df['signal_combined'] = 0
        df.loc[df['signal_combined_raw'] > threshold, 'signal_combined'] = 1
        df.loc[df['signal_combined_raw'] < -threshold, 'signal_combined'] = -1

        signal_dist = df['signal_combined'].value_counts()
        logger.info(f"  Combined Signal: {signal_dist.to_dict()}")

        return df

    def get_signal_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics for all signals"""

        signals = ['signal_mean_reversion', 'signal_momentum', 'signal_combined']
        available_signals = [s for s in signals if s in df.columns]

        summary = []

        for signal_name in available_signals:
            dist = df[signal_name].value_counts()

            summary.append({
                'signal': signal_name,
                'long_count': dist.get(1, 0),
                'short_count': dist.get(-1, 0),
                'neutral_count': dist.get(0, 0),
                'long_pct': dist.get(1, 0) / len(df) * 100,
                'short_pct': dist.get(-1, 0) / len(df) * 100,
                'neutral_pct': dist.get(0, 0) / len(df) * 100,
            })

        return pd.DataFrame(summary)