"""
SQL-style analytics implemented in Python/Pandas
All queries from queries.sql translated to pandas operations
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class SQLAnalytics:
    """
    Replicate SQL analytical queries using pandas
    More flexible than SQL for complex time-series operations
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with futures dataframe

        Args:
            df: DataFrame with at minimum 'timestamp' and 'price' columns
        """
        self.df = df.copy()
        self.results = {}

    def run_all_queries(self) -> Dict[str, pd.DataFrame]:
        """
        Execute all analytical queries

        Returns:
            Dictionary of query results
        """
        logger.info("Running all SQL-style analytics...")

        self.results['daily_summary'] = self.query_daily_summary()
        self.results['moving_averages'] = self.query_moving_averages()
        self.results['returns_volatility'] = self.query_returns_volatility()
        self.results['extreme_moves'] = self.query_extreme_moves()
        self.results['signal_performance'] = self.query_signal_performance()
        self.results['volatility_regimes'] = self.query_volatility_regimes()
        self.results['time_patterns'] = self.query_time_patterns()
        self.results['drawdown_analysis'] = self.query_drawdown_analysis()
        self.results['mean_reversion_speed'] = self.query_mean_reversion_speed()
        self.results['feature_correlations'] = self.query_feature_correlations()

        logger.info(f"✓ Completed {len(self.results)} analytical queries")
        return self.results

    def query_daily_summary(self) -> pd.DataFrame:
        """
        Query 1: Daily OHLC summary statistics

        SQL equivalent:
        SELECT DATE(timestamp), MIN(price), MAX(price), AVG(price), COUNT(*)
        FROM futures GROUP BY DATE(timestamp)
        """
        logger.info("Running Query 1: Daily Summary Statistics")

        daily = self.df.groupby(self.df['timestamp'].dt.date).agg({
            'price': ['first', 'min', 'max', 'last', 'mean', 'std', 'count']
        })

        daily.columns = ['open', 'low', 'high', 'close', 'avg', 'volatility', 'num_obs']
        daily.reset_index(inplace=True)
        daily.rename(columns={'timestamp': 'trade_date'}, inplace=True)

        # Daily returns
        daily['daily_return'] = (daily['close'] - daily['open']) / daily['open'] * 100

        return daily

    def query_moving_averages(self) -> pd.DataFrame:
        """
        Query 2: Moving averages with multiple windows

        SQL equivalent: Window functions with ROWS BETWEEN
        """
        logger.info("Running Query 2: Moving Averages Analysis")

        result = self.df[['timestamp', 'price']].copy()

        # Calculate multiple MAs
        for window in [5, 20, 50]:
            result[f'ma_{window}'] = result['price'].rolling(window=window).mean()
            result[f'deviation_from_ma_{window}'] = result['price'] - result[f'ma_{window}']
            result[f'deviation_pct_{window}'] = (result['price'] - result[f'ma_{window}']) / result[
                f'ma_{window}'] * 100

        return result

    def query_returns_volatility(self) -> pd.DataFrame:
        """
        Query 3: Returns and rolling volatility

        SQL equivalent: LAG() and window functions
        """
        logger.info("Running Query 3: Returns & Volatility Analysis")

        result = self.df[['timestamp', 'price']].copy()

        # Returns
        result['prev_price'] = result['price'].shift(1)
        result['return_pct'] = (result['price'] - result['prev_price']) / result['prev_price'] * 100
        result['log_return'] = np.log(result['price'] / result['prev_price'])

        # Rolling volatility (multiple windows)
        for window in [10, 20, 50]:
            result[f'rolling_vol_{window}'] = result['return_pct'].rolling(window=window).std()
            result[f'rolling_vol_{window}_annual'] = result[f'rolling_vol_{window}'] * np.sqrt(252)

        return result

    def query_extreme_moves(self) -> pd.DataFrame:
        """
        Query 4: Identify extreme price movements

        SQL equivalent: WHERE with absolute value filters
        """
        logger.info("Running Query 4: Extreme Moves Identification")

        # Calculate daily returns
        daily = self.df.groupby(self.df['timestamp'].dt.date).agg({
            'price': ['first', 'last']
        })
        daily.columns = ['open', 'close']
        daily['daily_return_pct'] = (daily['close'] - daily['open']) / daily['open'] * 100
        daily['abs_return_pct'] = abs(daily['daily_return_pct'])

        # Filter extreme moves (>1%)
        extreme = daily[daily['abs_return_pct'] > 1.0].copy()
        extreme = extreme.sort_values('abs_return_pct', ascending=False)
        extreme.reset_index(inplace=True)
        extreme.rename(columns={'timestamp': 'trade_date'}, inplace=True)

        return extreme.head(20)

    def query_signal_performance(self) -> pd.DataFrame:
        """
        Query 5: Signal performance analysis

        SQL equivalent: LEAD() with GROUP BY
        """
        logger.info("Running Query 5: Signal Performance Analysis")

        if 'signal' not in self.df.columns:
            logger.warning("No 'signal' column found, skipping signal performance query")
            return pd.DataFrame()

        result = self.df[self.df['signal'] != 0][['timestamp', 'price', 'signal', 'z_score']].copy()

        # Calculate future returns
        result['price_1_ahead'] = self.df['price'].shift(-1)
        result['price_5_ahead'] = self.df['price'].shift(-5)
        result['price_20_ahead'] = self.df['price'].shift(-20)

        result['return_1period'] = (result['price_1_ahead'] - result['price']) / result['price'] * 100
        result['return_5period'] = (result['price_5_ahead'] - result['price']) / result['price'] * 100
        result['return_20period'] = (result['price_20_ahead'] - result['price']) / result['price'] * 100

        # Group by signal
        summary = result.groupby('signal').agg({
            'signal': 'count',
            'return_1period': 'mean',
            'return_5period': 'mean',
            'return_20period': 'mean',
            'z_score': 'mean'
        })

        summary.columns = ['signal_count', 'avg_1period_return', 'avg_5period_return',
                           'avg_20period_return', 'avg_z_score']
        summary.reset_index(inplace=True)

        return summary

    def query_volatility_regimes(self) -> pd.DataFrame:
        """
        Query 6: Classify into volatility regimes

        SQL equivalent: PERCENTILE_CONT with CASE statements
        """
        logger.info("Running Query 6: Volatility Regime Classification")

        if 'volatility_20' not in self.df.columns:
            logger.warning("No 'volatility_20' column found, calculating it")
            self.df['returns'] = self.df['price'].pct_change()
            self.df['volatility_20'] = self.df['returns'].rolling(20).std()

        # Calculate percentiles
        vol_33 = self.df['volatility_20'].quantile(0.33)
        vol_67 = self.df['volatility_20'].quantile(0.67)

        # Classify regimes
        def classify_regime(vol):
            if pd.isna(vol):
                return 'UNKNOWN'
            elif vol < vol_33:
                return 'LOW_VOL'
            elif vol < vol_67:
                return 'MEDIUM_VOL'
            else:
                return 'HIGH_VOL'

        result = self.df.copy()
        result['vol_regime'] = result['volatility_20'].apply(classify_regime)

        # Aggregate by regime
        regime_summary = result.groupby('vol_regime').agg({
            'price': ['count', 'mean', 'std'],
            'returns': ['mean', 'std']
        })

        regime_summary.columns = ['observation_count', 'avg_price', 'price_std',
                                  'avg_return', 'return_std']
        regime_summary.reset_index(inplace=True)

        return regime_summary

    def query_time_patterns(self) -> pd.DataFrame:
        """
        Query 7: Time-based patterns (day of week, etc.)

        SQL equivalent: GROUP BY with date functions
        """
        logger.info("Running Query 7: Time-Based Pattern Analysis")

        if 'returns' not in self.df.columns:
            self.df['returns'] = self.df['price'].pct_change()

        result = self.df.copy()
        result['day_of_week'] = result['timestamp'].dt.dayofweek
        result['day_name'] = result['timestamp'].dt.day_name()

        # Aggregate by day of week
        day_summary = result.groupby(['day_of_week', 'day_name']).agg({
            'returns': ['count', 'mean', 'std'],
            'price': ['min', 'max']
        })

        day_summary.columns = ['observation_count', 'avg_return', 'return_volatility',
                               'min_price', 'max_price']
        day_summary.reset_index(inplace=True)
        day_summary = day_summary.sort_values('day_of_week')

        return day_summary

    def query_drawdown_analysis(self) -> pd.DataFrame:
        """
        Query 8: Drawdown calculation

        SQL equivalent: Running MAX with self-join
        """
        logger.info("Running Query 8: Drawdown Analysis")

        result = self.df[['timestamp', 'price']].copy()

        # Calculate running maximum
        result['running_max'] = result['price'].expanding().max()
        result['drawdown'] = result['price'] - result['running_max']
        result['drawdown_pct'] = (result['drawdown'] / result['running_max']) * 100

        # Filter significant drawdowns
        significant_dd = result[result['drawdown_pct'] < -0.5].copy()
        significant_dd = significant_dd.sort_values('drawdown_pct')

        return significant_dd.head(20)

    def query_mean_reversion_speed(self) -> pd.DataFrame:
        """
        Query 9: How quickly price reverts after extreme moves

        SQL equivalent: Complex self-join with date arithmetic
        """
        logger.info("Running Query 9: Mean Reversion Speed Analysis")

        if 'z_score' not in self.df.columns:
            logger.warning("No 'z_score' column found, skipping mean reversion speed query")
            return pd.DataFrame()

        # Find extreme z-score events
        extreme_events = self.df[abs(self.df['z_score']) > 1.5].copy()
        extreme_events['event_id'] = range(len(extreme_events))

        reversion_data = []

        # For each extreme event, track reversion
        for idx, row in extreme_events.iterrows():
            event_time = row['timestamp']
            event_zscore = row['z_score']

            # Look ahead for reversion
            future_data = self.df[self.df['timestamp'] > event_time].head(20)  # Next 20 periods

            for days_ahead, (_, future_row) in enumerate(future_data.iterrows(), 1):
                future_zscore = future_row.get('z_score', np.nan)

                if pd.notna(future_zscore):
                    reverted = abs(future_zscore) < 0.5

                    reversion_data.append({
                        'extreme_time': event_time,
                        'extreme_zscore': event_zscore,
                        'days_elapsed': days_ahead,
                        'future_zscore': future_zscore,
                        'reverted': reverted
                    })

        if reversion_data:
            result = pd.DataFrame(reversion_data)

            # Calculate average reversion time
            reversion_summary = result[result['reverted']].groupby('extreme_time').agg({
                'days_elapsed': 'min'
            }).reset_index()
            reversion_summary.columns = ['extreme_time', 'reversion_days']

            return reversion_summary

        return pd.DataFrame()

    def query_feature_correlations(self) -> pd.DataFrame:
        """
        Query 10: Feature correlation with future returns

        SQL equivalent: CORR() with LEAD()
        """
        logger.info("Running Query 10: Feature Correlation Analysis")

        if 'returns' not in self.df.columns:
            self.df['returns'] = self.df['price'].pct_change()

        # Calculate future returns
        self.df['return_1_ahead'] = self.df['returns'].shift(-1)
        self.df['return_5_ahead'] = self.df['returns'].shift(-5).rolling(5).sum()

        # Features to analyze
        features = ['rsi', 'z_score', 'momentum_5', 'volatility_20']

        correlations = []

        for feature in features:
            if feature in self.df.columns:
                corr_1 = self.df[feature].corr(self.df['return_1_ahead'])
                corr_5 = self.df[feature].corr(self.df['return_5_ahead'])

                correlations.append({
                    'feature': feature,
                    'correlation_1period': corr_1,
                    'correlation_5period': corr_5
                })

        return pd.DataFrame(correlations)

    def save_all_results(self, output_dir: str = 'output/sql_analytics'):
        """Save all query results to CSV"""
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for query_name, result_df in self.results.items():
            if not result_df.empty:
                filepath = f"{output_dir}/{query_name}.csv"
                result_df.to_csv(filepath, index=False)
                logger.info(f"✓ Saved {query_name}: {filepath}")