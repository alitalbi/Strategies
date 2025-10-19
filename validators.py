"""
Anomaly detection and data quality checks
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detect data quality issues and anomalies"""

    def __init__(self, config):
        self.config = config
        self.anomalies = {}

    def detect_all_anomalies(self, df: pd.DataFrame) -> Dict:
        """
        Run all anomaly detection methods

        Returns:
            Dictionary of anomaly reports
        """
        logger.info("Starting comprehensive anomaly detection...")

        self.anomalies = {
            'duplicates': self._check_duplicates(df),
            'missing_values': self._check_missing_values(df),
            'price_outliers': self._check_price_outliers(df),
            'price_spikes': self._check_price_spikes(df),
            'time_gaps': self._check_time_gaps(df),
            'flat_periods': self._check_flat_periods(df),
            'impossible_values': self._check_impossible_values(df),
        }

        self._log_anomaly_summary()

        return self.anomalies

    def _check_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate timestamps"""
        duplicates = df.duplicated(subset=['timestamp'], keep=False)
        dup_count = duplicates.sum()

        result = {
            'count': int(dup_count),
            'percentage': float(dup_count / len(df) * 100),
            'severity': 'HIGH' if dup_count > 0 else 'NONE',
            'action': 'REMOVE' if dup_count > 0 else 'NONE',
        }

        if dup_count > 0:
            logger.warning(f"Found {dup_count} duplicate timestamps ({result['percentage']:.2f}%)")
            # Show examples
            dup_examples = df[duplicates].head(3)
            logger.warning(f"Examples:\n{dup_examples[['timestamp', 'price']]}")

        return result

    def _check_missing_values(self, df: pd.DataFrame) -> Dict:
        """Check for missing values"""
        missing = df.isnull().sum()

        result = {
            'by_column': missing.to_dict(),
            'total': int(missing.sum()),
            'percentage': float(missing.sum() / (len(df) * len(df.columns)) * 100),
            'severity': 'MEDIUM' if missing.sum() > 0 else 'NONE',
        }

        if result['total'] > 0:
            logger.warning(f"Found {result['total']} missing values")
            for col, count in result['by_column'].items():
                if count > 0:
                    logger.warning(f"  {col}: {count} missing ({count / len(df) * 100:.2f}%)")

        return result

    def _check_price_outliers(self, df: pd.DataFrame) -> Dict:
        """Detect statistical outliers in prices using Z-score"""
        prices = df['price'].dropna()
        z_scores = np.abs((prices - prices.mean()) / prices.std())

        outliers = z_scores > self.config.ZSCORE_THRESHOLD
        outlier_count = outliers.sum()

        result = {
            'count': int(outlier_count),
            'percentage': float(outlier_count / len(prices) * 100),
            'threshold': self.config.ZSCORE_THRESHOLD,
            'severity': 'MEDIUM' if outlier_count > 10 else 'LOW',
            'action': 'INVESTIGATE',
        }

        if outlier_count > 0:
            outlier_prices = df.loc[outliers, ['timestamp', 'price']]
            logger.warning(f"Found {outlier_count} price outliers beyond {self.config.ZSCORE_THRESHOLD} sigma")
            logger.warning(
                f"Outlier range: ${outlier_prices['price'].min():.2f} - ${outlier_prices['price'].max():.2f}")

        return result

    def _check_price_spikes(self, df: pd.DataFrame) -> Dict:
        """Detect sudden price movements that may indicate errors"""
        df = df.copy()
        df['price_change_pct'] = df['price'].pct_change() * 100

        large_moves = np.abs(df['price_change_pct']) > self.config.MAX_SINGLE_MOVE_PCT
        spike_count = large_moves.sum()

        result = {
            'count': int(spike_count),
            'percentage': float(spike_count / len(df) * 100),
            'threshold_pct': self.config.MAX_SINGLE_MOVE_PCT,
            'max_move': float(df['price_change_pct'].abs().max()) if len(df) > 1 else 0,
            'severity': 'HIGH' if spike_count > 50 else 'MEDIUM' if spike_count > 10 else 'LOW',
        }

        if spike_count > 0:
            logger.warning(f"Found {spike_count} price spikes > {self.config.MAX_SINGLE_MOVE_PCT}%")
            # Show top 3 spikes
            top_spikes = df.nlargest(3, 'price_change_pct')[['timestamp', 'price', 'price_change_pct']]
            logger.warning(f"Largest spikes:\n{top_spikes}")

        return result

    def _check_time_gaps(self, df: pd.DataFrame) -> Dict:
        """Detect gaps in time series"""
        df = df.copy()
        df['time_diff'] = df['timestamp'].diff()

        # Infer typical frequency
        mode_diff = df['time_diff'].mode()
        if len(mode_diff) > 0:
            typical_freq = mode_diff[0]
        else:
            typical_freq = pd.Timedelta(minutes=1)  # Default assumption

        # Flag gaps > 2x typical frequency
        large_gaps = df['time_diff'] > typical_freq * 2
        gap_count = large_gaps.sum()

        result = {
            'count': int(gap_count),
            'typical_frequency': str(typical_freq),
            'max_gap': str(df['time_diff'].max()) if len(df) > 1 else 'N/A',
            'severity': 'MEDIUM' if gap_count > 20 else 'LOW',
        }

        if gap_count > 0:
            logger.warning(f"Found {gap_count} time gaps > {typical_freq * 2}")
            gap_examples = df[large_gaps][['timestamp', 'time_diff']].head(3)
            logger.warning(f"Examples:\n{gap_examples}")

        return result

    def _check_flat_periods(self, df: pd.DataFrame) -> Dict:
        """Detect periods where price doesn't change (suspicious for liquid futures)"""
        df = df.copy()
        df['price_unchanged'] = df['price'] == df['price'].shift(1)

        # Find consecutive runs of unchanged prices
        df['flat_group'] = (df['price_unchanged'] != df['price_unchanged'].shift()).cumsum()
        flat_runs = df[df['price_unchanged']].groupby('flat_group').size()

        # Flag runs > 10 consecutive periods
        long_flats = flat_runs[flat_runs > 10]

        result = {
            'num_long_flat_periods': len(long_flats),
            'max_consecutive_unchanged': int(flat_runs.max()) if len(flat_runs) > 0 else 0,
            'severity': 'MEDIUM' if len(long_flats) > 5 else 'LOW',
        }

        if len(long_flats) > 0:
            logger.warning(f"Found {len(long_flats)} periods with >10 consecutive unchanged prices")
            logger.warning(f"Max consecutive unchanged: {result['max_consecutive_unchanged']}")

        return result

    def _check_impossible_values(self, df: pd.DataFrame) -> Dict:
        """Check for values that are impossible/unrealistic"""
        issues = []

        # Negative or zero prices
        negative = (df['price'] <= 0).sum()
        if negative > 0:
            issues.append(f"{negative} non-positive prices")

        # Prices outside realistic range for 5Y futures
        below_min = (df['price'] < self.config.MIN_PRICE).sum()
        above_max = (df['price'] > self.config.MAX_PRICE).sum()

        if below_min > 0:
            issues.append(f"{below_min} prices below ${self.config.MIN_PRICE}")
        if above_max > 0:
            issues.append(f"{above_max} prices above ${self.config.MAX_PRICE}")

        result = {
            'issues_found': issues,
            'severity': 'HIGH' if issues else 'NONE',
            'negative_prices': int(negative),
            'below_min': int(below_min),
            'above_max': int(above_max),
        }

        if issues:
            logger.error(f"Found impossible values: {', '.join(issues)}")

        return result

    def _log_anomaly_summary(self):
        """Print summary of all anomalies"""
        logger.info("\n" + "=" * 60)
        logger.info("ANOMALY DETECTION SUMMARY")
        logger.info("=" * 60)

        for anomaly_type, details in self.anomalies.items():
            severity = details.get('severity', 'UNKNOWN')
            count = details.get('count', details.get('total', 'N/A'))
            logger.info(f"{anomaly_type.upper()}: {count} issues (Severity: {severity})")

        logger.info("=" * 60 + "\n")

    def generate_report(self) -> str:
        """Generate human-readable anomaly report"""
        report = []
        report.append("\n" + "=" * 60)
        report.append("ANOMALY DETECTION REPORT")
        report.append("=" * 60 + "\n")

        for anomaly_type, details in self.anomalies.items():
            report.append(f"\n{anomaly_type.upper().replace('_', ' ')}:")
            report.append("-" * 40)
            for key, value in details.items():
                report.append(f"  {key}: {value}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)