"""
ADIA Futures Data Pipeline - Main Execution Script
Complete end-to-end pipeline for 5-Year Treasury Futures analysis

Author: [Your Name]
Date: [Date]
Purpose: ADIA Interview Round 2 - Data Analysis Assessment
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import sys

# Local imports
from config import DATA_CONFIG, CLEANING_CONFIG, FEATURE_CONFIG, SIGNAL_CONFIG
from utils.data_loader import FuturesDataLoader
from utils.validators import AnomalyDetector
from utils.cleaner import FuturesDataCleaner
from utils.features import FeatureEngineer
from utils.database import DatabaseManager


# Setup logging
def setup_logging():
    """Configure logging to file and console"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Create logs directory
    Path('logs').mkdir(exist_ok=True)

    # File handler
    file_handler = logging.FileHandler(
        f'logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger


logger = setup_logging()


class FuturesAnalysisPipeline:
    """
    End-to-end pipeline for futures data analysis

    Pipeline stages:
    1. Data Loading & Standardization
    2. Anomaly Detection
    3. Data Cleaning
    4. Feature Engineering
    5. Signal Generation
    6. Database Storage
    7. SQL Analysis
    8. Reporting
    """

    def __init__(self):
        self.loader = FuturesDataLoader(DATA_CONFIG)
        self.detector = AnomalyDetector(CLEANING_CONFIG)
        self.cleaner = FuturesDataCleaner(CLEANING_CONFIG)
        self.engineer = FeatureEngineer(FEATURE_CONFIG)

        self.raw_data = None
        self.cleaned_data = None
        self.featured_data = None
        self.anomaly_report = None
        self.cleaning_report = None

        # Create output directory
        Path('output').mkdir(exist_ok=True)

    def run(self):
        """Execute full pipeline"""
        logger.info("=" * 70)
        logger.info("ADIA FUTURES ANALYSIS PIPELINE - STARTING")
        logger.info("=" * 70)

        start_time = datetime.now()

        try:
            # Stage 1: Load Data
            logger.info("\n[STAGE 1/8] DATA LOADING")
            logger.info("-" * 70)
            self.raw_data = self.load_and_prepare_data()

            # Stage 2: Detect Anomalies
            logger.info("\n[STAGE 2/8] ANOMALY DETECTION")
            logger.info("-" * 70)
            self.anomaly_report = self.detect_anomalies()

            # Stage 3: Clean Data
            logger.info("\n[STAGE 3/8] DATA CLEANING")
            logger.info("-" * 70)
            self.cleaned_data, self.cleaning_report = self.clean_data()

            # Stage 4: Engineer Features
            logger.info("\n[STAGE 4/8] FEATURE ENGINEERING")
            logger.info("-" * 70)
            self.featured_data = self.engineer_features()

            # Stage 5: Save to Database
            logger.info("\n[STAGE 5/8] DATABASE STORAGE")
            logger.info("-" * 70)
            self.save_to_database()

            # Stage 6: Execute SQL Queries
            logger.info("\n[STAGE 6/8] SQL ANALYSIS")
            logger.info("-" * 70)
            sql_results = self.run_sql_analysis()

            # Stage 7: Save Outputs
            logger.info("\n[STAGE 7/8] SAVING OUTPUTS")
            logger.info("-" * 70)
            self.save_outputs()

            # Stage 8: Generate Report
            logger.info("\n[STAGE 8/8] GENERATING FINAL REPORT")
            logger.info("-" * 70)
            self.generate_final_report()

            elapsed = datetime.now() - start_time
            logger.info("\n" + "=" * 70)
            logger.info(f"PIPELINE COMPLETED SUCCESSFULLY in {elapsed}")
            logger.info("=" * 70)

            return self.featured_data

        except Exception as e:
            logger.error(f"\n{'=' * 70}")
            logger.error(f"PIPELINE FAILED: {str(e)}")
            logger.error(f"{'=' * 70}")
            raise

    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and standardize raw data"""
        # Load
        df = self.loader.load_data(DATA_CONFIG.INPUT_FILE)

        # Standardize columns
        df = self.loader.standardize_columns(df)

        # Parse timestamps
        df = self.loader.parse_timestamps(df)

        # Initial validation
        df, validation_report = self.loader.initial_validation(df)

        logger.info(f"✓ Data loaded successfully: {len(df)} rows")
        return df

    def detect_anomalies(self) -> dict:
        """Run anomaly detection"""
        anomalies = self.detector.detect_all_anomalies(self.raw_data)

        # Save anomaly report
        report_text = self.detector.generate_report()
        with open('output/anomaly_report.txt', 'w') as f:
            f.write(report_text)

        logger.info("✓ Anomaly detection complete - report saved")
        return anomalies

    def clean_data(self) -> tuple:
        """Clean the data"""
        cleaned, report = self.cleaner.clean_data(self.raw_data, self.anomaly_report)

        # Print cleaning report
        self.cleaner.print_report(report)

        logger.info("✓ Data cleaning complete")
        return cleaned, report

    def engineer_features(self) -> pd.DataFrame:
        """Create features and signals"""
        featured = self.engineer.engineer_all_features(self.cleaned_data)

        # Get feature summary
        feature_summary = self.engineer.get_feature_importance_summary(featured)
        feature_summary.to_csv('output/feature_summary.csv', index=False)

        logger.info(f"✓ Feature engineering complete: {len(featured.columns)} total columns")
        return featured

    def save_to_database(self):
        """Save data to SQLite database"""
        with DatabaseManager(DATA_CONFIG.OUTPUT_DB) as db:
            # Save main table
            db.save_dataframe(self.featured_data, 'futures_clean', if_exists='replace')

            # Save metadata table
            metadata = {
                'pipeline_run_time': [datetime.now()],
                'input_file': [DATA_CONFIG.INPUT_FILE],
                'total_rows': [len(self.featured_data)],
                'date_range_start': [self.featured_data['timestamp'].min()],
                'date_range_end': [self.featured_data['timestamp'].max()],
                'num_features': [len(self.featured_data.columns)],
            }
            db.save_dataframe(pd.DataFrame(metadata), 'pipeline_metadata', if_exists='replace')

            logger.info(f"✓ Data saved to database: {DATA_CONFIG.OUTPUT_DB}")

    def run_sql_analysis(self) -> list:
        """Execute SQL queries"""
        results = []

        with DatabaseManager(DATA_CONFIG.OUTPUT_DB) as db:
            # Check if SQL file exists
            sql_file = Path('sql/queries.sql')
            if sql_file.exists():
                results = db.execute_query_file(str(sql_file))

                # Save query results
                for i, result in enumerate(results, 1):
                    result.to_csv(f'output/query_{i}_results.csv', index=False)
                    logger.info(f"  Query {i}: {len(result)} rows")
            else:
                logger.warning("No SQL file found, skipping SQL analysis")

        logger.info(f"✓ SQL analysis complete: {len(results)} queries executed")
        return results

    def save_outputs(self):
        """Save all output files"""
        # Save cleaned data CSV
        self.featured_data.to_csv(DATA_CONFIG.OUTPUT_CSV, index=False)
        logger.info(f"✓ Saved: {DATA_CONFIG.OUTPUT_CSV}")

        # Save summary statistics
        summary_stats = self.featured_data.describe()
        summary_stats.to_csv('output/summary_statistics.csv')
        logger.info("✓ Saved: output/summary_statistics.csv")

        # Save signal distribution
        signal_dist = self.featured_data['signal'].value_counts()
        signal_dist.to_csv('output/signal_distribution.csv')
        logger.info("✓ Saved: output/signal_distribution.csv")

    def generate_final_report(self):
        """Generate comprehensive final report"""
        report_lines = []

        report_lines.append("=" * 70)
        report_lines.append("ADIA FUTURES ANALYSIS - FINAL REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Input File: {DATA_CONFIG.INPUT_FILE}")

        # Data Summary
        report_lines.append("\n" + "=" * 70)
        report_lines.append("1. DATA SUMMARY")
        report_lines.append("=" * 70)
        report_lines.append(f"Original rows: {self.cleaning_report['original_rows']:,}")
        report_lines.append(f"Final rows: {self.cleaning_report['final_rows']:,}")
        report_lines.append(f"Retention rate: {self.cleaning_report['retention_rate']:.2f}%")
        report_lines.append(
            f"Date range: {self.featured_data['timestamp'].min()} to {self.featured_data['timestamp'].max()}")
        report_lines.append(
            f"Price range: ${self.featured_data['price'].min():.2f} - ${self.featured_data['price'].max():.2f}")

        # Anomalies
        report_lines.append("\n" + "=" * 70)
        report_lines.append("2. ANOMALIES DETECTED")
        report_lines.append("=" * 70)
        total_anomalies = sum([
            self.anomaly_report['duplicates']['count'],
            self.anomaly_report['price_outliers']['count'],
            self.anomaly_report['price_spikes']['count']
        ])
        report_lines.append(f"Total anomalies: {total_anomalies:,}")
        for anomaly_type, details in self.anomaly_report.items():
            count = details.get('count', 0)
            severity = details.get('severity', 'N/A')
            if count > 0:
                report_lines.append(f"  - {anomaly_type}: {count} ({severity} severity)")

        # Features Created
        report_lines.append("\n" + "=" * 70)
        report_lines.append("3. FEATURES ENGINEERED")
        report_lines.append("=" * 70)
        feature_cols = [c for c in self.featured_data.columns if c not in ['timestamp', 'price']]
        report_lines.append(f"Total features: {len(feature_cols)}")
        report_lines.append("\nFeature categories:")
        report_lines.append(f"  - Moving averages: {len([c for c in feature_cols if 'ma_' in c])}")
        report_lines.append(f"  - Volatility measures: {len([c for c in feature_cols if 'volatility' in c])}")
        report_lines.append(f"  - Momentum indicators: {len([c for c in feature_cols if 'momentum' in c])}")
        report_lines.append(
            f"  - Technical indicators: {len([c for c in feature_cols if any(x in c for x in ['rsi', 'bb_'])])}")

        # Signal Analysis
        report_lines.append("\n" + "=" * 70)
        report_lines.append("4. TRADING SIGNAL ANALYSIS")
        report_lines.append("=" * 70)
        signal_dist = self.featured_data['signal'].value_counts()
        total_signals = len(self.featured_data)
        report_lines.append(f"Signal distribution:")
        for signal, count in signal_dist.items():
            pct = count / total_signals * 100
            signal_name = {-1: 'SHORT', 0: 'NEUTRAL', 1: 'LONG'}.get(signal, 'UNKNOWN')
            report_lines.append(f"  - {signal_name}: {count:,} ({pct:.1f}%)")

        # Key Statistics
        report_lines.append("\n" + "=" * 70)
        report_lines.append("5. KEY STATISTICS")
        report_lines.append("=" * 70)
        report_lines.append(f"Average return: {self.featured_data['returns'].mean() * 100:.4f}%")
        report_lines.append(f"Return volatility: {self.featured_data['returns'].std() * 100:.4f}%")
        report_lines.append(
            f"Sharpe ratio (annualized): {self.featured_data['returns'].mean() / self.featured_data['returns'].std() * np.sqrt(252):.2f}")
        report_lines.append(
            f"Max drawdown: {(self.featured_data['price'] / self.featured_data['price'].expanding().max() - 1).min() * 100:.2f}%")

        # Recommendations
        report_lines.append("\n" + "=" * 70)
        report_lines.append("6. NEXT STEPS & RECOMMENDATIONS")
        report_lines.append("=" * 70)
        report_lines.append("- Paper trade signal for 3 months before live deployment")
        report_lines.append("- Conduct robustness tests across different time periods")
        report_lines.append("- Implement transaction cost modeling")
        report_lines.append("- Set up real-time monitoring dashboard")
        report_lines.append("- Define risk limits and stop-loss rules")

        # Files Generated
        report_lines.append("\n" + "=" * 70)
        report_lines.append("7. OUTPUT FILES")
        report_lines.append("=" * 70)
        output_files = [
            'output/cleaned_futures.csv',
            'output/futures.db',
            'output/feature_summary.csv',
            'output/summary_statistics.csv',
            'output/signal_distribution.csv',
            'output/anomaly_report.txt',
        ]
        for f in output_files:
            if Path(f).exists():
                size = Path(f).stat().st_size / 1024
                report_lines.append(f"  ✓ {f} ({size:.1f} KB)")

        report_lines.append("\n" + "=" * 70)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 70)

        # Save report
        report_text = "\n".join(report_lines)
        with open('output/FINAL_REPORT.txt', 'w') as f:
            f.write(report_text)

        # Also print to console
        print("\n" + report_text)

        logger.info("✓ Final report generated: output/FINAL_REPORT.txt")


def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("ADIA FUTURES ANALYSIS PIPELINE")
    print("5-Year Treasury Futures - Complete Analysis")
    print("=" * 70 + "\n")

    # Run pipeline
    pipeline = FuturesAnalysisPipeline()
    result = pipeline.run()

    print("\n✅ Pipeline execution complete!")
    print(f"📊 Output location: ./output/")
    print(f"📈 Database: {DATA_CONFIG.OUTPUT_DB}")
    print(f"📄 Report: ./output/FINAL_REPORT.txt")

    return result


if __name__ == "__main__":
    result = main()