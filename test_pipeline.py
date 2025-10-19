# test_pipeline.py
"""
Quick test to verify pipeline works before submission
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def create_dummy_data():
    """Create synthetic data for testing"""
    dates = pd.date_range('2020-01-01', '2025-01-01', freq='1H')

    # Generate realistic-ish price series
    np.random.seed(42)
    returns = np.random.normal(0, 0.001, len(dates))
    prices = 110 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'timestamp': dates,
        'price': prices
    })

    # Add some anomalies
    df.loc[100, 'price'] = np.nan  # Missing value
    df.loc[200:202, 'timestamp'] = df.loc[200, 'timestamp']  # Duplicates
    df.loc[300, 'price'] = 200  # Outlier

    return df


def test_pipeline():
    """Test the pipeline with dummy data"""
    print("Creating test data...")
    df = create_dummy_data()

    # Save test data
    Path('data').mkdir(exist_ok=True)
    df.to_csv('data/test_data.csv', index=False)
    print(f"✓ Created test data: {len(df)} rows")

    # Import and run pipeline
    print("\nRunning pipeline...")
    from main import FuturesAnalysisPipeline

    pipeline = FuturesAnalysisPipeline()
    result = pipeline.run()

    print(f"\n✅ Pipeline test successful!")
    print(f"   Output rows: {len(result)}")
    print(f"   Output columns: {len(result.columns)}")
    print(f"   Output files created: {len(list(Path('output').glob('*')))}")

    return result


if __name__ == "__main__":
    test_pipeline()