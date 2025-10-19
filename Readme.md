# ADIA Futures Analysis Pipeline

Complete data analysis pipeline for 5-Year Treasury Futures data.

## Quick Start

1. **Place your data file** in the `data/` directory
2. **Update config.py** with your actual column names
3. **Run the pipeline:**
```bash
   python main.py
```

## What This Pipeline Does

### 1. Data Loading & Validation
- Loads CSV data
- Standardizes column names
- Validates data quality

### 2. Anomaly Detection
- Finds duplicates, outliers, price spikes
- Detects time gaps and flat periods
- Generates detailed anomaly report

### 3. Data Cleaning
- Removes duplicates and impossible values
- Handles missing data intelligently
- Filters extreme outliers

### 4. Feature Engineering
- Creates 40+ technical features
- Generates mean reversion signal
- Adds time-based features

### 5. Database Storage
- Saves to SQLite database
- Creates indexed tables
- Enables SQL analysis

### 6. SQL Analysis
- Executes 10 analytical queries
- Performance metrics
- Signal quality assessment

### 7. Reporting
- Comprehensive final report
- Feature summaries
- Signal distribution analysis

## Output Files
```
output/
├── cleaned_futures.csv          # Cleaned dataset
├── futures.db                   # SQLite database
├── FINAL_REPORT.txt            # Main report
├── anomaly_report.txt          # Anomaly details
├── feature_summary.csv         # Feature statistics
├── summary_statistics.csv      # Data summary
├── signal_distribution.csv     # Signal analysis
└── query_*_results.csv         # SQL query results
```

## Signal Strategy

**Mean Reversion Signal** based on Z-score deviation from moving average:
- **LONG** when price is >1.5 std devs below 20-day MA
- **SHORT** when price is >1.5 std devs above 20-day MA
- **EXIT** when price returns to mean

**Rationale:** 5Y Treasury futures exhibit mean-reverting behavior over short horizons due to slow-moving macro fundamentals and dealer hedging activity.

## Configuration

Edit `config.py` to adjust:
- File paths
- Cleaning thresholds
- Feature parameters
- Signal logic

## Requirements
```
pandas
numpy
sqlite3 (built-in)
```

## Time to Complete

Estimated: 15-20 minutes on typical 5-year dataset

## Contact

[Your Name]
[Date]