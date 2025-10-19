"""
Visualization Module
Professional charts for analysis and reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class Visualizer:
    """
    Create publication-quality visualizations

    Chart types:
    - Price and signals
    - Equity curves
    - Drawdown charts
    - Feature importance
    - Correlation matrices
    - Performance comparisons
    - Regression diagnostics
    """

    def __init__(self, output_dir: str = 'output/charts'):
        """
        Initialize visualizer

        Args:
            output_dir: Directory to save charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_price_and_signals(self,
                               df: pd.DataFrame,
                               signal_col: str = 'signal_mean_reversion',
                               title: str = 'Price and Trading Signals'):
        """
        Plot price with trading signals overlaid

        Args:
            df: DataFrame with price and signal
            signal_col: Column name for signal
            title: Chart title
        """
        logger.info(f"Creating price and signals chart...")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                       gridspec_kw={'height_ratios': [3, 1]})

        # Price chart
        ax1.plot(df['timestamp'], df['price'], label='Price', color='black', linewidth=1)

        # Add moving average if available
        if 'ma_20' in df.columns:
            ax1.plot(df['timestamp'], df['ma_20'], label='20-day MA',
                     color='blue', linewidth=1, alpha=0.7)

        # Mark signals
        long_signals = df[df[signal_col] == 1]
        short_signals = df[df[signal_col] == -1]

        ax1.scatter(long_signals['timestamp'], long_signals['price'],
                    color='green', marker='^', s=100, label='Long Signal', zorder=5)
        ax1.scatter(short_signals['timestamp'], short_signals['price'],
                    color='red', marker='v', s=100, label='Short Signal', zorder=5)

        ax1.set_ylabel('Price', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Signal chart
        ax2.plot(df['timestamp'], df[signal_col], color='purple', linewidth=1.5)
        ax2.fill_between(df['timestamp'], 0, df[signal_col],
                         where=(df[signal_col] > 0), color='green', alpha=0.3, label='Long')
        ax2.fill_between(df['timestamp'], 0, df[signal_col],
                         where=(df[signal_col] < 0), color='red', alpha=0.3, label='Short')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Signal', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / 'price_and_signals.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved: {filepath}")

    def plot_equity_curve(self,
                          backtest_results: Dict,
                          title: str = 'Strategy Performance'):
        """
        Plot equity curve comparing strategy to buy-and-hold

        Args:
            backtest_results: Results from backtester
            title: Chart title
        """
        logger.info("Creating equity curve chart...")

        df = backtest_results['data']

        fig, ax = plt.subplots(figsize=(14, 8))

        ax.plot(df['timestamp'], df['cum_strategy_returns'],
                label='Strategy', color='blue', linewidth=2)
        ax.plot(df['timestamp'], df['cum_buy_hold'],
                label='Buy & Hold', color='gray', linewidth=2, linestyle='--')

        # Add 1.0 reference line
        ax.axhline(y=1.0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

        ax.set_ylabel('Cumulative Returns', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add performance text
        total_return = backtest_results['total_return']
        num_trades = backtest_results['num_trades']
        win_rate = backtest_results['win_rate']

        textstr = f'Total Return: {total_return:.2f}%\nTrades: {num_trades}\nWin Rate: {win_rate:.1f}%'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        filepath = self.output_dir / 'equity_curve.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved: {filepath}")

    def plot_drawdown(self, df: pd.DataFrame, title: str = 'Drawdown Analysis'):
        """
        Plot drawdown over time

        Args:
            df: DataFrame with returns
            title: Chart title
        """
        logger.info("Creating drawdown chart...")

        # Calculate drawdown
        cum_returns = (1 + df['strategy_returns_net']).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max * 100

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.fill_between(df['timestamp'], 0, drawdown, color='red', alpha=0.3)
        ax.plot(df['timestamp'], drawdown, color='darkred', linewidth=1)

        # Mark maximum drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min()
        ax.scatter(df.loc[max_dd_idx, 'timestamp'], max_dd_value,
                   color='red', s=200, zorder=5, marker='v')
        ax.annotate(f'Max DD: {max_dd_value:.2f}%',
                    xy=(df.loc[max_dd_idx, 'timestamp'], max_dd_value),
                    xytext=(20, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        filepath = self.output_dir / 'drawdown.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved: {filepath}")

    def plot_feature_importance(self,
                                feature_importance: pd.DataFrame,
                                title: str = 'Feature Importance',
                                top_n: int = 15):
        """
        Plot feature importance from regression

        Args:
            feature_importance: DataFrame with features and weights
            title: Chart title
            top_n: Number of top features to show
        """
        logger.info("Creating feature importance chart...")

        # Get top N features by absolute weight
        top_features = feature_importance.head(top_n).copy()

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = ['green' if w > 0 else 'red' for w in top_features['weight']]
        ax.barh(range(len(top_features)), top_features['abs_weight'], color=colors, alpha=0.6)

        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Absolute Weight', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        filepath = self.output_dir / 'feature_importance.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved: {filepath}")

    def plot_correlation_matrix(self,
                                df: pd.DataFrame,
                                features: List[str],
                                title: str = 'Feature Correlation Matrix'):
        """
        Plot correlation matrix heatmap

        Args:
            df: DataFrame with features
            features: List of feature columns
            title: Chart title
        """
        logger.info("Creating correlation matrix...")

        # Calculate correlation
        corr = df[features].corr()

        fig, ax = plt.subplots(figsize=(12, 10))

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                    annot=True, fmt='.2f', ax=ax)

        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        filepath = self.output_dir / 'correlation_matrix.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved: {filepath}")

    def plot_signal_comparison(self,
                               comparison_df: pd.DataFrame,
                               title: str = 'Signal Performance Comparison'):
        """
        Compare multiple signals side-by-side

        Args:
            comparison_df: DataFrame with comparison metrics
            title: Chart title
        """
        logger.info("Creating signal comparison chart...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Total Returns
        axes[0, 0].bar(range(len(comparison_df)), comparison_df['total_return'], color='steelblue')
        axes[0, 0].set_xticks(range(len(comparison_df)))
        axes[0, 0].set_xticklabels(comparison_df['strategy'], rotation=45, ha='right')
        axes[0, 0].set_ylabel('Total Return (%)')
        axes[0, 0].set_title('Total Returns')
        axes[0, 0].grid(True, alpha=0.3)

        # Number of Trades
        axes[0, 1].bar(range(len(comparison_df)), comparison_df['num_trades'], color='coral')
        axes[0, 1].set_xticks(range(len(comparison_df)))
        axes[0, 1].set_xticklabels(comparison_df['strategy'], rotation=45, ha='right')
        axes[0, 1].set_ylabel('Number of Trades')
        axes[0, 1].set_title('Trading Frequency')
        axes[0, 1].grid(True, alpha=0.3)

        # Win Rate
        axes[1, 0].bar(range(len(comparison_df)), comparison_df['win_rate'], color='green', alpha=0.6)
        axes[1, 0].axhline(y=50, color='red', linestyle='--', linewidth=1, label='50% baseline')
        axes[1, 0].set_xticks(range(len(comparison_df)))
        axes[1, 0].set_xticklabels(comparison_df['strategy'], rotation=45, ha='right')
        axes[1, 0].set_ylabel('Win Rate (%)')
        axes[1, 0].set_title('Win Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Excess Return vs Buy-Hold
        axes[1, 1].bar(range(len(comparison_df)), comparison_df['excess_return'], color='purple', alpha=0.6)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].set_xticks(range(len(comparison_df)))
        axes[1, 1].set_xticklabels(comparison_df['strategy'], rotation=45, ha='right')
        axes[1, 1].set_ylabel('Excess Return (%)')
        axes[1, 1].set_title('Return vs Buy & Hold')
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        filepath = self.output_dir / 'signal_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved: {filepath}")

    def plot_regression_diagnostics(self,
                                    y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    title: str = 'Regression Diagnostics'):
        """
        Plot regression diagnostic charts

        Args:
            y_true: Actual values
            y_pred: Predicted values
            title: Chart title
        """
        logger.info("Creating regression diagnostics...")

        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Predicted vs Actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=10)
        axes[0, 0].plot([y_true.min(), y_true.max()],
                        [y_true.min(), y_true.max()],
                        'r--', lw=2, label='Perfect Fit')
        axes[0, 0].set_xlabel('Actual')
        axes[0, 0].set_ylabel('Predicted')
        axes[0, 0].set_title('Predicted vs Actual')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Residuals vs Predicted
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)

        # Residuals Distribution
        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residuals Distribution')
        axes[1, 0].grid(True, alpha=0.3)

        # Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot')
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        filepath = self.output_dir / 'regression_diagnostics.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved: {filepath}")

    def plot_rolling_metrics(self,
                             df: pd.DataFrame,
                             metric_col: str,
                             title: str,
                             ylabel: str):
        """
        Plot rolling metric over time

        Args:
            df: DataFrame with timestamp and metric
            metric_col: Column name for metric
            title: Chart title
            ylabel: Y-axis label
        """
        logger.info(f"Creating rolling {metric_col} chart...")

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(df['timestamp'], df[metric_col], color='blue', linewidth=1.5)
        ax.fill_between(df['timestamp'], df[metric_col], alpha=0.3)

        # Add mean line
        mean_val = df[metric_col].mean()
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1,
                   label=f'Mean: {mean_val:.2f}')

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / f'rolling_{metric_col}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved: {filepath}")

    def plot_weight_evolution(self,
                              weights_df: pd.DataFrame,
                              title: str = 'Weight Evolution Over Time'):
        """
        Plot how regression weights change over time (Mode 2)

        Args:
            weights_df: DataFrame with weights over time
            title: Chart title
        """
        logger.info("Creating weight evolution chart...")

        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot each feature's weight over time
        weight_cols = [col for col in weights_df.columns if col not in ['iteration', 'date']]

        for col in weight_cols:
            ax.plot(weights_df['date'], weights_df[col], marker='o', label=col, linewidth=2)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Weight', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / 'weight_evolution.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved: {filepath}")

    def create_dashboard(self,
                         backtest_results: Dict,
                         performance_metrics: Dict):
        """
        Create comprehensive dashboard with multiple charts

        Args:
            backtest_results: Results from backtesting
            performance_metrics: Performance metrics
        """
        logger.info("Creating comprehensive dashboard...")

        df = backtest_results['data']

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Equity Curve
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df['timestamp'], df['cum_strategy_returns'], label='Strategy', linewidth=2)
        ax1.plot(df['timestamp'], df['cum_buy_hold'], label='Buy & Hold', linewidth=2, linestyle='--')
        ax1.set_title('Equity Curve', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, :])
        cum_returns = df['cum_strategy_returns']
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max * 100
        ax2.fill_between(df['timestamp'], 0, drawdown, color='red', alpha=0.3)
        ax2.plot(df['timestamp'], drawdown, color='darkred', linewidth=1)
        ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Returns Distribution
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.hist(df['strategy_returns_net'].dropna() * 100, bins=50, edgecolor='black', alpha=0.7)
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.set_title('Returns Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Return (%)')
        ax3.grid(True, alpha=0.3)

        # 4. Monthly Returns Heatmap (if enough data)
        ax4 = fig.add_subplot(gs[2, 1])
        try:
            monthly_returns = df.set_index('timestamp')['strategy_returns_net'].resample('M').apply(
                lambda x: (1 + x).prod() - 1
            ) * 100
            if len(monthly_returns) > 12:
                monthly_pivot = monthly_returns.to_frame()
                monthly_pivot['Year'] = monthly_pivot.index.year
                monthly_pivot['Month'] = monthly_pivot.index.month
                monthly_pivot = monthly_pivot.pivot(index='Year', columns='Month', values='strategy_returns_net')
                sns.heatmap(monthly_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                            cbar_kws={'label': 'Return (%)'}, ax=ax4)
                ax4.set_title('Monthly Returns', fontsize=12, fontweight='bold')
            else:
                ax4.text(0.5, 0.5, 'Insufficient data\nfor monthly heatmap',
                         ha='center', va='center', transform=ax4.transAxes)
                ax4.axis('off')
        except:
            ax4.text(0.5, 0.5, 'Monthly heatmap\nnot available',
                     ha='center', va='center', transform=ax4.transAxes)
            ax4.axis('off')

        # 5. Key Metrics Table
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.axis('off')

        metrics_text = f"""
        KEY METRICS
        ───────────────────
        Total Return: {performance_metrics.get('total_return_pct', 0):.2f}%
        CAGR: {performance_metrics.get('cagr_pct', 0):.2f}%
        Sharpe: {performance_metrics.get('sharpe_ratio', 0):.2f}
        Sortino: {performance_metrics.get('sortino_ratio', 0):.2f}
        Max DD: {performance_metrics.get('max_drawdown_pct', 0):.2f}%
        Win Rate: {performance_metrics.get('win_rate_pct', 0):.1f}%
        Profit Factor: {performance_metrics.get('profit_factor', 0):.2f}
        Total Trades: {performance_metrics.get('total_trades', 0)}
        """

        ax5.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
                 verticalalignment='center')

        plt.suptitle('Trading Strategy Dashboard', fontsize=16, fontweight='bold')

        filepath = self.output_dir / 'dashboard.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved dashboard: {filepath}")