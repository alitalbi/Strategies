"""
Performance Metrics Module
Comprehensive risk and return analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Calculate comprehensive performance metrics

    Metrics:
    - Return metrics: Total, CAGR, annualized
    - Risk metrics: Volatility, Sharpe, Sortino, Calmar
    - Drawdown analysis: Max DD, duration, recovery
    - Trade metrics: Win rate, profit factor, avg win/loss
    - Risk metrics: VaR, CVaR
    """

    def __init__(self, annual_factor: int = 252):
        """
        Initialize analyzer

        Args:
            annual_factor: Trading days per year (default 252)
        """
        self.annual_factor = annual_factor

    def calculate_all_metrics(self,
                              returns: pd.Series,
                              trades: Optional[pd.DataFrame] = None,
                              benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """
        Calculate comprehensive performance metrics

        Args:
            returns: Series of strategy returns
            trades: DataFrame of individual trades (optional)
            benchmark_returns: Benchmark returns for comparison (optional)

        Returns:
            Dictionary with all metrics
        """
        logger.info("Calculating performance metrics...")

        metrics = {}

        # Return metrics
        metrics.update(self._calculate_return_metrics(returns))

        # Risk metrics
        metrics.update(self._calculate_risk_metrics(returns))

        # Drawdown metrics
        metrics.update(self._calculate_drawdown_metrics(returns))

        # Trade metrics (if trades provided)
        if trades is not None and len(trades) > 0:
            metrics.update(self._calculate_trade_metrics(trades))

        # Benchmark comparison (if provided)
        if benchmark_returns is not None:
            metrics.update(self._calculate_benchmark_metrics(returns, benchmark_returns))

        logger.info("✓ Performance metrics calculated")

        return metrics

    def _calculate_return_metrics(self, returns: pd.Series) -> Dict:
        """Calculate return-based metrics"""

        # Total return
        cum_returns = (1 + returns).cumprod()
        total_return = (cum_returns.iloc[-1] - 1) * 100

        # Annualized return (CAGR)
        n_periods = len(returns)
        n_years = n_periods / self.annual_factor
        cagr = ((cum_returns.iloc[-1]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

        # Average return
        avg_return = returns.mean() * 100
        avg_return_annual = avg_return * self.annual_factor

        # Median return
        median_return = returns.median() * 100

        return {
            'total_return_pct': total_return,
            'cagr_pct': cagr,
            'avg_return_pct': avg_return,
            'avg_return_annual_pct': avg_return_annual,
            'median_return_pct': median_return,
        }

    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict:
        """Calculate risk-based metrics"""

        # Volatility
        volatility = returns.std()
        volatility_annual = volatility * np.sqrt(self.annual_factor)

        # Sharpe ratio (assuming 0 risk-free rate)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(self.annual_factor) if returns.std() > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino = (returns.mean() / downside_std) * np.sqrt(self.annual_factor) if downside_std > 0 else 0

        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # VaR and CVaR (95% confidence)
        var_95 = returns.quantile(0.05) * 100
        cvar_95 = returns[returns <= returns.quantile(0.05)].mean() * 100

        return {
            'volatility': volatility,
            'volatility_annual_pct': volatility_annual * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95_pct': var_95,
            'cvar_95_pct': cvar_95,
        }

    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict:
        """Calculate drawdown metrics"""

        # Cumulative returns
        cum_returns = (1 + returns).cumprod()

        # Running maximum
        running_max = cum_returns.expanding().max()

        # Drawdown
        drawdown = (cum_returns - running_max) / running_max

        # Max drawdown
        max_drawdown = drawdown.min() * 100
        max_dd_date = drawdown.idxmin()

        # Calmar ratio (CAGR / abs(max drawdown))
        n_years = len(returns) / self.annual_factor
        cagr = ((cum_returns.iloc[-1]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        # Drawdown duration
        dd_duration = self._calculate_drawdown_duration(drawdown)

        # Recovery time (if recovered)
        recovery_time = self._calculate_recovery_time(drawdown)

        return {
            'max_drawdown_pct': max_drawdown,
            'max_drawdown_date': max_dd_date,
            'calmar_ratio': calmar,
            'avg_drawdown_duration': dd_duration,
            'max_recovery_time': recovery_time,
        }

    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> float:
        """Calculate average drawdown duration"""

        # Identify drawdown periods
        in_drawdown = drawdown < 0

        # Group consecutive drawdowns
        drawdown_groups = (in_drawdown != in_drawdown.shift()).cumsum()

        # Calculate duration of each drawdown
        durations = []
        for group in drawdown_groups[in_drawdown].unique():
            duration = (in_drawdown & (drawdown_groups == group)).sum()
            durations.append(duration)

        return np.mean(durations) if durations else 0

    def _calculate_recovery_time(self, drawdown: pd.Series) -> float:
        """Calculate time to recover from max drawdown"""

        # Find max drawdown point
        max_dd_idx = drawdown.idxmin()

        # Find recovery (first time drawdown returns to 0)
        after_max_dd = drawdown.loc[max_dd_idx:]
        recovery_idx = after_max_dd[after_max_dd >= 0].first_valid_index()

        if recovery_idx is not None:
            recovery_time = (after_max_dd.index.get_loc(recovery_idx) -
                             after_max_dd.index.get_loc(max_dd_idx))
            return recovery_time
        else:
            # Not yet recovered
            return len(after_max_dd)

    def _calculate_trade_metrics(self, trades: pd.DataFrame) -> Dict:
        """Calculate trade-based metrics"""

        if len(trades) == 0:
            return {}

        # Win rate
        winning_trades = len(trades[trades['pnl'] > 0])
        losing_trades = len(trades[trades['pnl'] < 0])
        total_trades = len(trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Average win/loss
        avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades[trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

        # Profit factor
        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Average trade duration
        avg_duration = trades['duration_hours'].mean()

        # Best and worst trade
        best_trade = trades['pnl'].max()
        worst_trade = trades['pnl'].min()

        # Consecutive wins/losses
        trades['win'] = (trades['pnl'] > 0).astype(int)
        trades['win_streak'] = trades['win'].groupby((trades['win'] != trades['win'].shift()).cumsum()).cumsum()
        trades['loss_streak'] = (1 - trades['win']).groupby((trades['win'] != trades['win'].shift()).cumsum()).cumsum()

        max_win_streak = trades['win_streak'].max()
        max_loss_streak = trades['loss_streak'].max()

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'avg_trade_duration_hours': avg_duration,
            'best_trade_pct': best_trade,
            'worst_trade_pct': worst_trade,
            'max_win_streak': int(max_win_streak),
            'max_loss_streak': int(max_loss_streak),
        }

    def _calculate_benchmark_metrics(self,
                                     returns: pd.Series,
                                     benchmark_returns: pd.Series) -> Dict:
        """Calculate metrics relative to benchmark"""

        # Align series
        common_index = returns.index.intersection(benchmark_returns.index)
        returns_aligned = returns.loc[common_index]
        benchmark_aligned = benchmark_returns.loc[common_index]

        # Excess returns
        excess_returns = returns_aligned - benchmark_aligned

        # Information ratio
        tracking_error = excess_returns.std() * np.sqrt(self.annual_factor)
        information_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(
            self.annual_factor) if excess_returns.std() > 0 else 0

        # Beta
        covariance = np.cov(returns_aligned, benchmark_aligned)[0, 1]
        benchmark_variance = benchmark_aligned.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

        # Alpha (Jensen's alpha)
        avg_excess_return = excess_returns.mean() * self.annual_factor

        return {
            'information_ratio': information_ratio,
            'tracking_error_pct': tracking_error * 100,
            'beta': beta,
            'alpha_annual_pct': avg_excess_return * 100,
        }

    def create_metrics_report(self, metrics: Dict) -> str:
        """Create formatted report of all metrics"""

        report = []
        report.append("\n" + "=" * 70)
        report.append("PERFORMANCE METRICS REPORT")
        report.append("=" * 70)

        # Return Metrics
        report.append("\n📈 RETURN METRICS")
        report.append("-" * 70)
        report.append(f"Total Return:          {metrics.get('total_return_pct', 0):>10.2f}%")
        report.append(f"CAGR:                  {metrics.get('cagr_pct', 0):>10.2f}%")
        report.append(f"Avg Annual Return:     {metrics.get('avg_return_annual_pct', 0):>10.2f}%")

        # Risk Metrics
        report.append("\n⚠️  RISK METRICS")
        report.append("-" * 70)
        report.append(f"Annual Volatility:     {metrics.get('volatility_annual_pct', 0):>10.2f}%")
        report.append(f"Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):>10.2f}")
        report.append(f"Sortino Ratio:         {metrics.get('sortino_ratio', 0):>10.2f}")
        report.append(f"Calmar Ratio:          {metrics.get('calmar_ratio', 0):>10.2f}")
        report.append(f"Max Drawdown:          {metrics.get('max_drawdown_pct', 0):>10.2f}%")
        report.append(f"VaR (95%):             {metrics.get('var_95_pct', 0):>10.2f}%")
        report.append(f"CVaR (95%):            {metrics.get('cvar_95_pct', 0):>10.2f}%")

        # Trade Metrics
        if 'total_trades' in metrics:
            report.append("\n📊 TRADE METRICS")
            report.append("-" * 70)
            report.append(f"Total Trades:          {metrics.get('total_trades', 0):>10}")
            report.append(f"Win Rate:              {metrics.get('win_rate_pct', 0):>10.2f}%")
            report.append(f"Profit Factor:         {metrics.get('profit_factor', 0):>10.2f}")
            report.append(f"Avg Win:               {metrics.get('avg_win_pct', 0):>10.2f}%")
            report.append(f"Avg Loss:              {metrics.get('avg_loss_pct', 0):>10.2f}%")
            report.append(f"Best Trade:            {metrics.get('best_trade_pct', 0):>10.2f}%")
            report.append(f"Worst Trade:           {metrics.get('worst_trade_pct', 0):>10.2f}%")
            report.append(f"Max Win Streak:        {metrics.get('max_win_streak', 0):>10}")
            report.append(f"Max Loss Streak:       {metrics.get('max_loss_streak', 0):>10}")

        report.append("\n" + "=" * 70 + "\n")

        return "\n".join(report)


class RollingMetrics:
    """Calculate metrics on rolling windows"""

    def __init__(self, window: int = 252):
        self.window = window

    def calculate_rolling_sharpe(self, returns: pd.Series) -> pd.Series:
        """Calculate rolling Sharpe ratio"""
        rolling_mean = returns.rolling(window=self.window).mean()
        rolling_std = returns.rolling(window=self.window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        return rolling_sharpe

    def calculate_rolling_volatility(self, returns: pd.Series) -> pd.Series:
        """Calculate rolling volatility"""
        return returns.rolling(window=self.window).std() * np.sqrt(252)

    def calculate_rolling_drawdown(self, returns: pd.Series) -> pd.Series:
        """Calculate rolling maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.rolling(window=self.window, min_periods=1).max()
        rolling_dd = (cum_returns - rolling_max) / rolling_max
        return rolling_dd