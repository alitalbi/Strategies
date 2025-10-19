"""
Backtesting Engine
Vectorized backtesting with transaction costs, slippage, and detailed trade logging
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Backtester:
    """
    Comprehensive backtesting engine for trading signals

    Features:
    - Vectorized performance calculation
    - Transaction costs and slippage
    - Position tracking
    - Trade-by-trade logging
    - Multiple signal support
    """

    def __init__(self,
                 transaction_cost_bps: float = 1.0,
                 slippage_bps: float = 0.5):
        """
        Initialize backtester

        Args:
            transaction_cost_bps: Transaction cost in basis points (default 1.0)
            slippage_bps: Slippage in basis points (default 0.5)
        """
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.total_cost_bps = transaction_cost_bps + slippage_bps

        self.results = {}

    def backtest_signal(self,
                        df: pd.DataFrame,
                        signal_col: str,
                        signal_name: str = 'strategy') -> Dict:
        """
        Run backtest for a single signal

        Args:
            df: DataFrame with price and signal columns
            signal_col: Name of signal column
            signal_name: Name for this strategy

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Backtesting signal: {signal_name}")

        df = df.copy()

        # Ensure we have required columns
        if 'returns' not in df.columns:
            df['returns'] = df['price'].pct_change()

        if signal_col not in df.columns:
            logger.error(f"Signal column '{signal_col}' not found")
            return {}

        # Calculate strategy returns
        df['position'] = df[signal_col].shift(1)  # Use previous signal (no lookahead)
        df['strategy_returns'] = df['position'] * df['returns']

        # Calculate transaction costs
        df['position_change'] = df['position'].diff().abs()
        df['transaction_costs'] = df['position_change'] * (self.total_cost_bps / 10000)
        df['strategy_returns_net'] = df['strategy_returns'] - df['transaction_costs']

        # Calculate cumulative returns
        df['cum_returns'] = (1 + df['returns']).cumprod()
        df['cum_strategy_returns'] = (1 + df['strategy_returns_net']).cumprod()

        # Buy and hold baseline
        df['buy_hold_returns'] = df['returns']
        df['cum_buy_hold'] = (1 + df['buy_hold_returns']).cumprod()

        # Generate trade log
        trades = self._generate_trade_log(df, signal_col)

        # Calculate performance metrics (will be detailed in performance_metrics.py)
        total_return = (df['cum_strategy_returns'].iloc[-1] - 1) * 100
        total_return_bh = (df['cum_buy_hold'].iloc[-1] - 1) * 100

        # Count trades
        num_trades = int(df['position_change'].sum() / 2)  # Divide by 2 (entry and exit)

        # Calculate win rate
        winning_trades = len(trades[trades['pnl'] > 0]) if len(trades) > 0 else 0
        total_trades = len(trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        results = {
            'signal_name': signal_name,
            'signal_col': signal_col,
            'data': df,
            'trades': trades,
            'num_trades': num_trades,
            'total_return': total_return,
            'total_return_bh': total_return_bh,
            'excess_return': total_return - total_return_bh,
            'win_rate': win_rate,
            'total_costs': df['transaction_costs'].sum() * 100,
        }

        logger.info(f"✓ Backtest complete: {num_trades} trades, {total_return:.2f}% return")

        self.results[signal_name] = results
        return results

    def backtest_multiple_signals(self,
                                  df: pd.DataFrame,
                                  signal_configs: List[Dict]) -> Dict:
        """
        Backtest multiple signals for comparison

        Args:
            df: DataFrame with price and multiple signal columns
            signal_configs: List of dicts with 'signal_col' and 'signal_name'

        Returns:
            Dictionary with all results
        """
        logger.info(f"Backtesting {len(signal_configs)} signals...")

        all_results = {}

        for config in signal_configs:
            signal_col = config['signal_col']
            signal_name = config['signal_name']

            results = self.backtest_signal(df, signal_col, signal_name)
            all_results[signal_name] = results

        # Create comparison summary
        comparison = self._create_comparison_summary(all_results)
        all_results['comparison'] = comparison

        logger.info("✓ Multi-signal backtest complete")

        return all_results

    def _generate_trade_log(self, df: pd.DataFrame, signal_col: str) -> pd.DataFrame:
        """
        Generate detailed trade-by-trade log

        Each trade includes entry, exit, duration, and P&L
        """
        trades = []

        position = 0
        entry_idx = None
        entry_price = None
        entry_time = None

        for idx, row in df.iterrows():
            current_signal = row['position']  # Already shifted

            # Entry
            if position == 0 and current_signal != 0:
                position = current_signal
                entry_idx = idx
                entry_price = row['price']
                entry_time = row['timestamp']

            # Exit
            elif position != 0 and (current_signal == 0 or current_signal != position):
                exit_price = row['price']
                exit_time = row['timestamp']

                # Calculate P&L
                if position == 1:  # Long
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                else:  # Short
                    pnl_pct = (entry_price - exit_price) / entry_price * 100

                # Subtract costs
                pnl_pct -= self.total_cost_bps / 100

                # Duration
                duration = (exit_time - entry_time).total_seconds() / 3600  # Hours

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'duration_hours': duration,
                    'pnl': pnl_pct,
                    'cost': self.total_cost_bps / 100
                })

                # Update position
                if current_signal != 0:
                    # Immediate reversal
                    position = current_signal
                    entry_idx = idx
                    entry_price = exit_price
                    entry_time = exit_time
                else:
                    position = 0

        trades_df = pd.DataFrame(trades)
        return trades_df

    def _create_comparison_summary(self, all_results: Dict) -> pd.DataFrame:
        """Create comparison table for multiple signals"""

        comparison = []

        for signal_name, results in all_results.items():
            comparison.append({
                'strategy': signal_name,
                'total_return': results['total_return'],
                'excess_return': results['excess_return'],
                'num_trades': results['num_trades'],
                'win_rate': results['win_rate'],
                'total_costs': results['total_costs']
            })

        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('total_return', ascending=False)

        return comparison_df

    def get_equity_curve(self, signal_name: str) -> pd.DataFrame:
        """Get equity curve for a specific signal"""

        if signal_name not in self.results:
            logger.error(f"Signal '{signal_name}' not found in results")
            return pd.DataFrame()

        data = self.results[signal_name]['data']

        equity = data[['timestamp', 'cum_strategy_returns', 'cum_buy_hold']].copy()
        equity.columns = ['timestamp', 'strategy', 'buy_hold']

        return equity

    def get_all_trades(self, signal_name: str) -> pd.DataFrame:
        """Get all trades for a specific signal"""

        if signal_name not in self.results:
            logger.error(f"Signal '{signal_name}' not found in results")
            return pd.DataFrame()

        return self.results[signal_name]['trades']


class WalkForwardBacktester:
    """
    Walk-forward backtesting that integrates with regression analysis
    Tests strategy with rolling weight updates
    """

    def __init__(self, backtester: Backtester):
        self.backtester = backtester

    def backtest_with_rolling_weights(self,
                                      df: pd.DataFrame,
                                      weight_evolution: pd.DataFrame,
                                      signal_cols: List[str]) -> Dict:
        """
        Backtest using weights that change over time (from Mode 2 regression)

        Args:
            df: Full dataset with features
            weight_evolution: DataFrame with weights over time
            signal_cols: List of signal columns to combine

        Returns:
            Backtest results
        """
        logger.info("Running walk-forward backtest with rolling weights...")

        df = df.copy()

        # For each period, apply the locked weights
        df['combined_signal_wf'] = 0

        for idx, weight_row in weight_evolution.iterrows():
            iteration = weight_row['iteration']
            date = weight_row['date']

            # Get weights for this period
            weights = {col: weight_row.get(col, 0) for col in signal_cols if col in weight_row}

            # Apply weights to data in this period
            # This is simplified - in practice, match dates more carefully
            mask = df['timestamp'] >= date
            if idx < len(weight_evolution) - 1:
                next_date = weight_evolution.iloc[idx + 1]['date']
                mask = mask & (df['timestamp'] < next_date)

            # Calculate combined signal for this period
            combined = sum(df.loc[mask, col] * weights.get(col, 0) for col in signal_cols if col in df.columns)
            df.loc[mask, 'combined_signal_wf'] = combined

        # Convert to discrete signal
        df['signal_wf'] = 0
        df.loc[df['combined_signal_wf'] > 0.5, 'signal_wf'] = 1
        df.loc[df['combined_signal_wf'] < -0.5, 'signal_wf'] = -1

        # Run backtest
        results = self.backtester.backtest_signal(df, 'signal_wf', 'walk_forward_strategy')

        logger.info("✓ Walk-forward backtest complete")

        return results