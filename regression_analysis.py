"""
Rolling Regression Analysis Module
Implements Mode 1 (train/test split) and Mode 2 (rolling walk-forward)
Supports multiple regression types: linear, polynomial, spline, logistic
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from scipy.interpolate import UnivariateSpline
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RollingRegressionAnalyzer:
    """
    Rolling regression analysis with two modes:

    MODE 1: Simple train/test split (60/40)
    MODE 2: Rolling walk-forward (6-year training, 1-month rebalance)
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with feature dataframe

        Args:
            df: DataFrame with features and target (future returns)
        """
        self.df = df.copy()
        self.results = {}
        self.scaler = StandardScaler()

    def prepare_data(self, feature_cols: List[str], target_col: str = 'return_1_ahead'):
        """
        Prepare features and target for regression

        Args:
            feature_cols: List of feature column names
            target_col: Target variable name
        """
        # Create target if it doesn't exist
        if target_col not in self.df.columns:
            self.df['return_1_ahead'] = self.df['price'].pct_change().shift(-1) * 100

        # Remove rows with NaN in features or target
        self.df = self.df.dropna(subset=feature_cols + [target_col])

        self.feature_cols = feature_cols
        self.target_col = target_col

        logger.info(f"Prepared data: {len(self.df)} observations, {len(feature_cols)} features")

    def mode1_train_test_split(self,
                               regression_type: str = 'linear',
                               train_pct: float = 0.6) -> Dict:
        """
        MODE 1: Simple train/test split

        Train on first 60%, test on last 40%
        Weights locked after training

        Args:
            regression_type: 'linear', 'polynomial', 'spline', 'logistic'
            train_pct: Percentage of data for training (default 0.6)

        Returns:
            Dictionary with results and locked weights
        """
        logger.info(f"MODE 1: Train/Test Split ({train_pct * 100:.0f}/{(1 - train_pct) * 100:.0f})")
        logger.info(f"Regression type: {regression_type}")

        # Split data
        split_idx = int(len(self.df) * train_pct)

        train_data = self.df.iloc[:split_idx]
        test_data = self.df.iloc[split_idx:]

        logger.info(f"Train: {len(train_data)} obs, Test: {len(test_data)} obs")

        # Prepare features and target
        X_train = train_data[self.feature_cols].values
        y_train = train_data[self.target_col].values
        X_test = test_data[self.feature_cols].values
        y_test = test_data[self.target_col].values

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        model = self._get_model(regression_type)
        model.fit(X_train_scaled, y_train)

        # Get predictions
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)

        # Calculate metrics
        train_r2 = self._calculate_r2(y_train, train_pred)
        test_r2 = self._calculate_r2(y_test, test_pred)
        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
        test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))

        # Extract weights (if linear)
        weights = self._extract_weights(model, regression_type)

        results = {
            'mode': 'MODE_1_TRAIN_TEST',
            'regression_type': regression_type,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'weights': weights,
            'model': model,
            'train_predictions': train_pred,
            'test_predictions': test_pred,
            'feature_importance': self._get_feature_importance(weights, self.feature_cols)
        }

        logger.info(f"✓ Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        logger.info(f"✓ Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

        self.results['mode1'] = results
        return results

    def mode2_rolling_walkforward(self,
                                  regression_type: str = 'linear',
                                  train_window_years: int = 6,
                                  rebalance_period_months: int = 1) -> Dict:
        """
        MODE 2: Rolling walk-forward analysis

        Process:
        1. Train on 6 years of data
        2. Lock weights for 1 month
        3. After 1 month, retrain including that month
        4. Lock new weights for next month
        5. Continue rolling forward

        Args:
            regression_type: 'linear', 'polynomial', 'spline', 'logistic'
            train_window_years: Years of data for training (default 6)
            rebalance_period_months: How often to update weights (default 1 month)

        Returns:
            Dictionary with rolling results
        """
        logger.info(f"MODE 2: Rolling Walk-Forward Analysis")
        logger.info(f"Train window: {train_window_years} years, Rebalance: {rebalance_period_months} month(s)")
        logger.info(f"Regression type: {regression_type}")

        # Ensure data has datetime index
        self.df['date'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.set_index('date')

        # Calculate window sizes
        train_window = pd.DateOffset(years=train_window_years)
        rebalance_freq = pd.DateOffset(months=rebalance_period_months)

        # Get data range
        start_date = self.df.index.min()
        end_date = self.df.index.max()

        # First training period
        initial_train_end = start_date + train_window

        if initial_train_end >= end_date:
            logger.error("Not enough data for walk-forward analysis")
            return {}

        # Initialize results storage
        rolling_results = []
        all_predictions = []
        weight_evolution = []

        # Rolling walk-forward loop
        current_date = initial_train_end
        iteration = 0

        while current_date < end_date:
            iteration += 1

            # Define training period (expanding window)
            train_data = self.df.loc[start_date:current_date]

            # Define prediction period (next rebalance_period)
            pred_start = current_date
            pred_end = current_date + rebalance_freq
            pred_data = self.df.loc[pred_start:pred_end]

            if len(pred_data) == 0:
                break

            logger.info(f"  Iteration {iteration}: Train={len(train_data)}, Predict={len(pred_data)}")

            # Prepare data
            X_train = train_data[self.feature_cols].values
            y_train = train_data[self.target_col].values
            X_pred = pred_data[self.feature_cols].values
            y_true = pred_data[self.target_col].values

            # Scale features (fit on train only!)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_pred_scaled = scaler.transform(X_pred)

            # Train model with current data
            model = self._get_model(regression_type)
            model.fit(X_train_scaled, y_train)

            # Lock weights and predict next period
            predictions = model.predict(X_pred_scaled)

            # Extract weights
            weights = self._extract_weights(model, regression_type)

            # Calculate metrics for this period
            period_r2 = self._calculate_r2(y_true, predictions)
            period_rmse = np.sqrt(np.mean((y_true - predictions) ** 2))

            # Store results
            rolling_results.append({
                'iteration': iteration,
                'train_start': start_date,
                'train_end': current_date,
                'pred_start': pred_start,
                'pred_end': pred_end,
                'train_size': len(train_data),
                'pred_size': len(pred_data),
                'r2': period_r2,
                'rmse': period_rmse
            })

            # Store predictions with timestamps
            for i, (idx, pred, true) in enumerate(zip(pred_data.index, predictions, y_true)):
                all_predictions.append({
                    'timestamp': idx,
                    'prediction': pred,
                    'actual': true,
                    'iteration': iteration
                })

            # Store weight evolution
            if weights is not None:
                weight_dict = {'iteration': iteration, 'date': current_date}
                weight_dict.update(weights)
                weight_evolution.append(weight_dict)

            # Move to next period
            current_date = pred_end

        # Aggregate results
        results_df = pd.DataFrame(rolling_results)
        predictions_df = pd.DataFrame(all_predictions)
        weights_df = pd.DataFrame(weight_evolution)

        # Overall metrics
        overall_r2 = self._calculate_r2(predictions_df['actual'].values,
                                        predictions_df['prediction'].values)
        overall_rmse = np.sqrt(np.mean((predictions_df['actual'] - predictions_df['prediction']) ** 2))

        results = {
            'mode': 'MODE_2_ROLLING_WALKFORWARD',
            'regression_type': regression_type,
            'train_window_years': train_window_years,
            'rebalance_months': rebalance_period_months,
            'num_iterations': iteration,
            'overall_r2': overall_r2,
            'overall_rmse': overall_rmse,
            'rolling_results': results_df,
            'all_predictions': predictions_df,
            'weight_evolution': weights_df,
            'avg_period_r2': results_df['r2'].mean(),
            'std_period_r2': results_df['r2'].std()
        }

        logger.info(f"✓ Completed {iteration} iterations")
        logger.info(f"✓ Overall R²: {overall_r2:.4f}, RMSE: {overall_rmse:.4f}")
        logger.info(f"✓ Avg period R²: {results['avg_period_r2']:.4f} ± {results['std_period_r2']:.4f}")

        self.results['mode2'] = results
        return results

    def univariate_regression_analysis(self, regression_type: str = 'linear') -> pd.DataFrame:
        """
        Regress each feature individually against target
        Shows predictive power of each feature in isolation

        Args:
            regression_type: Type of regression to use

        Returns:
            DataFrame with results for each feature
        """
        logger.info("Running Univariate Regression Analysis...")
        logger.info(f"Testing {len(self.feature_cols)} features individually")

        results = []

        X_full = self.df[self.feature_cols].values
        y_full = self.df[self.target_col].values

        for i, feature_name in enumerate(self.feature_cols):
            # Single feature
            X_single = X_full[:, i].reshape(-1, 1)

            # Scale
            X_scaled = self.scaler.fit_transform(X_single)

            # Train model
            model = self._get_model(regression_type)
            model.fit(X_scaled, y_full)

            # Predictions
            y_pred = model.predict(X_scaled)

            # Metrics
            r2 = self._calculate_r2(y_full, y_pred)
            rmse = np.sqrt(np.mean((y_full - y_pred) ** 2))

            # Correlation
            correlation = np.corrcoef(X_full[:, i], y_full)[0, 1]

            # Extract coefficient (if linear)
            if regression_type == 'linear' and hasattr(model, 'coef_'):
                coefficient = model.coef_[0]
            else:
                coefficient = np.nan

            results.append({
                'feature': feature_name,
                'r2': r2,
                'rmse': rmse,
                'correlation': correlation,
                'coefficient': coefficient,
                'abs_correlation': abs(correlation)
            })

            logger.info(f"  {feature_name}: R²={r2:.4f}, Corr={correlation:.4f}")

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('abs_correlation', ascending=False)

        self.results['univariate'] = results_df

        logger.info(f"✓ Univariate analysis complete")
        logger.info(f"  Top 3 features: {results_df.head(3)['feature'].tolist()}")

        return results_df

    def _get_model(self, regression_type: str):
        """Get regression model based on type"""

        if regression_type == 'linear':
            return Ridge(alpha=1.0)  # Ridge to handle multicollinearity

        elif regression_type == 'polynomial':
            return Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('ridge', Ridge(alpha=1.0))
            ])

        elif regression_type == 'spline':
            # Note: Spline requires special handling, use Ridge as fallback
            logger.warning("Spline regression not fully implemented, using Ridge")
            return Ridge(alpha=1.0)

        elif regression_type == 'logistic':
            # Convert target to binary (positive/negative returns)
            return LogisticRegression(max_iter=1000)

        else:
            logger.warning(f"Unknown regression type: {regression_type}, using linear")
            return Ridge(alpha=1.0)

    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    def _extract_weights(self, model, regression_type: str) -> Optional[Dict]:
        """Extract weights from trained model"""

        if regression_type == 'linear':
            if hasattr(model, 'coef_'):
                return {f: w for f, w in zip(self.feature_cols, model.coef_)}

        elif regression_type == 'polynomial':
            if hasattr(model.named_steps['ridge'], 'coef_'):
                # Polynomial creates more features, return original feature weights only
                coefs = model.named_steps['ridge'].coef_
                return {f: coefs[i] for i, f in enumerate(self.feature_cols) if i < len(coefs)}

        return None

    def _get_feature_importance(self, weights: Optional[Dict], feature_cols: List[str]) -> pd.DataFrame:
        """Calculate feature importance from weights"""

        if weights is None:
            return pd.DataFrame()

        importance = []
        for feature, weight in weights.items():
            importance.append({
                'feature': feature,
                'weight': weight,
                'abs_weight': abs(weight)
            })

        df = pd.DataFrame(importance)
        df = df.sort_values('abs_weight', ascending=False)

        return df

    def compare_regression_methods(self, methods: List[str] = ['linear', 'polynomial']) -> pd.DataFrame:
        """
        Compare different regression methods

        Args:
            methods: List of regression types to compare

        Returns:
            Comparison DataFrame
        """
        logger.info(f"Comparing regression methods: {methods}")

        comparison = []

        for method in methods:
            logger.info(f"\n  Testing {method} regression...")

            # Mode 1 results
            results = self.mode1_train_test_split(regression_type=method, train_pct=0.6)

            comparison.append({
                'method': method,
                'train_r2': results['train_r2'],
                'test_r2': results['test_r2'],
                'train_rmse': results['train_rmse'],
                'test_rmse': results['test_rmse'],
                'r2_degradation': results['train_r2'] - results['test_r2']
            })

        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('test_r2', ascending=False)

        logger.info("\n✓ Method comparison complete")
        logger.info(
            f"\nBest method: {comparison_df.iloc[0]['method']} (Test R²: {comparison_df.iloc[0]['test_r2']:.4f})")

        return comparison_df