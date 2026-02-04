"""
Backtesting module for FX rate forecasting models.

Provides rolling window backtesting with non-overlapping windows,
distributional tests, and percentile-based tests.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from scipy import stats
from scipy.stats import (
    kstest, normaltest, jarque_bera, anderson,
    percentileofscore, rankdata
)
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestResult:
    """Container for backtest results."""
    model_name: str
    window_results: List[Dict]
    distributional_tests: Dict[str, Dict]
    percentile_tests: Dict[str, Dict]
    summary_stats: Dict[str, float]


class RollingWindowBacktester:
    """
    Rolling window backtesting with non-overlapping windows.
    """
    
    def __init__(
        self,
        window_size: int = 60,
        step_size: Optional[int] = None,
        min_train_size: int = 252
    ):
        """
        Initialize backtester.
        
        Args:
            window_size: Size of each test window (forecast horizon)
            step_size: Step between windows (if None, equals window_size for non-overlapping)
            min_train_size: Minimum training data required
        """
        self.window_size = window_size
        self.step_size = step_size or window_size  # Non-overlapping by default
        self.min_train_size = min_train_size
    
    def generate_windows(
        self,
        data: pd.Series
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Generate non-overlapping train/test windows.
        
        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        windows = []
        n = len(data)
        
        # Ensure index is timezone-naive to avoid offset issues
        if hasattr(data.index, 'tz') and data.index.tz is not None:
            data = data.copy()
            data.index = data.index.tz_localize(None)
        
        # Start first test window after minimum training data
        first_test_start = self.min_train_size
        
        for i in range(first_test_start, n - self.window_size + 1, self.step_size):
            train_start = data.index[0]
            train_end = data.index[i - 1]
            test_start = data.index[i]
            test_end = data.index[min(i + self.window_size - 1, n - 1)]
            
            windows.append((train_start, train_end, test_start, test_end))
        
        return windows
    
    def run_backtest(
        self,
        model_factory: Callable,
        data: pd.Series,
        model_name: str = "Model"
    ) -> BacktestResult:
        """
        Run rolling window backtest for a model.
        
        Args:
            model_factory: Function that creates and fits a model
            data: Time series data
            model_name: Name of the model
        
        Returns:
            BacktestResult with all window results
        """
        # Ensure timezone-naive index
        if hasattr(data.index, 'tz') and data.index.tz is not None:
            data = data.copy()
            data.index = data.index.tz_localize(None)
        
        windows = self.generate_windows(data)
        window_results = []
        
        print(f"Running backtest for {model_name}...")
        print(f"  Number of windows: {len(windows)}")
        print(f"  Window size: {self.window_size}")
        print(f"  Step size: {self.step_size}")
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            # Get train/test data using integer indexing to avoid timezone issues
            # Find the actual positions in the index
            try:
                train_start_idx = data.index.get_loc(train_start)
                train_end_idx = data.index.get_loc(train_end)
                test_start_idx = data.index.get_loc(test_start)
                test_end_idx = data.index.get_loc(test_end)
            except KeyError:
                # Fall back to label-based indexing if exact match not found
                train_data = data.loc[:train_end].loc[train_start:]
                test_data = data.loc[test_start:test_end]
            else:
                train_data = data.iloc[train_start_idx:train_end_idx+1]
                test_data = data.iloc[test_start_idx:test_end_idx+1]
            
            if len(test_data) == 0:
                continue
            
            try:
                # Fit model and predict
                model = model_factory()
                model.fit(train_data)
                predictions = model.predict(len(test_data))
                
                # Calculate errors
                actuals = test_data.values[:len(predictions)]
                errors = actuals - predictions
                
                window_result = {
                    'window': i,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'actuals': actuals,
                    'predictions': predictions,
                    'errors': errors,
                    'mse': np.mean(errors ** 2),
                    'mae': np.mean(np.abs(errors)),
                    'rmse': np.sqrt(np.mean(errors ** 2)),
                    'mape': np.mean(np.abs(errors / actuals)) * 100 if np.all(actuals != 0) else np.nan,
                    'mean_error': np.mean(errors),
                    'std_error': np.std(errors),
                    'directional_accuracy': self._directional_accuracy(actuals, predictions)
                }
                
                window_results.append(window_result)
                
            except Exception as e:
                print(f"  Warning: Window {i} failed: {e}")
                continue
        
        # Run distributional and percentile tests
        if window_results:
            all_errors = np.concatenate([w['errors'] for w in window_results])
            all_actuals = np.concatenate([w['actuals'] for w in window_results])
            all_predictions = np.concatenate([w['predictions'] for w in window_results])
            
            distributional_tests = self._run_distributional_tests(all_errors)
            percentile_tests = self._run_percentile_tests(all_actuals, all_predictions)
            summary_stats = self._compute_summary_stats(window_results)
        else:
            distributional_tests = {}
            percentile_tests = {}
            summary_stats = {}
        
        return BacktestResult(
            model_name=model_name,
            window_results=window_results,
            distributional_tests=distributional_tests,
            percentile_tests=percentile_tests,
            summary_stats=summary_stats
        )
    
    def _directional_accuracy(self, actuals: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate directional accuracy."""
        if len(actuals) < 2 or len(predictions) < 2:
            return 0.0
        
        actual_direction = np.sign(np.diff(actuals))
        pred_direction = np.sign(np.diff(predictions))
        correct = np.sum(actual_direction == pred_direction)
        return (correct / len(actual_direction)) * 100 if len(actual_direction) > 0 else 0.0
    
    def _run_distributional_tests(self, errors: np.ndarray) -> Dict[str, Dict]:
        """
        Run distributional tests on prediction errors.
        """
        results = {}
        
        # Remove NaN values
        errors_clean = errors[~np.isnan(errors)]
        
        if len(errors_clean) < 8:
            return results
        
        # 1. Kolmogorov-Smirnov Test (against normal distribution)
        try:
            mean, std = np.mean(errors_clean), np.std(errors_clean)
            ks_stat, ks_pvalue = kstest(errors_clean, 'norm', args=(mean, std))
            results['kolmogorov_smirnov'] = {
                'statistic': ks_stat,
                'p_value': ks_pvalue,
                'is_normal': ks_pvalue > 0.05,
                'interpretation': 'Errors are normally distributed' if ks_pvalue > 0.05 else 'Errors deviate from normal'
            }
        except Exception as e:
            results['kolmogorov_smirnov'] = {'error': str(e)}
        
        # 2. D'Agostino-Pearson Test (normality)
        try:
            dp_stat, dp_pvalue = normaltest(errors_clean)
            results['dagostino_pearson'] = {
                'statistic': dp_stat,
                'p_value': dp_pvalue,
                'is_normal': dp_pvalue > 0.05,
                'interpretation': 'Errors are normally distributed' if dp_pvalue > 0.05 else 'Errors are not normal'
            }
        except Exception as e:
            results['dagostino_pearson'] = {'error': str(e)}
        
        # 3. Jarque-Bera Test (normality, good for larger samples)
        try:
            jb_stat, jb_pvalue = jarque_bera(errors_clean)
            results['jarque_bera'] = {
                'statistic': jb_stat,
                'p_value': jb_pvalue,
                'is_normal': jb_pvalue > 0.05,
                'interpretation': 'Errors are normally distributed' if jb_pvalue > 0.05 else 'Errors show skewness/kurtosis'
            }
        except Exception as e:
            results['jarque_bera'] = {'error': str(e)}
        
        # 4. Anderson-Darling Test
        try:
            ad_result = anderson(errors_clean, dist='norm')
            # Critical values at 15%, 10%, 5%, 2.5%, 1%
            critical_5pct = ad_result.critical_values[2]
            results['anderson_darling'] = {
                'statistic': ad_result.statistic,
                'critical_5pct': critical_5pct,
                'is_normal': ad_result.statistic < critical_5pct,
                'interpretation': 'Errors are normally distributed (5% level)' if ad_result.statistic < critical_5pct else 'Errors deviate from normal'
            }
        except Exception as e:
            results['anderson_darling'] = {'error': str(e)}
        
        # 5. Ljung-Box Test for autocorrelation (simplified)
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(errors_clean, lags=10, return_df=True)
            lb_pvalue = lb_result['lb_pvalue'].iloc[-1]  # Use last lag
            results['ljung_box'] = {
                'statistic': lb_result['lb_stat'].iloc[-1],
                'p_value': lb_pvalue,
                'no_autocorrelation': lb_pvalue > 0.05,
                'interpretation': 'No significant autocorrelation' if lb_pvalue > 0.05 else 'Errors show autocorrelation'
            }
        except Exception as e:
            results['ljung_box'] = {'error': str(e)}
        
        # Summary statistics of errors
        results['summary'] = {
            'mean': np.mean(errors_clean),
            'std': np.std(errors_clean),
            'skewness': stats.skew(errors_clean),
            'kurtosis': stats.kurtosis(errors_clean),
            'min': np.min(errors_clean),
            'max': np.max(errors_clean),
            'median': np.median(errors_clean)
        }
        
        return results
    
    def _run_percentile_tests(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Run percentile-based tests (VaR-style backtesting).
        """
        results = {}
        
        # Calculate prediction errors/returns
        errors = actuals - predictions
        returns = np.diff(actuals) / actuals[:-1] if len(actuals) > 1 else np.array([])
        
        if len(errors) < 10:
            return results
        
        # 1. Value at Risk (VaR) Backtesting - Kupiec Test
        for alpha in [0.01, 0.05, 0.10]:
            try:
                var_result = self._var_backtest_kupiec(errors, alpha)
                results[f'var_kupiec_{int(alpha*100):02d}'] = var_result
            except Exception as e:
                results[f'var_kupiec_{int(alpha*100):02d}'] = {'error': str(e)}
        
        # 2. Quantile Coverage Test
        try:
            quantile_result = self._quantile_coverage_test(actuals, predictions)
            results['quantile_coverage'] = quantile_result
        except Exception as e:
            results['quantile_coverage'] = {'error': str(e)}
        
        # 3. Pinball Loss (Quantile Loss) for different quantiles
        try:
            for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
                pinball_loss = self._pinball_loss(actuals, predictions, q)
                results[f'pinball_q{int(q*100):02d}'] = {
                    'quantile': q,
                    'pinball_loss': pinball_loss
                }
        except Exception as e:
            results['pinball_error'] = {'error': str(e)}
        
        # 4. Conditional Coverage Test (Christoffersen)
        try:
            for alpha in [0.05]:
                cc_result = self._conditional_coverage_test(errors, alpha)
                results[f'conditional_coverage_{int(alpha*100):02d}'] = cc_result
        except Exception as e:
            results['conditional_coverage'] = {'error': str(e)}
        
        # 5. Percentile Rank Analysis
        try:
            rank_result = self._percentile_rank_analysis(actuals, predictions)
            results['percentile_rank'] = rank_result
        except Exception as e:
            results['percentile_rank'] = {'error': str(e)}
        
        return results
    
    def _var_backtest_kupiec(self, errors: np.ndarray, alpha: float) -> Dict:
        """
        Kupiec's Proportion of Failures (POF) test for VaR backtesting.
        
        Tests if the observed number of VaR violations matches the expected number.
        """
        # Historical VaR (simplified - using error distribution)
        var_threshold = np.percentile(errors, alpha * 100)
        
        # Count violations (errors worse than VaR)
        violations = errors < var_threshold
        n_violations = np.sum(violations)
        n_total = len(errors)
        
        # Expected violations
        expected_violations = alpha * n_total
        
        # Kupiec test statistic (LR)
        if n_violations == 0 or n_violations == n_total:
            lr_stat = 0
            p_value = 1.0
        else:
            violation_rate = n_violations / n_total
            lr_stat = -2 * np.log(
                (alpha ** n_violations) * ((1 - alpha) ** (n_total - n_violations)) /
                ((violation_rate ** n_violations) * ((1 - violation_rate) ** (n_total - n_violations)))
            )
            p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        
        return {
            'alpha': alpha,
            'var_threshold': var_threshold,
            'violations': int(n_violations),
            'expected_violations': expected_violations,
            'violation_rate': n_violations / n_total,
            'lr_statistic': lr_stat,
            'p_value': p_value,
            'is_accurate': p_value > 0.05,
            'interpretation': f"VaR violations match expected {int(alpha*100)}% level" if p_value > 0.05 else f"VaR violations deviate from expected {int(alpha*100)}% level"
        }
    
    def _quantile_coverage_test(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
    ) -> Dict:
        """
        Test if the model's predictions cover the correct quantiles of actuals.
        """
        errors = actuals - predictions
        
        coverage_results = {}
        for q in quantiles:
            # Calculate empirical quantile of errors
            empirical_quantile = np.percentile(errors, q * 100)
            
            # Theoretical quantile (assuming centered errors)
            theoretical_quantile = 0  # For errors centered at 0
            
            # Coverage: proportion of errors below quantile
            coverage = np.mean(errors <= empirical_quantile)
            
            coverage_results[f'q{int(q*100):02d}'] = {
                'target_quantile': q,
                'empirical_quantile_value': empirical_quantile,
                'actual_coverage': coverage,
                'coverage_error': abs(coverage - q)
            }
        
        # Overall test statistic
        coverage_errors = [coverage_results[f'q{int(q*100):02d}']['coverage_error'] 
                          for q in quantiles]
        
        return {
            'quantiles': coverage_results,
            'max_coverage_error': max(coverage_errors),
            'mean_coverage_error': np.mean(coverage_errors),
            'is_well_calibrated': max(coverage_errors) < 0.1  # Within 10%
        }
    
    def _pinball_loss(self, actuals: np.ndarray, predictions: np.ndarray, quantile: float) -> float:
        """
        Calculate pinball loss (quantile loss) for a specific quantile.
        """
        errors = actuals - predictions
        return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))
    
    def _conditional_coverage_test(self, errors: np.ndarray, alpha: float) -> Dict:
        """
        Christoffersen's Conditional Coverage Test.
        
        Tests both unconditional coverage and independence of violations.
        """
        var_threshold = np.percentile(errors, alpha * 100)
        violations = errors < var_threshold
        
        n = len(violations)
        n00 = np.sum(~violations[:-1] & ~violations[1:])  # No violation -> No violation
        n01 = np.sum(~violations[:-1] & violations[1:])   # No violation -> Violation
        n10 = np.sum(violations[:-1] & ~violations[1:])   # Violation -> No violation
        n11 = np.sum(violations[:-1] & violations[1:])    # Violation -> Violation
        
        # Transition probabilities
        pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
        
        # Overall violation probability
        pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0
        
        # Likelihood ratio for independence
        if pi0 > 0 and pi1 > 0 and pi > 0:
            lr_ind = -2 * np.log(
                ((1 - pi) ** (n00 + n10) * pi ** (n01 + n11)) /
                ((1 - pi0) ** n00 * pi0 ** n01 * (1 - pi1) ** n10 * pi1 ** n11)
            )
        else:
            lr_ind = 0
        
        # Combine with Kupiec test
        n_violations = np.sum(violations)
        if n_violations > 0 and n_violations < n:
            violation_rate = n_violations / n
            lr_uc = -2 * np.log(
                (alpha ** n_violations) * ((1 - alpha) ** (n - n_violations)) /
                (violation_rate ** n_violations * (1 - violation_rate) ** (n - n_violations))
            )
        else:
            lr_uc = 0
        
        lr_cc = lr_uc + lr_ind
        p_value = 1 - stats.chi2.cdf(lr_cc, df=2)
        
        return {
            'alpha': alpha,
            'violations': int(n_violations),
            'transition_00': int(n00),
            'transition_01': int(n01),
            'transition_10': int(n10),
            'transition_11': int(n11),
            'pi0': pi0,
            'pi1': pi1,
            'lr_statistic': lr_cc,
            'p_value': p_value,
            'is_valid': p_value > 0.05,
            'interpretation': 'Conditional coverage is valid' if p_value > 0.05 else 'Violations show clustering (dependence)'
        }
    
    def _percentile_rank_analysis(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray
    ) -> Dict:
        """
        Analyze the percentile ranks of actuals relative to predictions.
        """
        # Calculate percentile rank of each actual value given the prediction
        # This assumes predictions are point estimates - we'd need distributions for full analysis
        errors = actuals - predictions
        
        # Rank actuals relative to predictions
        percentiles = np.array([percentileofscore(predictions, a, kind='rank') 
                               for a in actuals])
        
        return {
            'mean_percentile': np.mean(percentiles),
            'median_percentile': np.median(percentiles),
            'std_percentile': np.std(percentiles),
            'percentiles': percentiles.tolist(),
            'is_centered': 40 < np.median(percentiles) < 60,
            'interpretation': 'Predictions well-centered' if 40 < np.median(percentiles) < 60 else 'Predictions biased high/low'
        }
    
    def _compute_summary_stats(self, window_results: List[Dict]) -> Dict[str, float]:
        """Compute summary statistics across all windows."""
        if not window_results:
            return {}
        
        metrics = ['mse', 'mae', 'rmse', 'mape', 'directional_accuracy']
        summary = {}
        
        for metric in metrics:
            values = [w[metric] for w in window_results if not np.isnan(w[metric])]
            if values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_min'] = np.min(values)
                summary[f'{metric}_max'] = np.max(values)
        
        return summary


class BacktestComparator:
    """
    Compare backtesting results across multiple models.
    """
    
    def __init__(self):
        self.results: Dict[str, BacktestResult] = {}
    
    def add_result(self, result: BacktestResult):
        """Add a model's backtest result."""
        self.results[result.model_name] = result
    
    def get_summary_table(self) -> pd.DataFrame:
        """Get summary statistics table for all models."""
        rows = []
        for name, result in self.results.items():
            row = {'model': name}
            row.update(result.summary_stats)
            rows.append(row)
        
        return pd.DataFrame(rows).set_index('model') if rows else pd.DataFrame()
    
    def get_distributional_tests_table(self) -> pd.DataFrame:
        """Get distributional tests results."""
        rows = []
        test_names = ['kolmogorov_smirnov', 'dagostino_pearson', 'jarque_bera', 'anderson_darling']
        
        for model_name, result in self.results.items():
            row = {'model': model_name}
            for test in test_names:
                if test in result.distributional_tests:
                    test_result = result.distributional_tests[test]
                    if 'p_value' in test_result:
                        row[f'{test}_pvalue'] = test_result['p_value']
                        row[f'{test}_is_normal'] = test_result.get('is_normal', False)
            rows.append(row)
        
        return pd.DataFrame(rows).set_index('model') if rows else pd.DataFrame()
    
    def get_var_tests_table(self) -> pd.DataFrame:
        """Get VaR backtesting results."""
        rows = []
        var_levels = ['var_kupiec_01', 'var_kupiec_05', 'var_kupiec_10']
        
        for model_name, result in self.results.items():
            row = {'model': model_name}
            for level in var_levels:
                if level in result.percentile_tests:
                    var_result = result.percentile_tests[level]
                    if 'p_value' in var_result:
                        row[f'{level}_pvalue'] = var_result['p_value']
                        row[f'{level}_violations'] = var_result.get('violations', 0)
                        row[f'{level}_is_accurate'] = var_result.get('is_accurate', False)
            rows.append(row)
        
        return pd.DataFrame(rows).set_index('model') if rows else pd.DataFrame()
    
    def print_summary(self):
        """Print comprehensive backtest summary."""
        print("\n" + "=" * 70)
        print("BACKTEST SUMMARY")
        print("=" * 70)
        
        # Summary stats
        print("\n--- Performance Metrics (Cross-Validation) ---")
        summary_df = self.get_summary_table()
        if not summary_df.empty:
            print(summary_df.round(4))
        
        # Distributional tests
        print("\n--- Distributional Tests (Error Normality) ---")
        dist_df = self.get_distributional_tests_table()
        if not dist_df.empty:
            print(dist_df.round(4))
        
        # VaR tests
        print("\n--- VaR Backtesting (Kupiec Tests) ---")
        var_df = self.get_var_tests_table()
        if not var_df.empty:
            print(var_df.round(4))
        
        # Best models
        print("\n--- Best Models by Metric ---")
        if not summary_df.empty:
            for metric in ['rmse_mean', 'mae_mean', 'directional_accuracy_mean']:
                if metric in summary_df.columns:
                    ascending = 'accuracy' not in metric
                    best = summary_df[metric].sort_values(ascending=ascending).index[0]
                    print(f"  {metric}: {best} ({summary_df.loc[best, metric]:.4f})")


if __name__ == "__main__":
    # Example usage
    print("Backtesting module loaded successfully.")
    print("\nExample usage:")
    print("  backtester = RollingWindowBacktester(window_size=20, min_train_size=252)")
    print("  result = backtester.run_backtest(model_factory, data, 'MyModel')")
    print("  comparator = BacktestComparator()")
    print("  comparator.add_result(result)")
    print("  comparator.print_summary()")