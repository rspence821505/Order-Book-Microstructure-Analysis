"""
Model Profiling and Latency Measurement Module

This module provides functions for production-ready model performance analysis:
- Model inference latency benchmarking
- Latency distribution measurement (p50, p90, p99)
- Model memory footprint analysis
- Model compression analysis (tree count optimization)
- Performance decay monitoring
- Robustness testing
- Multi-model comparison

Extracted from notebook 75_production_benchmarks.ipynb
"""

import numpy as np
import pandas as pd
import time
import pickle
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings

warnings.filterwarnings("ignore")


def benchmark_model_inference(
    model: BaseEstimator,
    X_sample: Union[np.ndarray, pd.DataFrame],
    n_iterations: int = 1000,
    warmup_iterations: int = 100,
) -> Dict[str, Any]:
    """
    Benchmark model inference latency with statistical analysis.

    Performs warm-up iterations to ensure JIT compilation and cache loading,
    then measures inference latency over multiple iterations.

    Parameters
    ----------
    model : BaseEstimator
        Trained scikit-learn model
    X_sample : array-like of shape (n_features,) or (1, n_features)
        Single sample for prediction
    n_iterations : int, default=1000
        Number of benchmark iterations
    warmup_iterations : int, default=100
        Number of warm-up iterations before benchmarking

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'mean': Mean latency in microseconds
        - 'std': Standard deviation in microseconds
        - 'p50': 50th percentile (median) in microseconds
        - 'p90': 90th percentile in microseconds
        - 'p99': 99th percentile in microseconds
        - 'min': Minimum latency in microseconds
        - 'max': Maximum latency in microseconds
        - 'latencies': Array of all latency measurements

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier(n_estimators=100)
    >>> model.fit(X_train, y_train)
    >>> X_sample = X_test[0].reshape(1, -1)
    >>> results = benchmark_model_inference(model, X_sample)
    >>> print(f"P90 latency: {results['p90']:.1f}μs")
    P90 latency: 245.3μs
    """
    latencies_us = []

    # Ensure single sample (1, n_features)
    if isinstance(X_sample, pd.DataFrame):
        X_sample = X_sample.values
    if len(X_sample.shape) == 1:
        X_sample = X_sample.reshape(1, -1)

    # Warm-up (JIT compilation, cache loading)
    for _ in range(warmup_iterations):
        _ = model.predict(X_sample)

    # Actual benchmark
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = model.predict(X_sample)
        end = time.perf_counter()
        latencies_us.append((end - start) * 1_000_000)  # Convert to microseconds

    latencies_us = np.array(latencies_us)

    return {
        "mean": float(latencies_us.mean()),
        "std": float(latencies_us.std()),
        "p50": float(np.percentile(latencies_us, 50)),
        "p90": float(np.percentile(latencies_us, 90)),
        "p99": float(np.percentile(latencies_us, 99)),
        "min": float(latencies_us.min()),
        "max": float(latencies_us.max()),
        "latencies": latencies_us,
    }


def benchmark_multiple_models(
    models: Dict[str, BaseEstimator],
    X_sample: Union[np.ndarray, pd.DataFrame],
    n_iterations: int = 1000,
    target_latency_us: Optional[float] = None,
) -> pd.DataFrame:
    """
    Benchmark inference latency for multiple models.

    Parameters
    ----------
    models : dict
        Dictionary mapping model names to trained models
    X_sample : array-like
        Single sample for prediction
    n_iterations : int, default=1000
        Number of benchmark iterations per model
    target_latency_us : float, optional
        Target latency in microseconds for comparison

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns: model, mean_us, std_us, p50_us, p90_us, p99_us, meets_target

    Examples
    --------
    >>> models = {
    ...     'Decision Tree': dt_model,
    ...     'Random Forest': rf_model,
    ...     'Gradient Boosting': gb_model
    ... }
    >>> results = benchmark_multiple_models(models, X_sample, target_latency_us=500)
    >>> print(results)
    """
    results = []

    for model_name, model in models.items():
        print(f"  Benchmarking {model_name}...", end=" ", flush=True)
        benchmark_result = benchmark_model_inference(model, X_sample, n_iterations)

        result_dict = {
            "model": model_name,
            "mean_us": benchmark_result["mean"],
            "std_us": benchmark_result["std"],
            "p50_us": benchmark_result["p50"],
            "p90_us": benchmark_result["p90"],
            "p99_us": benchmark_result["p99"],
            "min_us": benchmark_result["min"],
            "max_us": benchmark_result["max"],
        }

        if target_latency_us is not None:
            result_dict["meets_target"] = benchmark_result["p90"] < target_latency_us

        results.append(result_dict)
        print("done")

    results_df = pd.DataFrame(results)
    return results_df


def get_model_memory_footprint(model: BaseEstimator) -> float:
    """
    Get approximate model size in megabytes.

    Parameters
    ----------
    model : BaseEstimator
        Trained scikit-learn model

    Returns
    -------
    size_mb : float
        Model size in megabytes

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier(n_estimators=100)
    >>> model.fit(X_train, y_train)
    >>> size_mb = get_model_memory_footprint(model)
    >>> print(f"Model size: {size_mb:.2f} MB")
    Model size: 45.32 MB
    """
    pickled = pickle.dumps(model)
    return len(pickled) / (1024**2)  # Convert bytes to MB


def analyze_model_compression(
    model_class: type,
    model_params: Dict[str, Any],
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: np.ndarray,
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: np.ndarray,
    complexity_param: str,
    complexity_values: List[int],
    n_iterations: int = 500,
) -> pd.DataFrame:
    """
    Analyze trade-off between model complexity and performance.

    Tests multiple complexity values (e.g., n_estimators for Random Forest)
    and measures accuracy, F1 score, latency, and memory footprint.

    Parameters
    ----------
    model_class : type
        Model class (e.g., RandomForestClassifier)
    model_params : dict
        Base model parameters (excluding complexity parameter)
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    complexity_param : str
        Parameter name to vary (e.g., 'n_estimators')
    complexity_values : list of int
        Values to test for complexity parameter
    n_iterations : int, default=500
        Number of latency benchmark iterations

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns: complexity_value, accuracy, f1_score,
        latency_p90_us, size_mb

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> results = analyze_model_compression(
    ...     model_class=RandomForestClassifier,
    ...     model_params={'max_depth': 12, 'random_state': 42},
    ...     X_train=X_train, y_train=y_train,
    ...     X_test=X_test, y_test=y_test,
    ...     complexity_param='n_estimators',
    ...     complexity_values=[10, 25, 50, 100, 200]
    ... )
    >>> print(results)
    """
    results = []

    if isinstance(X_test, pd.DataFrame):
        X_sample = X_test.iloc[0].values.reshape(1, -1)
    else:
        X_sample = X_test[0].reshape(1, -1)

    for complexity_value in complexity_values:
        # Create model with current complexity
        params = model_params.copy()
        params[complexity_param] = complexity_value
        model = model_class(**params)

        # Train model
        model.fit(X_train, y_train)

        # Evaluate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Benchmark latency
        latency_result = benchmark_model_inference(model, X_sample, n_iterations)

        # Get model size
        size_mb = get_model_memory_footprint(model)

        results.append(
            {
                complexity_param: complexity_value,
                "accuracy": accuracy,
                "f1_score": f1,
                "latency_p90_us": latency_result["p90"],
                "size_mb": size_mb,
            }
        )

    return pd.DataFrame(results)


def analyze_performance_decay(
    models: Dict[str, BaseEstimator],
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: np.ndarray,
    n_windows: int = 5,
) -> pd.DataFrame:
    """
    Analyze model performance across time-ordered windows.

    Simulates how model accuracy degrades over time without retraining.

    Parameters
    ----------
    models : dict
        Dictionary mapping model names to trained models
    X_test : array-like
        Test features (time-ordered)
    y_test : array-like
        Test labels (time-ordered)
    n_windows : int, default=5
        Number of time windows to split test set into

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns: window, model, accuracy, f1_score

    Examples
    --------
    >>> models = {'Random Forest': rf_model, 'Gradient Boosting': gb_model}
    >>> results = analyze_performance_decay(models, X_test, y_test)
    >>> print(results.pivot_table(index='window', columns='model', values='accuracy'))
    """
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.reset_index(drop=True)

    window_size = len(X_test) // n_windows
    results = []

    for window_idx in range(n_windows):
        start_idx = window_idx * window_size
        end_idx = min((window_idx + 1) * window_size, len(X_test))

        if isinstance(X_test, pd.DataFrame):
            X_window = X_test.iloc[start_idx:end_idx]
        else:
            X_window = X_test[start_idx:end_idx]
        y_window = y_test[start_idx:end_idx]

        # Evaluate each model on this window
        for model_name, model in models.items():
            y_pred = model.predict(X_window)
            accuracy = accuracy_score(y_window, y_pred)
            f1 = f1_score(y_window, y_pred)

            results.append(
                {
                    "window": window_idx + 1,
                    "model": model_name,
                    "accuracy": accuracy,
                    "f1_score": f1,
                }
            )

    return pd.DataFrame(results)


def test_model_robustness(
    models: Dict[str, BaseEstimator],
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: np.ndarray,
    scenarios: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Test model robustness to data quality issues.

    Tests models under various scenarios:
    - Baseline (clean data)
    - Missing features (10% features set to zero)
    - Gaussian noise (σ=0.1)
    - Extreme outliers (5% samples multiplied by 10)

    Parameters
    ----------
    models : dict
        Dictionary mapping model names to trained models
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    scenarios : list of str, optional
        List of scenario names to test. If None, tests all scenarios.
        Available: 'baseline', 'missing', 'noise', 'outliers'

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns: scenario, model, accuracy

    Examples
    --------
    >>> models = {'Random Forest': rf_model, 'Gradient Boosting': gb_model}
    >>> results = test_model_robustness(models, X_test, y_test)
    >>> print(results.pivot_table(index='scenario', columns='model', values='accuracy'))
    """
    if scenarios is None:
        scenarios = ["baseline", "missing", "noise", "outliers"]

    if isinstance(X_test, pd.DataFrame):
        X_test_array = X_test.values
        feature_cols = X_test.columns.tolist()
    else:
        X_test_array = X_test
        feature_cols = list(range(X_test.shape[1]))

    results = []

    # Baseline (clean data)
    if "baseline" in scenarios:
        for model_name, model in models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results.append(
                {"scenario": "Baseline (clean)", "model": model_name, "accuracy": accuracy}
            )

    # Scenario 1: 10% features set to zero (missing data)
    if "missing" in scenarios:
        X_test_missing = X_test_array.copy()
        n_features_missing = int(0.1 * len(feature_cols))
        missing_features = np.random.choice(
            len(feature_cols), n_features_missing, replace=False
        )
        X_test_missing[:, missing_features] = 0

        for model_name, model in models.items():
            y_pred = model.predict(X_test_missing)
            accuracy = accuracy_score(y_test, y_pred)
            results.append(
                {
                    "scenario": "10% missing features",
                    "model": model_name,
                    "accuracy": accuracy,
                }
            )

    # Scenario 2: Add random noise
    if "noise" in scenarios:
        X_test_noisy = X_test_array.copy()
        noise = np.random.normal(0, 0.1, X_test_noisy.shape)
        X_test_noisy = X_test_noisy + noise

        for model_name, model in models.items():
            y_pred = model.predict(X_test_noisy)
            accuracy = accuracy_score(y_test, y_pred)
            results.append(
                {
                    "scenario": "Gaussian noise (σ=0.1)",
                    "model": model_name,
                    "accuracy": accuracy,
                }
            )

    # Scenario 3: Extreme outliers
    if "outliers" in scenarios:
        X_test_outliers = X_test_array.copy()
        n_outliers = int(0.05 * len(X_test))  # 5% of samples
        outlier_indices = np.random.choice(len(X_test), n_outliers, replace=False)
        X_test_outliers[outlier_indices] *= 10  # Multiply by 10

        for model_name, model in models.items():
            y_pred = model.predict(X_test_outliers)
            accuracy = accuracy_score(y_test, y_pred)
            results.append(
                {
                    "scenario": "5% extreme outliers",
                    "model": model_name,
                    "accuracy": accuracy,
                }
            )

    return pd.DataFrame(results)


def measure_latency_distribution(
    model: BaseEstimator,
    X_samples: Union[np.ndarray, pd.DataFrame],
    n_iterations_per_sample: int = 100,
) -> Dict[str, Any]:
    """
    Measure latency distribution across multiple samples.

    Parameters
    ----------
    model : BaseEstimator
        Trained scikit-learn model
    X_samples : array-like of shape (n_samples, n_features)
        Multiple samples for prediction
    n_iterations_per_sample : int, default=100
        Number of iterations per sample

    Returns
    -------
    results : dict
        Dictionary containing aggregate latency statistics across all samples

    Examples
    --------
    >>> X_samples = X_test[:10]  # Test on 10 samples
    >>> results = measure_latency_distribution(model, X_samples)
    >>> print(f"Mean latency: {results['mean']:.1f}μs")
    """
    all_latencies = []

    if isinstance(X_samples, pd.DataFrame):
        X_samples = X_samples.values

    for i in range(len(X_samples)):
        X_sample = X_samples[i].reshape(1, -1)
        result = benchmark_model_inference(
            model, X_sample, n_iterations=n_iterations_per_sample, warmup_iterations=10
        )
        all_latencies.extend(result["latencies"])

    all_latencies = np.array(all_latencies)

    return {
        "mean": float(all_latencies.mean()),
        "std": float(all_latencies.std()),
        "p50": float(np.percentile(all_latencies, 50)),
        "p90": float(np.percentile(all_latencies, 90)),
        "p99": float(np.percentile(all_latencies, 99)),
        "min": float(all_latencies.min()),
        "max": float(all_latencies.max()),
    }


def compare_model_speed(
    models: Dict[str, BaseEstimator],
    X_sample: Union[np.ndarray, pd.DataFrame],
    n_iterations: int = 1000,
) -> pd.DataFrame:
    """
    Compare inference speed of multiple models.

    Parameters
    ----------
    models : dict
        Dictionary mapping model names to trained models
    X_sample : array-like
        Single sample for prediction
    n_iterations : int, default=1000
        Number of benchmark iterations

    Returns
    -------
    comparison_df : pd.DataFrame
        DataFrame sorted by p90 latency with speedup ratios

    Examples
    --------
    >>> models = {
    ...     'Decision Tree': dt_model,
    ...     'Random Forest': rf_model,
    ...     'Gradient Boosting': gb_model
    ... }
    >>> comparison = compare_model_speed(models, X_sample)
    >>> print(comparison)
    """
    results = []

    for model_name, model in models.items():
        benchmark_result = benchmark_model_inference(model, X_sample, n_iterations)
        results.append(
            {
                "model": model_name,
                "p50_us": benchmark_result["p50"],
                "p90_us": benchmark_result["p90"],
                "p99_us": benchmark_result["p99"],
            }
        )

    comparison_df = pd.DataFrame(results).sort_values("p90_us")

    # Calculate speedup relative to slowest model
    slowest_p90 = comparison_df["p90_us"].max()
    comparison_df["speedup"] = slowest_p90 / comparison_df["p90_us"]

    return comparison_df


def benchmark_feature_extraction(
    feature_vector: np.ndarray, n_iterations: int = 1000
) -> Dict[str, float]:
    """
    Benchmark feature vector preparation (simulated).

    In production, this would involve actual computation from raw order book/trade data.
    This function simulates basic array operations as a placeholder.

    Parameters
    ----------
    feature_vector : array-like
        Feature vector to process
    n_iterations : int, default=1000
        Number of benchmark iterations

    Returns
    -------
    results : dict
        Dictionary with mean, p50, p90, p99 latencies in microseconds

    Examples
    --------
    >>> feature_vec = X_test[0]
    >>> results = benchmark_feature_extraction(feature_vec)
    >>> print(f"P90 feature extraction: {results['p90']:.1f}μs")
    """
    latencies_us = []

    for _ in range(n_iterations):
        start = time.perf_counter()

        # Simulate feature computation (array operations)
        _ = np.array(feature_vector)
        _ = np.sum(feature_vector)
        _ = np.mean(feature_vector)
        _ = np.std(feature_vector)

        end = time.perf_counter()
        latencies_us.append((end - start) * 1_000_000)

    latencies_us = np.array(latencies_us)

    return {
        "mean": float(latencies_us.mean()),
        "p50": float(np.percentile(latencies_us, 50)),
        "p90": float(np.percentile(latencies_us, 90)),
        "p99": float(np.percentile(latencies_us, 99)),
    }


def estimate_total_pipeline_latency(
    model: BaseEstimator,
    X_sample: Union[np.ndarray, pd.DataFrame],
    estimated_feature_latency_us: float,
    n_iterations: int = 1000,
) -> Dict[str, float]:
    """
    Estimate total pipeline latency (feature extraction + model inference).

    Parameters
    ----------
    model : BaseEstimator
        Trained model
    X_sample : array-like
        Single sample for prediction
    estimated_feature_latency_us : float
        Estimated feature extraction latency in microseconds
    n_iterations : int, default=1000
        Number of benchmark iterations

    Returns
    -------
    results : dict
        Dictionary with component and total latencies

    Examples
    --------
    >>> results = estimate_total_pipeline_latency(
    ...     model=rf_model,
    ...     X_sample=X_test[0],
    ...     estimated_feature_latency_us=180
    ... )
    >>> print(f"Total P90: {results['total_p90_us']:.1f}μs")
    """
    model_latency = benchmark_model_inference(model, X_sample, n_iterations)

    return {
        "feature_extraction_us": estimated_feature_latency_us,
        "model_inference_p50_us": model_latency["p50"],
        "model_inference_p90_us": model_latency["p90"],
        "model_inference_p99_us": model_latency["p99"],
        "total_p50_us": estimated_feature_latency_us + model_latency["p50"],
        "total_p90_us": estimated_feature_latency_us + model_latency["p90"],
        "total_p99_us": estimated_feature_latency_us + model_latency["p99"],
    }


def check_latency_target(
    latency_us: float, target_us: float, percentile: str = "p90"
) -> Tuple[bool, str]:
    """
    Check if latency meets target threshold.

    Parameters
    ----------
    latency_us : float
        Measured latency in microseconds
    target_us : float
        Target latency threshold in microseconds
    percentile : str, default='p90'
        Percentile being checked (for status message)

    Returns
    -------
    meets_target : bool
        Whether latency is below target
    status_msg : str
        Status message with result

    Examples
    --------
    >>> meets_target, msg = check_latency_target(245.3, 500, 'p90')
    >>> print(msg)
    ✓ MEETS target: 245.3μs < 500μs (p90)
    """
    meets_target = latency_us < target_us

    if meets_target:
        status_msg = f"✓ MEETS target: {latency_us:.1f}μs < {target_us}μs ({percentile})"
    else:
        status_msg = (
            f"✗ EXCEEDS target: {latency_us:.1f}μs >= {target_us}μs ({percentile})"
        )

    return meets_target, status_msg


def analyze_feature_selection_impact(
    model_class: type,
    model_params: Dict[str, Any],
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: np.ndarray,
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: np.ndarray,
    feature_sets: Dict[str, List[str]],
    n_iterations: int = 500,
) -> pd.DataFrame:
    """
    Analyze impact of feature selection on accuracy and latency.

    Parameters
    ----------
    model_class : type
        Model class to train
    model_params : dict
        Model parameters
    X_train : pd.DataFrame
        Training features
    y_train : array-like
        Training labels
    X_test : pd.DataFrame
        Test features
    y_test : array-like
        Test labels
    feature_sets : dict
        Dictionary mapping set names to lists of feature names
    n_iterations : int, default=500
        Number of latency benchmark iterations

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns: feature_set, n_features, accuracy, f1_score, latency_p90_us

    Examples
    --------
    >>> feature_sets = {
    ...     'All features': feature_cols,
    ...     'Top 20 features': top_20_features,
    ...     'Top 10 features': top_10_features
    ... }
    >>> results = analyze_feature_selection_impact(
    ...     RandomForestClassifier, {'n_estimators': 100},
    ...     X_train, y_train, X_test, y_test, feature_sets
    ... )
    """
    if not isinstance(X_train, pd.DataFrame):
        raise ValueError("X_train must be a pandas DataFrame for feature selection")

    results = []

    for set_name, features in feature_sets.items():
        # Select features
        X_train_subset = X_train[features]
        X_test_subset = X_test[features]

        # Train model
        model = model_class(**model_params)
        model.fit(X_train_subset, y_train)

        # Evaluate
        y_pred = model.predict(X_test_subset)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Benchmark
        X_sample_subset = X_test_subset.iloc[0].values.reshape(1, -1)
        latency_result = benchmark_model_inference(
            model, X_sample_subset, n_iterations=n_iterations
        )

        results.append(
            {
                "feature_set": set_name,
                "n_features": len(features),
                "accuracy": accuracy,
                "f1_score": f1,
                "latency_p90_us": latency_result["p90"],
            }
        )

    return pd.DataFrame(results)


def generate_production_readiness_report(
    benchmark_results: Dict[str, Dict[str, float]],
    memory_footprints: Dict[str, float],
    target_model_latency_us: float,
    target_total_latency_us: float,
    estimated_feature_latency_us: float,
) -> str:
    """
    Generate a production readiness assessment report.

    Parameters
    ----------
    benchmark_results : dict
        Dictionary mapping model names to benchmark results
    memory_footprints : dict
        Dictionary mapping model names to memory sizes (MB)
    target_model_latency_us : float
        Target model inference latency in microseconds
    target_total_latency_us : float
        Target total pipeline latency in microseconds
    estimated_feature_latency_us : float
        Estimated feature extraction latency in microseconds

    Returns
    -------
    report : str
        Formatted production readiness report

    Examples
    --------
    >>> report = generate_production_readiness_report(
    ...     benchmark_results, memory_footprints,
    ...     target_model_latency_us=500,
    ...     target_total_latency_us=1000,
    ...     estimated_feature_latency_us=180
    ... )
    >>> print(report)
    """
    report = []
    report.append("=" * 80)
    report.append("PRODUCTION READINESS ASSESSMENT")
    report.append("=" * 80)

    report.append(f"\nPERFORMANCE TARGETS:")
    report.append(f"  Feature extraction: <{estimated_feature_latency_us}μs")
    report.append(f"  Model inference: <{target_model_latency_us}μs")
    report.append(f"  Total pipeline: <{target_total_latency_us}μs")

    report.append(f"\nMODEL INFERENCE LATENCY (P90):")
    for model_name, results in benchmark_results.items():
        status = "✓" if results["p90"] < target_model_latency_us else "✗"
        report.append(f"  {status} {model_name:20s}: {results['p90']:6.1f}μs")

    report.append(f"\nESTIMATED TOTAL PIPELINE LATENCY (P90):")
    for model_name, results in benchmark_results.items():
        total = estimated_feature_latency_us + results["p90"]
        status = "✓" if total < target_total_latency_us else "✗"
        report.append(
            f"  {status} {model_name:20s}: {total:6.1f}μs (feature + model)"
        )

    report.append(f"\nMODEL MEMORY FOOTPRINT:")
    for model_name, size_mb in memory_footprints.items():
        report.append(f"  {model_name:20s}: {size_mb:6.2f} MB")

    report.append(f"\nPRODUCTION READINESS:")
    all_meet_target = all(
        res["p90"] < target_model_latency_us for res in benchmark_results.values()
    )
    if all_meet_target:
        report.append(
            f"  ✓ ALL MODELS PRODUCTION-READY for HFT (<{target_total_latency_us}μs total)"
        )
    else:
        report.append(f"  ⚠ Some models exceed latency target")
        passing_models = [
            name
            for name, res in benchmark_results.items()
            if res["p90"] < target_model_latency_us
        ]
        if passing_models:
            report.append(
                f"  ✓ Passing models: {', '.join(passing_models)}"
            )

    report.append("=" * 80)

    return "\n".join(report)


if __name__ == "__main__":
    print("Model Profiling Module - Production Latency Measurement")
    print("=" * 70)
    print("\nKey functions:")
    print("  - benchmark_model_inference(): Measure single-model latency")
    print("  - benchmark_multiple_models(): Compare multiple models")
    print("  - get_model_memory_footprint(): Measure model size")
    print("  - analyze_model_compression(): Trade-off analysis")
    print("  - analyze_performance_decay(): Monitor decay over time")
    print("  - test_model_robustness(): Test data quality resilience")
    print("  - estimate_total_pipeline_latency(): End-to-end latency")
    print("  - generate_production_readiness_report(): Assessment report")
    print("\nSee function docstrings for detailed usage examples.")
