"""
Feature Profiling and Production Benchmarking Module

This module provides functions for production-ready performance analysis:
- Feature computation benchmarking
- Model inference latency profiling
- Bottleneck identification
- Pipeline end-to-end benchmarking
- Model compression analysis

Extracted from notebook 75_production_benchmarks.ipynb
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from sklearn.base import BaseEstimator
import warnings


def benchmark_model_inference(
    model: BaseEstimator,
    X_sample: Union[pd.DataFrame, np.ndarray],
    n_iterations: int = 1000,
    warmup_iterations: int = 100,
) -> Dict[str, float]:
    """
    Benchmark model inference latency with statistical analysis.

    Measures single-prediction latency including warm-up for JIT compilation
    and cache loading.

    Parameters
    ----------
    model : BaseEstimator
        Trained sklearn model
    X_sample : pd.DataFrame or np.ndarray
        Single sample or batch to predict on
    n_iterations : int, default=1000
        Number of iterations for benchmark
    warmup_iterations : int, default=100
        Number of warm-up iterations before measurement

    Returns
    -------
    Dict[str, float]
        Dictionary with latency statistics (in microseconds):
        - mean: Mean latency
        - std: Standard deviation
        - p50: Median (50th percentile)
        - p90: 90th percentile
        - p99: 99th percentile
        - min: Minimum latency
        - max: Maximum latency
        - latencies: Full array of latencies

    Examples
    --------
    >>> X_sample = X_test.iloc[0].values.reshape(1, -1)
    >>> results = benchmark_model_inference(rf_model, X_sample, n_iterations=1000)
    >>> print(f"P90 latency: {results['p90']:.1f}μs")
    """
    # Ensure proper shape
    if isinstance(X_sample, pd.DataFrame):
        X_array = X_sample.values
    else:
        X_array = X_sample

    if len(X_array.shape) == 1:
        X_array = X_array.reshape(1, -1)

    # Warm-up (JIT compilation, cache loading)
    for _ in range(warmup_iterations):
        _ = model.predict(X_array)

    # Actual benchmark
    latencies_us = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = model.predict(X_array)
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
    X_sample: Union[pd.DataFrame, np.ndarray],
    n_iterations: int = 1000,
) -> pd.DataFrame:
    """
    Benchmark multiple models and return comparison table.

    Parameters
    ----------
    models : Dict[str, BaseEstimator]
        Dictionary mapping model names to trained models
    X_sample : pd.DataFrame or np.ndarray
        Sample to predict on
    n_iterations : int, default=1000
        Number of iterations per model

    Returns
    -------
    pd.DataFrame
        Comparison table with latency statistics for each model

    Examples
    --------
    >>> models = {'RF': rf_model, 'GB': gb_model, 'DT': dt_model}
    >>> results = benchmark_multiple_models(models, X_sample)
    >>> print(results)
    """
    results = {}

    for model_name, model in models.items():
        benchmark = benchmark_model_inference(model, X_sample, n_iterations)
        results[model_name] = {
            "mean_us": benchmark["mean"],
            "std_us": benchmark["std"],
            "p50_us": benchmark["p50"],
            "p90_us": benchmark["p90"],
            "p99_us": benchmark["p99"],
        }

    return pd.DataFrame(results).T


def profile_features(
    feature_computation_fn: Callable,
    data_sample: Any,
    n_iterations: int = 1000,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Profile feature computation time.

    Measures the latency of feature extraction from raw data.

    Parameters
    ----------
    feature_computation_fn : Callable
        Function that computes features from raw data
        Should accept data_sample and return features
    data_sample : Any
        Sample of raw data to process
    n_iterations : int, default=1000
        Number of iterations for benchmark
    feature_names : List[str], optional
        Names of features being computed

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - latency_stats: Statistical summary (mean, p50, p90, p99)
        - n_features: Number of features computed
        - feature_names: Names of features (if provided)
        - per_feature_us: Average time per feature

    Examples
    --------
    >>> def compute_features(data):
    ...     return np.array([data.mean(), data.std(), data.max()])
    >>>
    >>> results = profile_features(compute_features, raw_data, n_iterations=1000)
    >>> print(f"Total time: {results['latency_stats']['p90']:.1f}μs")
    """
    latencies_us = []

    # Warm-up
    for _ in range(100):
        _ = feature_computation_fn(data_sample)

    # Benchmark
    for _ in range(n_iterations):
        start = time.perf_counter()
        features = feature_computation_fn(data_sample)
        end = time.perf_counter()
        latencies_us.append((end - start) * 1_000_000)

    latencies_us = np.array(latencies_us)

    # Get number of features
    if isinstance(features, np.ndarray):
        n_features = len(features)
    elif isinstance(features, (list, tuple)):
        n_features = len(features)
    else:
        n_features = 1

    # Calculate stats
    latency_stats = {
        "mean": float(latencies_us.mean()),
        "std": float(latencies_us.std()),
        "p50": float(np.percentile(latencies_us, 50)),
        "p90": float(np.percentile(latencies_us, 90)),
        "p99": float(np.percentile(latencies_us, 99)),
    }

    return {
        "latency_stats": latency_stats,
        "n_features": n_features,
        "feature_names": feature_names,
        "per_feature_us": latency_stats["mean"] / n_features if n_features > 0 else 0,
    }


def identify_bottlenecks(
    pipeline_components: Dict[str, Callable],
    data_sample: Any,
    n_iterations: int = 100,
) -> pd.DataFrame:
    """
    Identify bottlenecks in a multi-stage pipeline.

    Profiles each component and identifies the slowest stages.

    Parameters
    ----------
    pipeline_components : Dict[str, Callable]
        Dictionary mapping component names to functions
        Each function should accept data and return output
    data_sample : Any
        Sample data to process through pipeline
    n_iterations : int, default=100
        Number of iterations per component

    Returns
    -------
    pd.DataFrame
        DataFrame with component profiling results:
        - component: Component name
        - mean_us: Mean latency in microseconds
        - p90_us: 90th percentile latency
        - percentage: Percentage of total pipeline time

    Examples
    --------
    >>> components = {
    ...     'load_data': lambda x: load_data(x),
    ...     'extract_features': lambda x: extract_features(x),
    ...     'predict': lambda x: model.predict(x),
    ... }
    >>> bottlenecks = identify_bottlenecks(components, sample_data)
    >>> print(bottlenecks.sort_values('percentage', ascending=False))
    """
    results = []
    current_data = data_sample

    for component_name, component_fn in pipeline_components.items():
        latencies_us = []

        # Warm-up
        for _ in range(10):
            _ = component_fn(current_data)

        # Benchmark
        for _ in range(n_iterations):
            start = time.perf_counter()
            output = component_fn(current_data)
            end = time.perf_counter()
            latencies_us.append((end - start) * 1_000_000)

        # Update current data for next component
        current_data = output

        latencies_us = np.array(latencies_us)

        results.append({
            "component": component_name,
            "mean_us": float(latencies_us.mean()),
            "std_us": float(latencies_us.std()),
            "p50_us": float(np.percentile(latencies_us, 50)),
            "p90_us": float(np.percentile(latencies_us, 90)),
            "p99_us": float(np.percentile(latencies_us, 99)),
        })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Calculate percentage of total time
    total_time = df["mean_us"].sum()
    df["percentage"] = (df["mean_us"] / total_time * 100).round(2)

    # Sort by percentage (descending)
    df = df.sort_values("percentage", ascending=False).reset_index(drop=True)

    return df


def benchmark_pipeline(
    pipeline_fn: Callable,
    data_samples: List[Any],
    n_iterations: int = 100,
    target_latency_us: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Benchmark end-to-end pipeline performance.

    Measures total latency from raw data input to final prediction.

    Parameters
    ----------
    pipeline_fn : Callable
        Complete pipeline function (data -> prediction)
    data_samples : List[Any]
        List of data samples to test
    n_iterations : int, default=100
        Number of iterations per sample
    target_latency_us : float, optional
        Target latency in microseconds for comparison

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - latency_stats: Statistical summary
        - meets_target: Whether p90 meets target (if provided)
        - samples_tested: Number of samples tested
        - latencies: Full array of latencies

    Examples
    --------
    >>> def pipeline(data):
    ...     features = extract_features(data)
    ...     prediction = model.predict(features)
    ...     return prediction
    >>>
    >>> results = benchmark_pipeline(
    ...     pipeline, test_samples, target_latency_us=1000
    ... )
    >>> print(f"Meets target: {results['meets_target']}")
    """
    all_latencies = []

    for sample in data_samples:
        # Warm-up
        for _ in range(10):
            _ = pipeline_fn(sample)

        # Benchmark
        for _ in range(n_iterations):
            start = time.perf_counter()
            _ = pipeline_fn(sample)
            end = time.perf_counter()
            all_latencies.append((end - start) * 1_000_000)

    all_latencies = np.array(all_latencies)

    latency_stats = {
        "mean": float(all_latencies.mean()),
        "std": float(all_latencies.std()),
        "p50": float(np.percentile(all_latencies, 50)),
        "p90": float(np.percentile(all_latencies, 90)),
        "p99": float(np.percentile(all_latencies, 99)),
        "min": float(all_latencies.min()),
        "max": float(all_latencies.max()),
    }

    results = {
        "latency_stats": latency_stats,
        "samples_tested": len(data_samples),
        "iterations_per_sample": n_iterations,
        "total_measurements": len(all_latencies),
        "latencies": all_latencies,
    }

    if target_latency_us is not None:
        results["meets_target"] = latency_stats["p90"] < target_latency_us
        results["target_latency_us"] = target_latency_us

    return results


def analyze_model_compression(
    model_class: type,
    model_params: Dict[str, Any],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: np.ndarray,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: np.ndarray,
    complexity_param: str,
    complexity_values: List[Any],
) -> pd.DataFrame:
    """
    Analyze trade-off between model complexity and performance.

    Useful for Random Forest (n_estimators), Gradient Boosting (n_estimators),
    or any model with tunable complexity parameter.

    Parameters
    ----------
    model_class : type
        Model class (e.g., RandomForestClassifier)
    model_params : Dict[str, Any]
        Base model parameters
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_test : pd.DataFrame or np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    complexity_param : str
        Parameter name to vary (e.g., 'n_estimators')
    complexity_values : List[Any]
        Values to test for complexity parameter

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - complexity_value: Value of complexity parameter
        - accuracy: Test accuracy
        - latency_p90_us: 90th percentile latency
        - model_size_mb: Model size in memory

    Examples
    --------
    >>> results = analyze_model_compression(
    ...     RandomForestClassifier,
    ...     {'max_depth': 10, 'random_state': 42},
    ...     X_train, y_train, X_test, y_test,
    ...     'n_estimators', [10, 50, 100, 200]
    ... )
    >>> print(results)
    """
    from sklearn.metrics import accuracy_score
    import pickle

    results = []
    X_sample = X_test[:1] if isinstance(X_test, np.ndarray) else X_test.iloc[:1]

    for value in complexity_values:
        # Create model with this complexity
        params = model_params.copy()
        params[complexity_param] = value

        model = model_class(**params)
        model.fit(X_train, y_train)

        # Evaluate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Benchmark latency
        latency = benchmark_model_inference(model, X_sample, n_iterations=500)

        # Get model size
        pickled = pickle.dumps(model)
        size_mb = len(pickled) / (1024**2)

        results.append({
            "complexity_value": value,
            "accuracy": float(accuracy),
            "latency_p90_us": latency["p90"],
            "model_size_mb": float(size_mb),
        })

    return pd.DataFrame(results)


def get_model_memory_footprint(
    model: BaseEstimator,
) -> Dict[str, float]:
    """
    Get detailed memory footprint of a model.

    Parameters
    ----------
    model : BaseEstimator
        Trained sklearn model

    Returns
    -------
    Dict[str, float]
        Dictionary with memory statistics:
        - size_bytes: Size in bytes
        - size_kb: Size in kilobytes
        - size_mb: Size in megabytes

    Examples
    --------
    >>> footprint = get_model_memory_footprint(rf_model)
    >>> print(f"Model size: {footprint['size_mb']:.2f} MB")
    """
    import pickle

    pickled = pickle.dumps(model)
    size_bytes = len(pickled)

    return {
        "size_bytes": float(size_bytes),
        "size_kb": float(size_bytes / 1024),
        "size_mb": float(size_bytes / (1024**2)),
    }


def check_latency_target(
    latency_us: float,
    target_us: float,
) -> Dict[str, Any]:
    """
    Check if latency meets target and calculate margin.

    Parameters
    ----------
    latency_us : float
        Measured latency in microseconds
    target_us : float
        Target latency in microseconds

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - meets_target: Boolean
        - margin_us: Margin (negative if exceeds target)
        - margin_pct: Margin as percentage of target

    Examples
    --------
    >>> result = check_latency_target(450, 500)
    >>> print(f"Meets target: {result['meets_target']}")
    >>> print(f"Margin: {result['margin_us']:.1f}μs ({result['margin_pct']:.1f}%)")
    """
    meets_target = latency_us < target_us
    margin_us = target_us - latency_us
    margin_pct = (margin_us / target_us) * 100

    return {
        "meets_target": bool(meets_target),
        "margin_us": float(margin_us),
        "margin_pct": float(margin_pct),
    }


def compare_latency_distributions(
    latency_results: Dict[str, Dict[str, float]],
    target_us: Optional[float] = None,
) -> pd.DataFrame:
    """
    Create comparison table of latency distributions.

    Parameters
    ----------
    latency_results : Dict[str, Dict[str, float]]
        Dictionary mapping names to latency statistics
    target_us : float, optional
        Target latency for comparison

    Returns
    -------
    pd.DataFrame
        Comparison table with all latency statistics

    Examples
    --------
    >>> results = {
    ...     'Model A': {'mean': 100, 'p90': 150, 'p99': 200},
    ...     'Model B': {'mean': 50, 'p90': 75, 'p99': 100},
    ... }
    >>> comparison = compare_latency_distributions(results, target_us=200)
    >>> print(comparison)
    """
    df = pd.DataFrame(latency_results).T

    if target_us is not None:
        df["meets_target"] = df["p90"] < target_us
        df["margin_us"] = target_us - df["p90"]

    return df


def simulate_load_test(
    prediction_fn: Callable,
    data_samples: List[Any],
    requests_per_second: int,
    duration_seconds: int = 10,
) -> Dict[str, Any]:
    """
    Simulate load testing for a prediction service.

    Tests how the system performs under sustained load.

    Parameters
    ----------
    prediction_fn : Callable
        Prediction function to test
    data_samples : List[Any]
        Pool of samples to randomly select from
    requests_per_second : int
        Target request rate
    duration_seconds : int, default=10
        Test duration in seconds

    Returns
    -------
    Dict[str, Any]
        Load test results:
        - total_requests: Number of requests processed
        - successful_requests: Number of successful predictions
        - failed_requests: Number of failures
        - latency_stats: Latency statistics
        - actual_rps: Actual requests per second achieved

    Examples
    --------
    >>> results = simulate_load_test(
    ...     lambda x: model.predict(x),
    ...     test_samples,
    ...     requests_per_second=1000,
    ...     duration_seconds=10
    ... )
    >>> print(f"Achieved RPS: {results['actual_rps']:.0f}")
    """
    import random

    target_interval = 1.0 / requests_per_second
    latencies = []
    successful = 0
    failed = 0

    start_time = time.perf_counter()
    next_request_time = start_time

    while (time.perf_counter() - start_time) < duration_seconds:
        # Wait until next request time
        current_time = time.perf_counter()
        if current_time < next_request_time:
            time.sleep(next_request_time - current_time)

        # Make request
        sample = random.choice(data_samples)

        try:
            request_start = time.perf_counter()
            _ = prediction_fn(sample)
            request_end = time.perf_counter()

            latencies.append((request_end - request_start) * 1_000_000)
            successful += 1
        except Exception:
            failed += 1

        # Schedule next request
        next_request_time += target_interval

    end_time = time.perf_counter()
    total_duration = end_time - start_time
    total_requests = successful + failed

    latencies = np.array(latencies) if latencies else np.array([0])

    return {
        "total_requests": total_requests,
        "successful_requests": successful,
        "failed_requests": failed,
        "actual_rps": total_requests / total_duration,
        "target_rps": requests_per_second,
        "latency_stats": {
            "mean": float(latencies.mean()),
            "p50": float(np.percentile(latencies, 50)),
            "p90": float(np.percentile(latencies, 90)),
            "p99": float(np.percentile(latencies, 99)),
        },
        "duration_seconds": total_duration,
    }
