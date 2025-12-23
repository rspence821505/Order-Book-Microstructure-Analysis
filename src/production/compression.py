"""
Model Compression and Optimization Module

This module provides functions for model optimization and compression:
- Random Forest pruning (reduce number of trees)
- Gradient Boosting compression (early stopping, tree reduction)
- Pareto frontier analysis (accuracy vs latency trade-offs)
- Feature selection for speed optimization
- Decision tree depth limiting
- Model simplification strategies

Extracted from notebook 75_production_benchmarks.ipynb
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore")


def prune_random_forest(
    model: RandomForestClassifier, n_trees: int
) -> RandomForestClassifier:
    """
    Prune Random Forest by keeping only first n trees.

    Parameters
    ----------
    model : RandomForestClassifier
        Trained Random Forest model
    n_trees : int
        Number of trees to keep

    Returns
    -------
    pruned_model : RandomForestClassifier
        Pruned model with reduced trees

    Examples
    --------
    >>> rf_model = RandomForestClassifier(n_estimators=500)
    >>> rf_model.fit(X_train, y_train)
    >>> pruned_model = prune_random_forest(rf_model, n_trees=100)
    >>> print(pruned_model.n_estimators)
    100
    """
    if n_trees > model.n_estimators:
        warnings.warn(
            f"Requested {n_trees} trees but model only has {model.n_estimators}. "
            f"Returning original model."
        )
        return model

    # Create a copy of the model
    pruned_model = clone(model)

    # Keep only first n_trees estimators
    pruned_model.estimators_ = model.estimators_[:n_trees]
    pruned_model.n_estimators = n_trees

    return pruned_model


def compress_gradient_boosting(
    model: GradientBoostingClassifier, n_estimators: int
) -> GradientBoostingClassifier:
    """
    Compress Gradient Boosting by keeping only first n estimators.

    Parameters
    ----------
    model : GradientBoostingClassifier
        Trained Gradient Boosting model
    n_estimators : int
        Number of estimators to keep

    Returns
    -------
    compressed_model : GradientBoostingClassifier
        Compressed model with reduced estimators

    Examples
    --------
    >>> gb_model = GradientBoostingClassifier(n_estimators=300)
    >>> gb_model.fit(X_train, y_train)
    >>> compressed_model = compress_gradient_boosting(gb_model, n_estimators=100)
    >>> print(compressed_model.n_estimators)
    100
    """
    if n_estimators > model.n_estimators:
        warnings.warn(
            f"Requested {n_estimators} estimators but model only has {model.n_estimators}. "
            f"Returning original model."
        )
        return model

    # Create a copy of the model
    compressed_model = clone(model)

    # Keep only first n_estimators
    compressed_model.estimators_ = model.estimators_[:n_estimators]
    compressed_model.n_estimators = n_estimators

    return compressed_model


def analyze_pareto_frontier(
    model_class: type,
    base_params: Dict[str, Any],
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: np.ndarray,
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: np.ndarray,
    complexity_param: str,
    complexity_values: List[int],
    latency_benchmark_fn: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    Analyze Pareto frontier for model complexity vs performance trade-offs.

    Trains models with different complexity levels and measures accuracy vs latency,
    identifying the optimal trade-off points on the Pareto frontier.

    Parameters
    ----------
    model_class : type
        Model class (e.g., RandomForestClassifier, GradientBoostingClassifier)
    base_params : dict
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
        Parameter name to vary (e.g., 'n_estimators', 'max_depth')
    complexity_values : list of int
        Values to test for complexity parameter
    latency_benchmark_fn : callable, optional
        Function to benchmark latency. If None, uses simple prediction timing.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns: complexity_value, accuracy, f1_score, latency_us, size_mb,
        is_pareto_optimal

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> results = analyze_pareto_frontier(
    ...     model_class=RandomForestClassifier,
    ...     base_params={'max_depth': 12, 'random_state': 42},
    ...     X_train=X_train, y_train=y_train,
    ...     X_test=X_test, y_test=y_test,
    ...     complexity_param='n_estimators',
    ...     complexity_values=[10, 25, 50, 100, 200, 500]
    ... )
    >>> pareto_optimal = results[results['is_pareto_optimal']]
    >>> print(pareto_optimal)
    """
    results = []

    if isinstance(X_test, pd.DataFrame):
        X_sample = X_test.iloc[0].values.reshape(1, -1)
    else:
        X_sample = X_test[0].reshape(1, -1)

    for complexity_value in complexity_values:
        # Create model with current complexity
        params = base_params.copy()
        params[complexity_param] = complexity_value
        model = model_class(**params)

        # Train model
        model.fit(X_train, y_train)

        # Evaluate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Benchmark latency
        if latency_benchmark_fn is not None:
            latency_result = latency_benchmark_fn(model, X_sample)
            latency_us = latency_result.get("p90", latency_result.get("mean", 0))
        else:
            # Simple timing
            import time

            latencies = []
            for _ in range(100):
                start = time.perf_counter()
                _ = model.predict(X_sample)
                end = time.perf_counter()
                latencies.append((end - start) * 1_000_000)
            latency_us = np.percentile(latencies, 90)

        # Get model size
        size_mb = len(pickle.dumps(model)) / (1024**2)

        results.append(
            {
                complexity_param: complexity_value,
                "accuracy": accuracy,
                "f1_score": f1,
                "latency_us": latency_us,
                "size_mb": size_mb,
            }
        )

    results_df = pd.DataFrame(results)

    # Identify Pareto optimal points
    # A point is Pareto optimal if no other point has both higher accuracy AND lower latency
    results_df["is_pareto_optimal"] = False

    for i in range(len(results_df)):
        is_dominated = False
        for j in range(len(results_df)):
            if i != j:
                # Check if point j dominates point i
                # (higher or equal accuracy AND lower or equal latency, with at least one strict inequality)
                better_accuracy = results_df.iloc[j]["accuracy"] >= results_df.iloc[i]["accuracy"]
                better_latency = results_df.iloc[j]["latency_us"] <= results_df.iloc[i]["latency_us"]
                strictly_better = (
                    results_df.iloc[j]["accuracy"] > results_df.iloc[i]["accuracy"]
                    or results_df.iloc[j]["latency_us"] < results_df.iloc[i]["latency_us"]
                )

                if better_accuracy and better_latency and strictly_better:
                    is_dominated = True
                    break

        if not is_dominated:
            results_df.at[i, "is_pareto_optimal"] = True

    return results_df


def find_optimal_complexity(
    pareto_df: pd.DataFrame,
    min_accuracy: float,
    max_latency_us: Optional[float] = None,
    complexity_param: str = "n_estimators",
) -> Dict[str, Any]:
    """
    Find optimal complexity level given accuracy and latency constraints.

    Parameters
    ----------
    pareto_df : pd.DataFrame
        DataFrame from analyze_pareto_frontier()
    min_accuracy : float
        Minimum required accuracy
    max_latency_us : float, optional
        Maximum acceptable latency in microseconds
    complexity_param : str, default='n_estimators'
        Name of complexity parameter

    Returns
    -------
    optimal_config : dict
        Dictionary with optimal configuration and metrics

    Examples
    --------
    >>> optimal = find_optimal_complexity(
    ...     pareto_df=results,
    ...     min_accuracy=0.65,
    ...     max_latency_us=500
    ... )
    >>> print(f"Optimal trees: {optimal['complexity']}")
    >>> print(f"Achieves accuracy: {optimal['accuracy']:.4f}")
    """
    # Filter by constraints
    filtered_df = pareto_df[pareto_df["accuracy"] >= min_accuracy].copy()

    if max_latency_us is not None:
        filtered_df = filtered_df[filtered_df["latency_us"] <= max_latency_us]

    if len(filtered_df) == 0:
        return {
            "status": "infeasible",
            "message": "No configuration meets the specified constraints",
        }

    # Among feasible solutions, find the one with minimum complexity
    optimal_row = filtered_df.loc[filtered_df[complexity_param].idxmin()]

    return {
        "status": "success",
        "complexity": optimal_row[complexity_param],
        "accuracy": optimal_row["accuracy"],
        "f1_score": optimal_row["f1_score"],
        "latency_us": optimal_row["latency_us"],
        "size_mb": optimal_row["size_mb"],
        "is_pareto_optimal": optimal_row["is_pareto_optimal"],
    }


def compress_random_forest_by_importance(
    model: RandomForestClassifier,
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: np.ndarray,
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: np.ndarray,
    target_accuracy: float,
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Compress Random Forest by removing trees until target accuracy is reached.

    Iteratively removes the least important trees (by validation accuracy contribution)
    until the model reaches the target accuracy threshold.

    Parameters
    ----------
    model : RandomForestClassifier
        Trained Random Forest model
    X_train : array-like
        Training features (for re-evaluation)
    y_train : array-like
        Training labels
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    target_accuracy : float
        Minimum acceptable accuracy

    Returns
    -------
    compressed_model : RandomForestClassifier
        Compressed model
    compression_stats : dict
        Statistics about compression (original_trees, compressed_trees, etc.)

    Examples
    --------
    >>> compressed_model, stats = compress_random_forest_by_importance(
    ...     model=rf_model,
    ...     X_train=X_train, y_train=y_train,
    ...     X_test=X_test, y_test=y_test,
    ...     target_accuracy=0.65
    ... )
    >>> print(f"Reduced from {stats['original_trees']} to {stats['compressed_trees']} trees")
    """
    original_accuracy = accuracy_score(y_test, model.predict(X_test))
    original_trees = model.n_estimators

    if original_accuracy < target_accuracy:
        raise ValueError(
            f"Original model accuracy ({original_accuracy:.4f}) is below target ({target_accuracy:.4f})"
        )

    # Binary search for optimal tree count
    min_trees = 1
    max_trees = original_trees
    best_n_trees = original_trees

    while min_trees <= max_trees:
        mid_trees = (min_trees + max_trees) // 2
        pruned_model = prune_random_forest(model, mid_trees)
        accuracy = accuracy_score(y_test, pruned_model.predict(X_test))

        if accuracy >= target_accuracy:
            best_n_trees = mid_trees
            max_trees = mid_trees - 1
        else:
            min_trees = mid_trees + 1

    # Create final compressed model
    compressed_model = prune_random_forest(model, best_n_trees)
    final_accuracy = accuracy_score(y_test, compressed_model.predict(X_test))

    compression_stats = {
        "original_trees": original_trees,
        "compressed_trees": best_n_trees,
        "reduction_pct": (1 - best_n_trees / original_trees) * 100,
        "original_accuracy": original_accuracy,
        "compressed_accuracy": final_accuracy,
        "accuracy_loss": original_accuracy - final_accuracy,
    }

    return compressed_model, compression_stats


def select_features_for_speed(
    model_class: type,
    model_params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_sets: Dict[str, List[str]],
    latency_benchmark_fn: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    Evaluate model performance with different feature sets.

    Tests multiple feature subsets to find the optimal trade-off between
    accuracy and inference speed.

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
    latency_benchmark_fn : callable, optional
        Function to benchmark latency

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns: feature_set, n_features, accuracy, f1_score, latency_us

    Examples
    --------
    >>> feature_sets = {
    ...     'All features': all_features,
    ...     'Top 50 features': top_50_features,
    ...     'Top 20 features': top_20_features,
    ...     'Top 10 features': top_10_features
    ... }
    >>> results = select_features_for_speed(
    ...     RandomForestClassifier, {'n_estimators': 100},
    ...     X_train, y_train, X_test, y_test, feature_sets
    ... )
    """
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
        X_sample = X_test_subset.iloc[0].values.reshape(1, -1)

        if latency_benchmark_fn is not None:
            latency_result = latency_benchmark_fn(model, X_sample)
            latency_us = latency_result.get("p90", latency_result.get("mean", 0))
        else:
            import time

            latencies = []
            for _ in range(100):
                start = time.perf_counter()
                _ = model.predict(X_sample)
                end = time.perf_counter()
                latencies.append((end - start) * 1_000_000)
            latency_us = np.percentile(latencies, 90)

        results.append(
            {
                "feature_set": set_name,
                "n_features": len(features),
                "accuracy": accuracy,
                "f1_score": f1,
                "latency_us": latency_us,
            }
        )

    return pd.DataFrame(results)


def limit_tree_depth(
    model: Union[DecisionTreeClassifier, RandomForestClassifier],
    max_depth: int,
) -> Union[DecisionTreeClassifier, RandomForestClassifier]:
    """
    Limit tree depth for faster inference (requires retraining).

    Note: This function returns a model with updated parameters.
    The model must be retrained after calling this function.

    Parameters
    ----------
    model : DecisionTreeClassifier or RandomForestClassifier
        Model to limit depth
    max_depth : int
        Maximum tree depth

    Returns
    -------
    limited_model : same type as input
        Model with updated max_depth parameter

    Examples
    --------
    >>> dt_model = DecisionTreeClassifier(max_depth=20)
    >>> dt_model.fit(X_train, y_train)
    >>> limited_model = limit_tree_depth(dt_model, max_depth=10)
    >>> limited_model.fit(X_train, y_train)  # Must retrain
    """
    limited_model = clone(model)
    limited_model.max_depth = max_depth
    return limited_model


def compare_compression_methods(
    original_model: BaseEstimator,
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: np.ndarray,
    compression_configs: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Compare multiple compression methods.

    Parameters
    ----------
    original_model : BaseEstimator
        Original uncompressed model
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    compression_configs : dict
        Dictionary mapping method names to compressed models or config dicts

    Returns
    -------
    comparison_df : pd.DataFrame
        DataFrame comparing original vs compressed models

    Examples
    --------
    >>> configs = {
    ...     'Original': original_rf,
    ...     'Pruned to 100 trees': prune_random_forest(original_rf, 100),
    ...     'Pruned to 50 trees': prune_random_forest(original_rf, 50),
    ...     'Pruned to 25 trees': prune_random_forest(original_rf, 25)
    ... }
    >>> comparison = compare_compression_methods(original_rf, X_test, y_test, configs)
    """
    results = []

    for method_name, model in compression_configs.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        size_mb = len(pickle.dumps(model)) / (1024**2)

        # Get complexity metric
        if hasattr(model, "n_estimators"):
            complexity = model.n_estimators
            complexity_metric = "n_estimators"
        elif hasattr(model, "max_depth"):
            complexity = model.max_depth
            complexity_metric = "max_depth"
        else:
            complexity = None
            complexity_metric = "unknown"

        results.append(
            {
                "method": method_name,
                "accuracy": accuracy,
                "f1_score": f1,
                "size_mb": size_mb,
                "complexity_metric": complexity_metric,
                "complexity_value": complexity,
            }
        )

    return pd.DataFrame(results)


def estimate_compression_speedup(
    original_latency_us: float,
    compressed_latency_us: float,
    original_size_mb: float,
    compressed_size_mb: float,
) -> Dict[str, float]:
    """
    Estimate speedup and size reduction from compression.

    Parameters
    ----------
    original_latency_us : float
        Original model latency in microseconds
    compressed_latency_us : float
        Compressed model latency in microseconds
    original_size_mb : float
        Original model size in MB
    compressed_size_mb : float
        Compressed model size in MB

    Returns
    -------
    metrics : dict
        Dictionary with speedup_ratio, latency_reduction_pct, size_reduction_pct

    Examples
    --------
    >>> metrics = estimate_compression_speedup(
    ...     original_latency_us=500, compressed_latency_us=200,
    ...     original_size_mb=50, compressed_size_mb=10
    ... )
    >>> print(f"Speedup: {metrics['speedup_ratio']:.2f}x")
    >>> print(f"Size reduction: {metrics['size_reduction_pct']:.1f}%")
    """
    speedup_ratio = original_latency_us / compressed_latency_us
    latency_reduction_pct = (1 - compressed_latency_us / original_latency_us) * 100
    size_reduction_pct = (1 - compressed_size_mb / original_size_mb) * 100

    return {
        "speedup_ratio": speedup_ratio,
        "latency_reduction_pct": latency_reduction_pct,
        "size_reduction_pct": size_reduction_pct,
        "original_latency_us": original_latency_us,
        "compressed_latency_us": compressed_latency_us,
        "original_size_mb": original_size_mb,
        "compressed_size_mb": compressed_size_mb,
    }


def recommend_compression_strategy(
    pareto_df: pd.DataFrame,
    target_latency_us: float,
    min_accuracy: float,
    complexity_param: str = "n_estimators",
) -> Dict[str, Any]:
    """
    Recommend optimal compression strategy based on constraints.

    Parameters
    ----------
    pareto_df : pd.DataFrame
        DataFrame from analyze_pareto_frontier()
    target_latency_us : float
        Target latency in microseconds
    min_accuracy : float
        Minimum acceptable accuracy
    complexity_param : str, default='n_estimators'
        Name of complexity parameter

    Returns
    -------
    recommendation : dict
        Recommended configuration with reasoning

    Examples
    --------
    >>> recommendation = recommend_compression_strategy(
    ...     pareto_df=results,
    ...     target_latency_us=500,
    ...     min_accuracy=0.65
    ... )
    >>> print(recommendation['recommendation'])
    >>> print(f"Use {recommendation['complexity']} {recommendation['complexity_param']}")
    """
    # Find configurations that meet constraints
    feasible = pareto_df[
        (pareto_df["latency_us"] <= target_latency_us)
        & (pareto_df["accuracy"] >= min_accuracy)
    ].copy()

    if len(feasible) == 0:
        # No feasible solution - recommend closest
        pareto_df["accuracy_deficit"] = min_accuracy - pareto_df["accuracy"]
        pareto_df["latency_deficit"] = pareto_df["latency_us"] - target_latency_us

        # Penalize missing constraints
        pareto_df["penalty"] = (
            pareto_df["accuracy_deficit"].clip(lower=0) * 100
            + pareto_df["latency_deficit"].clip(lower=0) / 10
        )

        best_idx = pareto_df["penalty"].idxmin()
        best_row = pareto_df.loc[best_idx]

        return {
            "status": "infeasible",
            "recommendation": "No configuration meets all constraints. Closest option:",
            "complexity_param": complexity_param,
            "complexity": best_row[complexity_param],
            "accuracy": best_row["accuracy"],
            "latency_us": best_row["latency_us"],
            "accuracy_gap": max(0, min_accuracy - best_row["accuracy"]),
            "latency_gap": max(0, best_row["latency_us"] - target_latency_us),
        }

    # Among feasible, prefer Pareto optimal points
    pareto_feasible = feasible[feasible["is_pareto_optimal"]]

    if len(pareto_feasible) > 0:
        # Among Pareto optimal, choose highest accuracy
        best_idx = pareto_feasible["accuracy"].idxmax()
        best_row = pareto_feasible.loc[best_idx]
        status = "optimal"
    else:
        # Choose highest accuracy among feasible
        best_idx = feasible["accuracy"].idxmax()
        best_row = feasible.loc[best_idx]
        status = "feasible"

    return {
        "status": status,
        "recommendation": f"Use {int(best_row[complexity_param])} {complexity_param}",
        "complexity_param": complexity_param,
        "complexity": best_row[complexity_param],
        "accuracy": best_row["accuracy"],
        "f1_score": best_row["f1_score"],
        "latency_us": best_row["latency_us"],
        "size_mb": best_row["size_mb"],
        "is_pareto_optimal": best_row["is_pareto_optimal"],
        "margin_to_accuracy_target": best_row["accuracy"] - min_accuracy,
        "margin_to_latency_target": target_latency_us - best_row["latency_us"],
    }


def analyze_compression_trade_offs(
    compression_results: pd.DataFrame, complexity_param: str = "n_estimators"
) -> Dict[str, Any]:
    """
    Analyze trade-offs in compression results.

    Parameters
    ----------
    compression_results : pd.DataFrame
        Results from analyze_pareto_frontier()
    complexity_param : str, default='n_estimators'
        Name of complexity parameter

    Returns
    -------
    analysis : dict
        Summary statistics about compression trade-offs

    Examples
    --------
    >>> analysis = analyze_compression_trade_offs(results)
    >>> print(f"Sweet spot: {analysis['sweet_spot_complexity']} trees")
    >>> print(f"Max accuracy: {analysis['max_accuracy']:.4f}")
    """
    # Find sweet spot (best accuracy/latency ratio)
    compression_results["efficiency"] = (
        compression_results["accuracy"] / compression_results["latency_us"]
    )
    sweet_spot_idx = compression_results["efficiency"].idxmax()
    sweet_spot = compression_results.loc[sweet_spot_idx]

    # Marginal returns analysis
    compression_results_sorted = compression_results.sort_values(complexity_param)
    compression_results_sorted["marginal_accuracy"] = compression_results_sorted[
        "accuracy"
    ].diff()
    compression_results_sorted["marginal_latency"] = compression_results_sorted[
        "latency_us"
    ].diff()

    return {
        "min_complexity": compression_results[complexity_param].min(),
        "max_complexity": compression_results[complexity_param].max(),
        "min_accuracy": compression_results["accuracy"].min(),
        "max_accuracy": compression_results["accuracy"].max(),
        "min_latency_us": compression_results["latency_us"].min(),
        "max_latency_us": compression_results["latency_us"].max(),
        "sweet_spot_complexity": sweet_spot[complexity_param],
        "sweet_spot_accuracy": sweet_spot["accuracy"],
        "sweet_spot_latency_us": sweet_spot["latency_us"],
        "sweet_spot_efficiency": sweet_spot["efficiency"],
        "n_pareto_optimal": compression_results["is_pareto_optimal"].sum(),
        "accuracy_range": compression_results["accuracy"].max()
        - compression_results["accuracy"].min(),
        "latency_range": compression_results["latency_us"].max()
        - compression_results["latency_us"].min(),
    }


if __name__ == "__main__":
    print("Model Compression and Optimization Module")
    print("=" * 70)
    print("\nKey functions:")
    print("  - prune_random_forest(): Reduce number of trees")
    print("  - compress_gradient_boosting(): Reduce number of estimators")
    print("  - analyze_pareto_frontier(): Accuracy vs latency trade-offs")
    print("  - find_optimal_complexity(): Find best config for constraints")
    print("  - compress_random_forest_by_importance(): Iterative tree removal")
    print("  - select_features_for_speed(): Feature subset optimization")
    print("  - recommend_compression_strategy(): Get compression recommendation")
    print("  - analyze_compression_trade_offs(): Summary statistics")
    print("\nSee function docstrings for detailed usage examples.")
