"""
Regime-Conditional Modeling

This module provides functionality for training separate models for each market regime
and making regime-aware predictions. It supports comparing regime-conditional approaches
against global (regime-agnostic) models.

Key Functions:
- train_per_regime(): Train separate models for each regime
- predict_regime_aware(): Make predictions using regime-specific models
- compare_global_vs_conditional(): Compare performance metrics
- bootstrap_accuracy_test(): Statistical significance testing
- get_regime_feature_importance(): Extract feature importance by regime

Author: Rylan Spence
Date: 2024-12-17
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, List
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import warnings


def train_per_regime(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    regimes_train: np.ndarray,
    model_class: type,
    model_params: Optional[Dict[str, Any]] = None,
    min_samples: int = 20,
    adjust_params: bool = True,
    verbose: bool = True,
) -> Dict[str, BaseEstimator]:
    """
    Train separate models for each market regime.

    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training targets (n_samples,)
        regimes_train: Regime labels for training samples (n_samples,)
        model_class: Scikit-learn model class (e.g., RandomForestClassifier)
        model_params: Dictionary of model hyperparameters
        min_samples: Minimum samples required to train a regime-specific model
        adjust_params: Whether to adjust min_samples_split for small regimes
        verbose: Whether to print training progress

    Returns:
        Dictionary mapping regime names to trained models
        {regime_name: fitted_model, ...}
    """
    if model_params is None:
        model_params = {}

    regime_models = {}
    unique_regimes = np.unique(regimes_train)

    if verbose:
        print("=" * 80)
        print("TRAINING REGIME-SPECIFIC MODELS")
        print("=" * 80)

    for regime in unique_regimes:
        # Get regime-specific data
        regime_mask = regimes_train == regime
        X_regime = X_train[regime_mask]
        y_regime = y_train[regime_mask]

        if verbose:
            print(f"\n{regime} Regime:")
            print(f"  Training samples: {len(X_regime):,}")

        # Check if we have enough samples and both classes
        n_classes = len(np.unique(y_regime))

        if len(X_regime) < min_samples:
            if verbose:
                print(
                    f"  ⚠ Skipping - insufficient samples (need at least {min_samples})"
                )
                print(f"  → Will use global model for {regime} predictions")
            continue

        if n_classes < 2:
            if verbose:
                print(
                    f"  ⚠ Skipping - only {n_classes} class in training data (need 2)"
                )
                print(f"  → Will use global model for {regime} predictions")
            continue

        # Adjust model parameters for small sample sizes
        params = model_params.copy()
        if adjust_params and "min_samples_split" in params:
            # Ensure min_samples_split doesn't exceed 20% of regime samples
            max_split = max(2, len(X_regime) // 5)
            params["min_samples_split"] = min(params["min_samples_split"], max_split)

        # Train model for this regime
        try:
            model = model_class(**params)
            model.fit(X_regime, y_regime)
            regime_models[regime] = model

            if verbose:
                print(f"  ✓ Trained {model_class.__name__} for {regime}")

        except Exception as e:
            if verbose:
                print(f"  ✗ Error training model for {regime}: {str(e)}")
                print(f"  → Will use global model for {regime} predictions")
            continue

    if verbose:
        print(f"\n✓ Trained {len(regime_models)} regime-specific models")
        if len(regime_models) < len(unique_regimes):
            n_skipped = len(unique_regimes) - len(regime_models)
            print(
                f"  Note: {n_skipped} regime(s) will use the global model for predictions"
            )

    return regime_models


def predict_regime_aware(
    X_test: pd.DataFrame,
    regimes_test: np.ndarray,
    regime_models: Dict[str, BaseEstimator],
    global_model: BaseEstimator,
    return_proba: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Make predictions using regime-specific models when available, falling back to global model.

    Args:
        X_test: Test features (n_samples, n_features)
        regimes_test: Regime labels for test samples (n_samples,)
        regime_models: Dictionary of regime-specific models from train_per_regime()
        global_model: Fallback model for regimes without specific models
        return_proba: Whether to return probability predictions
        verbose: Whether to print prediction progress

    Returns:
        predictions: Class predictions (n_samples,)
        probabilities: Class probabilities (n_samples,) if return_proba=True, else None
    """
    n_samples = len(X_test)
    predictions = np.zeros(n_samples, dtype=int)
    probabilities = np.zeros(n_samples) if return_proba else None

    unique_regimes = np.unique(regimes_test)

    if verbose:
        print("Making regime-conditional predictions...")

    for regime in unique_regimes:
        # Get test samples for this regime
        regime_mask = regimes_test == regime
        X_regime_test = X_test[regime_mask]
        n_regime_samples = regime_mask.sum()

        # Use regime-specific model if available, otherwise use global model
        if regime in regime_models:
            model = regime_models[regime]
            model_type = "regime-specific"
        else:
            model = global_model
            model_type = "global"

        # Make predictions
        predictions[regime_mask] = model.predict(X_regime_test)

        if return_proba:
            probabilities[regime_mask] = model.predict_proba(X_regime_test)[:, 1]

        if verbose:
            print(f"  ✓ Predicted {n_regime_samples:,} {regime} samples ({model_type})")

    if verbose:
        print("✓ Regime-conditional predictions complete")

    return predictions, probabilities


def compare_global_vs_conditional(
    y_true: np.ndarray,
    y_pred_global: np.ndarray,
    y_pred_conditional: np.ndarray,
    y_proba_global: Optional[np.ndarray] = None,
    y_proba_conditional: Optional[np.ndarray] = None,
    regimes: Optional[np.ndarray] = None,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Compare performance between global and regime-conditional models.

    Args:
        y_true: True labels (n_samples,)
        y_pred_global: Predictions from global model (n_samples,)
        y_pred_conditional: Predictions from regime-conditional model (n_samples,)
        y_proba_global: Probability predictions from global model (optional)
        y_proba_conditional: Probability predictions from conditional model (optional)
        regimes: Regime labels for samples (optional, for per-regime analysis)
        model_name: Name of model type for reporting

    Returns:
        Dictionary containing comparison metrics and statistics
    """
    results = {
        "model_name": model_name,
        "global": {},
        "regime_conditional": {},
        "improvement": {},
        "per_regime": {},
    }

    # Calculate global model metrics
    results["global"]["accuracy"] = float(accuracy_score(y_true, y_pred_global))
    results["global"]["precision"] = float(
        precision_score(y_true, y_pred_global, zero_division=0)
    )
    results["global"]["recall"] = float(
        recall_score(y_true, y_pred_global, zero_division=0)
    )
    results["global"]["f1"] = float(f1_score(y_true, y_pred_global, zero_division=0))

    if y_proba_global is not None:
        results["global"]["roc_auc"] = float(roc_auc_score(y_true, y_proba_global))

    # Calculate regime-conditional model metrics
    results["regime_conditional"]["accuracy"] = float(
        accuracy_score(y_true, y_pred_conditional)
    )
    results["regime_conditional"]["precision"] = float(
        precision_score(y_true, y_pred_conditional, zero_division=0)
    )
    results["regime_conditional"]["recall"] = float(
        recall_score(y_true, y_pred_conditional, zero_division=0)
    )
    results["regime_conditional"]["f1"] = float(
        f1_score(y_true, y_pred_conditional, zero_division=0)
    )

    if y_proba_conditional is not None:
        results["regime_conditional"]["roc_auc"] = float(
            roc_auc_score(y_true, y_proba_conditional)
        )

    # Calculate improvements
    for metric in ["accuracy", "precision", "recall", "f1"]:
        global_val = results["global"][metric]
        conditional_val = results["regime_conditional"][metric]
        improvement = conditional_val - global_val
        pct_change = (
            100 * improvement / global_val if global_val > 0 else 0
        )

        results["improvement"][metric] = float(improvement)
        results["improvement"][f"{metric}_pct"] = float(pct_change)

    if y_proba_global is not None and y_proba_conditional is not None:
        global_auc = results["global"]["roc_auc"]
        conditional_auc = results["regime_conditional"]["roc_auc"]
        auc_improvement = conditional_auc - global_auc
        auc_pct = 100 * auc_improvement / global_auc if global_auc > 0 else 0

        results["improvement"]["roc_auc"] = float(auc_improvement)
        results["improvement"]["roc_auc_pct"] = float(auc_pct)

    # Per-regime analysis if regimes provided
    if regimes is not None:
        unique_regimes = np.unique(regimes)

        for regime in unique_regimes:
            regime_mask = regimes == regime
            y_regime = y_true[regime_mask]

            regime_results = {
                "n_samples": int(regime_mask.sum()),
                "global_accuracy": float(
                    accuracy_score(y_regime, y_pred_global[regime_mask])
                ),
                "conditional_accuracy": float(
                    accuracy_score(y_regime, y_pred_conditional[regime_mask])
                ),
            }

            regime_results["improvement"] = float(
                regime_results["conditional_accuracy"]
                - regime_results["global_accuracy"]
            )

            results["per_regime"][regime] = regime_results

    return results


def print_comparison_results(results: Dict[str, Any], verbose: bool = True) -> None:
    """
    Print formatted comparison results from compare_global_vs_conditional().

    Args:
        results: Results dictionary from compare_global_vs_conditional()
        verbose: Whether to print detailed per-regime results
    """
    print("=" * 80)
    print(f"{results['model_name'].upper()}: GLOBAL vs REGIME-CONDITIONAL COMPARISON")
    print("=" * 80)

    # Overall results
    print("\nGlobal Model (regime-agnostic):")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        if metric in results["global"]:
            print(f"  {metric.capitalize():10s}: {results['global'][metric]:.4f}")

    print("\nRegime-Conditional Model:")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        if metric in results["regime_conditional"]:
            print(
                f"  {metric.capitalize():10s}: {results['regime_conditional'][metric]:.4f}"
            )

    print("\nIMPROVEMENT (Regime-Conditional over Global):")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        if metric in results["improvement"]:
            improvement = results["improvement"][metric]
            pct = results["improvement"].get(f"{metric}_pct", 0)
            print(f"  {metric.capitalize():10s}: {improvement:+.4f} ({pct:+.1f}%)")

    # Per-regime results
    if verbose and results["per_regime"]:
        print("\n" + "=" * 80)
        print("PER-REGIME PERFORMANCE ANALYSIS")
        print("=" * 80)

        for regime, regime_results in results["per_regime"].items():
            print(f"\n{regime} Regime ({regime_results['n_samples']:,} samples):")
            print(f"  Global:      {regime_results['global_accuracy']:.4f}")
            print(
                f"  Conditional: {regime_results['conditional_accuracy']:.4f} ({regime_results['improvement']:+.4f})"
            )


def bootstrap_accuracy_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Bootstrap test for statistical significance of accuracy difference between two models.

    Args:
        y_true: True labels (n_samples,)
        y_pred1: Predictions from model 1 (baseline) (n_samples,)
        y_pred2: Predictions from model 2 (comparison) (n_samples,)
        n_bootstrap: Number of bootstrap iterations
        alpha: Significance level for confidence intervals
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing test results:
        - mean_diff: Mean accuracy difference (model2 - model1)
        - ci_lower: Lower bound of confidence interval
        - ci_upper: Upper bound of confidence interval
        - p_value: One-sided p-value (H1: model2 > model1)
        - significant: Whether result is significant at alpha level
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(y_true)
    diffs = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        acc1 = accuracy_score(y_true[indices], y_pred1[indices])
        acc2 = accuracy_score(y_true[indices], y_pred2[indices])
        diffs[i] = acc2 - acc1

    # Calculate statistics
    mean_diff = float(np.mean(diffs))
    ci_lower = float(np.percentile(diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    p_value = float(np.mean(diffs <= 0))  # One-sided test: model2 > model1

    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "significant": p_value < alpha,
        "alpha": alpha,
    }


def get_regime_feature_importance(
    regime_models: Dict[str, BaseEstimator],
    feature_names: List[str],
    top_n: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    Extract feature importance from regime-specific models.

    Args:
        regime_models: Dictionary of regime-specific models from train_per_regime()
        feature_names: List of feature names
        top_n: Number of top features to return per regime

    Returns:
        Dictionary mapping regime names to DataFrames with feature importances
        {regime_name: DataFrame(feature, importance), ...}
    """
    regime_importance = {}

    for regime, model in regime_models.items():
        # Check if model has feature_importances_ attribute
        if not hasattr(model, "feature_importances_"):
            warnings.warn(
                f"Model for {regime} does not have feature_importances_ attribute. Skipping."
            )
            continue

        # Create DataFrame with feature importances
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        # Keep top N features
        if top_n is not None:
            importance_df = importance_df.head(top_n)

        regime_importance[regime] = importance_df

    return regime_importance


def create_regime_performance_dataframe(
    y_true: np.ndarray,
    y_pred_global: np.ndarray,
    y_pred_conditional: np.ndarray,
    regimes: np.ndarray,
) -> pd.DataFrame:
    """
    Create a DataFrame summarizing per-regime performance for both global and conditional models.

    Args:
        y_true: True labels (n_samples,)
        y_pred_global: Predictions from global model (n_samples,)
        y_pred_conditional: Predictions from regime-conditional model (n_samples,)
        regimes: Regime labels for samples (n_samples,)

    Returns:
        DataFrame with columns:
        - regime: Regime name
        - n_samples: Number of samples in regime
        - global_accuracy: Global model accuracy
        - conditional_accuracy: Conditional model accuracy
        - improvement: Accuracy improvement (conditional - global)
    """
    unique_regimes = np.unique(regimes)
    regime_data = []

    for regime in unique_regimes:
        regime_mask = regimes == regime
        y_regime = y_true[regime_mask]

        global_acc = accuracy_score(y_regime, y_pred_global[regime_mask])
        conditional_acc = accuracy_score(y_regime, y_pred_conditional[regime_mask])

        regime_data.append(
            {
                "regime": regime,
                "n_samples": int(regime_mask.sum()),
                "global_accuracy": float(global_acc),
                "conditional_accuracy": float(conditional_acc),
                "improvement": float(conditional_acc - global_acc),
            }
        )

    return pd.DataFrame(regime_data)
