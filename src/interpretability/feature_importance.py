"""
Feature Importance Analysis Module

This module provides functions for computing and analyzing feature importance
using multiple methods:
- Built-in importances (Gini/Gain for tree-based models)
- Permutation importance (model-agnostic)
- SHAP values (game-theoretic attribution)

Extracted from notebooks 60_feature_importance.ipynb and 65_model_interpretability.ipynb
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Any
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
import warnings


def compute_builtin_importance(
    model: BaseEstimator,
    feature_names: List[str],
    model_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract built-in feature importances from tree-based models.

    For Decision Tree and Random Forest, this uses Gini importance (impurity reduction).
    For Gradient Boosting, this uses Gain-based importance (loss reduction).

    Parameters
    ----------
    model : BaseEstimator
        Trained sklearn tree-based model with feature_importances_ attribute
    feature_names : List[str]
        List of feature names corresponding to model features
    model_name : str, optional
        Name of the model (for display purposes)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature', 'importance', 'model_name']
        sorted by importance (descending)

    Raises
    ------
    AttributeError
        If model does not have feature_importances_ attribute

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> rf = RandomForestClassifier(n_estimators=100, random_state=42)
    >>> rf.fit(X_train, y_train)
    >>> importance_df = compute_builtin_importance(rf, feature_names, "Random Forest")
    >>> print(importance_df.head(10))
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"Model {type(model).__name__} does not have feature_importances_ attribute. "
            "This function only works with tree-based models."
        )

    if len(feature_names) != len(model.feature_importances_):
        raise ValueError(
            f"Length mismatch: {len(feature_names)} feature names provided, "
            f"but model has {len(model.feature_importances_)} features"
        )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    })

    if model_name is not None:
        importance_df["model_name"] = model_name

    # Sort by importance (descending)
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df


def compute_permutation_importance(
    model: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: str = "f1",
    n_jobs: int = -1,
    model_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute permutation importance for a trained model.

    Permutation importance measures the drop in model performance when a feature's
    values are randomly shuffled. This is a model-agnostic method that captures
    the true predictive power of each feature.

    Parameters
    ----------
    model : BaseEstimator
        Trained sklearn model
    X : pd.DataFrame or np.ndarray
        Feature matrix (test set recommended)
    y : np.ndarray
        Target vector
    feature_names : List[str]
        List of feature names
    n_repeats : int, default=10
        Number of times to permute each feature
    random_state : int, default=42
        Random seed for reproducibility
    scoring : str, default='f1'
        Scoring metric ('f1', 'accuracy', 'roc_auc', etc.)
    n_jobs : int, default=-1
        Number of parallel jobs (-1 uses all cores)
    model_name : str, optional
        Name of the model (for display purposes)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature', 'importance_mean', 'importance_std', 'model_name']
        sorted by importance_mean (descending)

    Examples
    --------
    >>> importance_df = compute_permutation_importance(
    ...     rf_model, X_test, y_test, feature_names,
    ...     n_repeats=10, scoring='f1'
    ... )
    >>> print(importance_df.head(10))
    """
    if isinstance(X, pd.DataFrame):
        if len(feature_names) != X.shape[1]:
            warnings.warn(
                f"Feature names length ({len(feature_names)}) does not match "
                f"X columns ({X.shape[1]}). Using provided feature_names."
            )
    else:
        if len(feature_names) != X.shape[1]:
            raise ValueError(
                f"Length mismatch: {len(feature_names)} feature names provided, "
                f"but X has {X.shape[1]} features"
            )

    # Compute permutation importance
    perm_result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=n_jobs,
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std,
    })

    if model_name is not None:
        importance_df["model_name"] = model_name

    # Sort by mean importance (descending)
    importance_df = importance_df.sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return importance_df


def rank_features(
    importance_values: Union[np.ndarray, pd.Series],
    feature_names: List[str],
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    """
    Rank features by importance values.

    Simple utility function to rank features by their importance scores.

    Parameters
    ----------
    importance_values : np.ndarray or pd.Series
        Array of importance values
    feature_names : List[str]
        List of feature names
    top_k : int, optional
        If provided, return only top K features

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['rank', 'feature', 'importance']
        sorted by importance (descending)

    Examples
    --------
    >>> feature_importances = model.feature_importances_
    >>> ranked_df = rank_features(feature_importances, feature_names, top_k=20)
    >>> print(ranked_df)
    """
    if len(importance_values) != len(feature_names):
        raise ValueError(
            f"Length mismatch: {len(importance_values)} importance values provided, "
            f"but {len(feature_names)} feature names"
        )

    # Create DataFrame
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance_values,
    })

    # Sort by importance (descending)
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    # Add rank column (1-indexed)
    df.insert(0, "rank", range(1, len(df) + 1))

    # Return top K if specified
    if top_k is not None:
        df = df.head(top_k)

    return df


def compare_importance_methods(
    builtin_importance: pd.DataFrame,
    permutation_importance: pd.DataFrame,
    shap_importance: Optional[pd.DataFrame] = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Compare feature importance across multiple methods.

    Merges importance scores from different methods and optionally normalizes them
    to enable cross-method comparison.

    Parameters
    ----------
    builtin_importance : pd.DataFrame
        DataFrame from compute_builtin_importance() with columns ['feature', 'importance']
    permutation_importance : pd.DataFrame
        DataFrame from compute_permutation_importance() with columns ['feature', 'importance_mean']
    shap_importance : pd.DataFrame, optional
        DataFrame with SHAP importances with columns ['feature', 'importance']
    normalize : bool, default=True
        If True, normalize each importance method to sum to 1

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature', 'builtin', 'permutation', 'shap' (if provided), 'avg_importance']
        sorted by avg_importance (descending)

    Examples
    --------
    >>> comparison_df = compare_importance_methods(
    ...     builtin_df, permutation_df, shap_df, normalize=True
    ... )
    >>> print(comparison_df.head(20))
    """
    # Start with builtin importance
    comparison = builtin_importance[["feature", "importance"]].copy()
    comparison.rename(columns={"importance": "builtin"}, inplace=True)

    # Merge permutation importance
    perm_df = permutation_importance[["feature", "importance_mean"]].copy()
    perm_df.rename(columns={"importance_mean": "permutation"}, inplace=True)
    comparison = comparison.merge(perm_df, on="feature", how="outer")

    # Merge SHAP importance if provided
    if shap_importance is not None:
        shap_df = shap_importance[["feature", "importance"]].copy()
        shap_df.rename(columns={"importance": "shap"}, inplace=True)
        comparison = comparison.merge(shap_df, on="feature", how="outer")

    # Fill NaN values with 0
    comparison = comparison.fillna(0)

    # Normalize if requested
    if normalize:
        for col in comparison.columns:
            if col != "feature":
                col_sum = comparison[col].sum()
                if col_sum > 0:
                    comparison[col] = comparison[col] / col_sum

    # Compute average importance across methods
    importance_cols = [col for col in comparison.columns if col != "feature"]
    comparison["avg_importance"] = comparison[importance_cols].mean(axis=1)

    # Sort by average importance (descending)
    comparison = comparison.sort_values("avg_importance", ascending=False).reset_index(drop=True)

    return comparison


def get_top_features(
    importance_df: pd.DataFrame,
    top_k: int = 10,
    importance_col: str = "importance",
) -> List[str]:
    """
    Get list of top K most important features.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with feature importance (from any importance function)
    top_k : int, default=10
        Number of top features to return
    importance_col : str, default='importance'
        Name of the importance column to use for ranking

    Returns
    -------
    List[str]
        List of top K feature names

    Examples
    --------
    >>> top_features = get_top_features(importance_df, top_k=20)
    >>> print(f"Top 20 features: {top_features}")
    """
    if importance_col not in importance_df.columns:
        raise ValueError(
            f"Importance column '{importance_col}' not found in DataFrame. "
            f"Available columns: {importance_df.columns.tolist()}"
        )

    # Sort by importance and get top K
    sorted_df = importance_df.sort_values(importance_col, ascending=False)
    top_features = sorted_df["feature"].head(top_k).tolist()

    return top_features


def compute_importance_concentration(
    importance_df: pd.DataFrame,
    percentiles: List[int] = [10, 20, 50],
    importance_col: str = "importance",
) -> Dict[str, float]:
    """
    Compute concentration of importance in top features.

    Calculates what percentage of total importance is captured by the
    top 10%, 20%, 50% of features (by default).

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with feature importance
    percentiles : List[int], default=[10, 20, 50]
        List of percentiles to compute (as percentages)
    importance_col : str, default='importance'
        Name of the importance column

    Returns
    -------
    Dict[str, float]
        Dictionary mapping percentile to cumulative importance percentage

    Examples
    --------
    >>> concentration = compute_importance_concentration(importance_df)
    >>> print(f"Top 10 features: {concentration['top_10_pct']:.1f}%")
    >>> print(f"Top 20 features: {concentration['top_20_pct']:.1f}%")
    """
    if importance_col not in importance_df.columns:
        raise ValueError(
            f"Importance column '{importance_col}' not found in DataFrame"
        )

    # Sort by importance (descending)
    sorted_df = importance_df.sort_values(importance_col, ascending=False).copy()

    # Compute cumulative sum
    total_importance = sorted_df[importance_col].sum()
    sorted_df["cumulative_pct"] = (
        sorted_df[importance_col].cumsum() / total_importance * 100
    )

    # Compute concentration for each percentile
    n_features = len(sorted_df)
    concentration = {}

    for pct in percentiles:
        # Number of features in top X%
        n_top = max(1, int(np.ceil(n_features * pct / 100)))

        # Cumulative importance at that point
        cumulative_importance = sorted_df.iloc[n_top - 1]["cumulative_pct"]

        concentration[f"top_{pct}_pct"] = float(cumulative_importance)

    return concentration


def filter_features_by_importance(
    X: pd.DataFrame,
    importance_df: pd.DataFrame,
    threshold: float = 0.01,
    importance_col: str = "importance",
) -> pd.DataFrame:
    """
    Filter feature matrix to keep only important features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    importance_df : pd.DataFrame
        DataFrame with feature importance
    threshold : float, default=0.01
        Minimum importance threshold (features below this are dropped)
    importance_col : str, default='importance'
        Name of the importance column

    Returns
    -------
    pd.DataFrame
        Filtered feature matrix with only important features

    Examples
    --------
    >>> X_filtered = filter_features_by_importance(X, importance_df, threshold=0.01)
    >>> print(f"Features before: {X.shape[1]}, after: {X_filtered.shape[1]}")
    """
    if importance_col not in importance_df.columns:
        raise ValueError(
            f"Importance column '{importance_col}' not found in DataFrame"
        )

    # Get features above threshold
    important_features = importance_df[
        importance_df[importance_col] >= threshold
    ]["feature"].tolist()

    # Filter X to keep only important features
    # Handle case where some features might not be in X
    available_features = [f for f in important_features if f in X.columns]

    if len(available_features) < len(important_features):
        warnings.warn(
            f"Only {len(available_features)} out of {len(important_features)} "
            f"important features found in X"
        )

    X_filtered = X[available_features].copy()

    return X_filtered
