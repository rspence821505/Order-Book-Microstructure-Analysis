"""
Regime-Conditional Prediction Module

This module provides functions for making regime-aware predictions:
- Regime-conditional prediction using regime-specific models
- Ensemble predictions combining multiple models
- Confidence adjustment based on regime characteristics
- Model selection based on market regime

Extracted from notebooks 55_regime_conditional_models.ipynb and 70_regime_validation.ipynb
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.base import BaseEstimator
import warnings


def predict_with_regime(
    X: Union[pd.DataFrame, np.ndarray],
    regimes: Union[pd.Series, np.ndarray],
    regime_models: Dict[str, BaseEstimator],
    global_model: Optional[BaseEstimator] = None,
    return_proba: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Make regime-conditional predictions using regime-specific models.

    For each sample, uses the model trained for its specific regime.
    Falls back to global model if regime-specific model is not available.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    regimes : pd.Series or np.ndarray
        Regime labels for each sample
    regime_models : Dict[str, BaseEstimator]
        Dictionary mapping regime names to trained models
    global_model : BaseEstimator, optional
        Fallback model to use when regime-specific model is unavailable
    return_proba : bool, default=True
        Whether to return prediction probabilities

    Returns
    -------
    predictions : np.ndarray
        Predicted class labels (0 or 1)
    probabilities : np.ndarray, optional
        Predicted probabilities for class 1 (returned if return_proba=True)

    Raises
    ------
    ValueError
        If no model is available for a regime and no global_model provided

    Examples
    --------
    >>> predictions, probabilities = predict_with_regime(
    ...     X_test, regimes_test, regime_models, global_model
    ... )
    >>> print(f"Predictions: {predictions[:10]}")
    >>> print(f"Probabilities: {probabilities[:10]}")
    """
    # Convert inputs to numpy arrays if needed
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X

    if isinstance(regimes, pd.Series):
        regimes_array = regimes.values
    else:
        regimes_array = regimes

    # Initialize output arrays
    n_samples = len(X_array)
    predictions = np.zeros(n_samples, dtype=int)
    probabilities = np.zeros(n_samples, dtype=float)

    # Get unique regimes
    unique_regimes = np.unique(regimes_array)

    # Make predictions for each regime
    for regime in unique_regimes:
        # Get samples for this regime
        regime_mask = regimes_array == regime
        X_regime = X_array[regime_mask]

        if len(X_regime) == 0:
            continue

        # Check if we have a regime-specific model
        if regime in regime_models:
            model = regime_models[regime]
        elif global_model is not None:
            model = global_model
            warnings.warn(
                f"No model for regime '{regime}', using global model as fallback"
            )
        else:
            raise ValueError(
                f"No model available for regime '{regime}' and no global_model provided"
            )

        # Make predictions
        predictions[regime_mask] = model.predict(X_regime)

        if return_proba:
            proba = model.predict_proba(X_regime)
            # Get probability for class 1
            if proba.ndim == 2 and proba.shape[1] == 2:
                probabilities[regime_mask] = proba[:, 1]
            else:
                probabilities[regime_mask] = proba

    if return_proba:
        return predictions, probabilities
    else:
        return predictions


def ensemble_predictions(
    X: Union[pd.DataFrame, np.ndarray],
    models: Dict[str, BaseEstimator],
    weights: Optional[Dict[str, float]] = None,
    voting: str = "soft",
    return_proba: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Ensemble predictions from multiple models.

    Combines predictions from different models using weighted voting.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    models : Dict[str, BaseEstimator]
        Dictionary mapping model names to trained models
    weights : Dict[str, float], optional
        Dictionary mapping model names to weights
        If None, uses equal weights
    voting : str, default='soft'
        Voting method:
        - 'soft': Weighted average of predicted probabilities
        - 'hard': Weighted majority vote of predicted classes
    return_proba : bool, default=True
        Whether to return prediction probabilities

    Returns
    -------
    predictions : np.ndarray
        Ensemble predicted class labels
    probabilities : np.ndarray, optional
        Ensemble predicted probabilities (returned if return_proba=True)

    Examples
    --------
    >>> models = {'RF': rf_model, 'GB': gb_model, 'DT': dt_model}
    >>> weights = {'RF': 0.5, 'GB': 0.3, 'DT': 0.2}
    >>> predictions, probabilities = ensemble_predictions(
    ...     X_test, models, weights, voting='soft'
    ... )
    """
    # Convert to array if needed
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X

    # Set equal weights if not provided
    if weights is None:
        weights = {name: 1.0 / len(models) for name in models.keys()}

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    n_samples = len(X_array)

    if voting == "soft":
        # Weighted average of probabilities
        ensemble_proba = np.zeros(n_samples)

        for model_name, model in models.items():
            weight = normalized_weights.get(model_name, 0)
            proba = model.predict_proba(X_array)

            # Get probability for class 1
            if proba.ndim == 2 and proba.shape[1] == 2:
                proba_class1 = proba[:, 1]
            else:
                proba_class1 = proba

            ensemble_proba += weight * proba_class1

        # Convert probabilities to predictions
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)

        if return_proba:
            return ensemble_pred, ensemble_proba
        else:
            return ensemble_pred

    elif voting == "hard":
        # Weighted majority vote
        vote_scores = np.zeros((n_samples, 2))  # [down_votes, up_votes]

        for model_name, model in models.items():
            weight = normalized_weights.get(model_name, 0)
            pred = model.predict(X_array)

            # Add weighted votes
            for i, p in enumerate(pred):
                vote_scores[i, int(p)] += weight

        # Get class with most votes
        ensemble_pred = np.argmax(vote_scores, axis=1)

        if return_proba:
            # Convert vote scores to probabilities
            ensemble_proba = vote_scores[:, 1] / vote_scores.sum(axis=1)
            return ensemble_pred, ensemble_proba
        else:
            return ensemble_pred

    else:
        raise ValueError(f"Invalid voting method: {voting}. Must be 'soft' or 'hard'")


def adjust_confidence(
    probabilities: np.ndarray,
    regimes: Union[pd.Series, np.ndarray],
    regime_performance: Optional[Dict[str, float]] = None,
    adjustment_strength: float = 0.5,
) -> np.ndarray:
    """
    Adjust prediction confidence based on regime-specific model performance.

    Increases confidence in regimes where the model performs well,
    decreases confidence in regimes with poor performance.

    Parameters
    ----------
    probabilities : np.ndarray
        Raw prediction probabilities
    regimes : pd.Series or np.ndarray
        Regime labels for each sample
    regime_performance : Dict[str, float], optional
        Dictionary mapping regime names to performance metrics (e.g., accuracy)
        Values should be in [0, 1] range
        If None, no adjustment is performed
    adjustment_strength : float, default=0.5
        Strength of confidence adjustment (0 = no adjustment, 1 = full adjustment)
        Controls how much to scale probabilities based on regime performance

    Returns
    -------
    np.ndarray
        Adjusted prediction probabilities

    Examples
    --------
    >>> regime_performance = {'Calm': 0.75, 'Volatile': 0.55, 'Trending': 0.68}
    >>> adjusted_proba = adjust_confidence(
    ...     raw_proba, regimes_test, regime_performance, adjustment_strength=0.5
    ... )
    """
    # Convert regimes to array if needed
    if isinstance(regimes, pd.Series):
        regimes_array = regimes.values
    else:
        regimes_array = regimes

    # If no performance metrics provided, return original probabilities
    if regime_performance is None:
        return probabilities.copy()

    # Initialize adjusted probabilities
    adjusted_proba = probabilities.copy()

    # Adjust for each regime
    for regime, performance in regime_performance.items():
        regime_mask = regimes_array == regime

        if regime_mask.sum() == 0:
            continue

        # Calculate adjustment factor
        # Performance > 0.5: increase confidence
        # Performance < 0.5: decrease confidence
        adjustment_factor = 1.0 + adjustment_strength * (performance - 0.5) * 2

        # Adjust probabilities
        regime_proba = adjusted_proba[regime_mask]

        # Scale distance from 0.5 by adjustment factor
        adjusted_proba[regime_mask] = 0.5 + (regime_proba - 0.5) * adjustment_factor

    # Clip to valid probability range
    adjusted_proba = np.clip(adjusted_proba, 0.0, 1.0)

    return adjusted_proba


def select_model_by_regime(
    regime: str,
    regime_model_mapping: Dict[str, str],
    models: Dict[str, BaseEstimator],
    default_model_name: Optional[str] = None,
) -> BaseEstimator:
    """
    Select the best model for a given regime.

    Useful for regime-aware model selection strategies where different
    model types perform better in different regimes.

    Parameters
    ----------
    regime : str
        Regime label
    regime_model_mapping : Dict[str, str]
        Dictionary mapping regime names to model names
        Example: {'Calm': 'RF', 'Volatile': 'GB', 'Trending': 'RF'}
    models : Dict[str, BaseEstimator]
        Dictionary mapping model names to trained models
    default_model_name : str, optional
        Name of default model to use if regime not in mapping

    Returns
    -------
    BaseEstimator
        Selected model for the regime

    Raises
    ------
    ValueError
        If regime not in mapping and no default_model_name provided

    Examples
    --------
    >>> regime_mapping = {'Calm': 'RF', 'Volatile': 'GB'}
    >>> models = {'RF': rf_model, 'GB': gb_model}
    >>> model = select_model_by_regime('Calm', regime_mapping, models)
    """
    # Get model name for this regime
    if regime in regime_model_mapping:
        model_name = regime_model_mapping[regime]
    elif default_model_name is not None:
        model_name = default_model_name
        warnings.warn(
            f"Regime '{regime}' not in mapping, using default model '{default_model_name}'"
        )
    else:
        raise ValueError(
            f"Regime '{regime}' not in mapping and no default_model_name provided"
        )

    # Get the model
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found in models dictionary")

    return models[model_name]


def predict_with_regime_strategy(
    X: Union[pd.DataFrame, np.ndarray],
    regimes: Union[pd.Series, np.ndarray],
    regime_model_mapping: Dict[str, str],
    models: Dict[str, BaseEstimator],
    default_model_name: Optional[str] = None,
    return_proba: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Make predictions using regime-aware model selection strategy.

    Instead of training separate models per regime, this selects the best
    existing model type for each regime.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    regimes : pd.Series or np.ndarray
        Regime labels for each sample
    regime_model_mapping : Dict[str, str]
        Dictionary mapping regime names to model names
        Example: {'Calm': 'RF', 'Volatile': 'GB', 'Trending': 'RF'}
    models : Dict[str, BaseEstimator]
        Dictionary mapping model names to trained models
    default_model_name : str, optional
        Name of default model for unmapped regimes
    return_proba : bool, default=True
        Whether to return prediction probabilities

    Returns
    -------
    predictions : np.ndarray
        Predicted class labels
    probabilities : np.ndarray, optional
        Predicted probabilities (returned if return_proba=True)

    Examples
    --------
    >>> # Strategy: Use RF for Calm, GB for Volatile
    >>> mapping = {'Calm': 'RF', 'Volatile': 'GB', 'Trending': 'RF'}
    >>> models = {'RF': rf_model, 'GB': gb_model}
    >>> predictions, probabilities = predict_with_regime_strategy(
    ...     X_test, regimes_test, mapping, models, default_model_name='RF'
    ... )
    """
    # Convert inputs
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X

    if isinstance(regimes, pd.Series):
        regimes_array = regimes.values
    else:
        regimes_array = regimes

    # Initialize outputs
    n_samples = len(X_array)
    predictions = np.zeros(n_samples, dtype=int)
    probabilities = np.zeros(n_samples, dtype=float)

    # Get unique regimes
    unique_regimes = np.unique(regimes_array)

    # Make predictions for each regime using selected model
    for regime in unique_regimes:
        regime_mask = regimes_array == regime
        X_regime = X_array[regime_mask]

        if len(X_regime) == 0:
            continue

        # Select model for this regime
        model = select_model_by_regime(
            regime, regime_model_mapping, models, default_model_name
        )

        # Make predictions
        predictions[regime_mask] = model.predict(X_regime)

        if return_proba:
            proba = model.predict_proba(X_regime)
            if proba.ndim == 2 and proba.shape[1] == 2:
                probabilities[regime_mask] = proba[:, 1]
            else:
                probabilities[regime_mask] = proba

    if return_proba:
        return predictions, probabilities
    else:
        return predictions


def compute_regime_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    regimes: Union[pd.Series, np.ndarray],
    metric: str = "accuracy",
) -> Dict[str, float]:
    """
    Compute model performance for each regime.

    Useful for understanding which regimes the model performs well/poorly in.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    regimes : pd.Series or np.ndarray
        Regime labels
    metric : str, default='accuracy'
        Performance metric to compute
        Options: 'accuracy', 'precision', 'recall', 'f1'

    Returns
    -------
    Dict[str, float]
        Dictionary mapping regime names to performance scores

    Examples
    --------
    >>> regime_perf = compute_regime_performance(
    ...     y_test, y_pred, regimes_test, metric='accuracy'
    ... )
    >>> print(regime_perf)
    {'Calm': 0.75, 'Volatile': 0.55, 'Trending': 0.68}
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Convert regimes to array
    if isinstance(regimes, pd.Series):
        regimes_array = regimes.values
    else:
        regimes_array = regimes

    # Get unique regimes
    unique_regimes = np.unique(regimes_array)

    # Compute performance for each regime
    regime_performance = {}

    for regime in unique_regimes:
        regime_mask = regimes_array == regime
        y_true_regime = y_true[regime_mask]
        y_pred_regime = y_pred[regime_mask]

        if len(y_true_regime) == 0:
            continue

        # Compute metric
        if metric == "accuracy":
            score = accuracy_score(y_true_regime, y_pred_regime)
        elif metric == "precision":
            score = precision_score(y_true_regime, y_pred_regime, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true_regime, y_pred_regime, zero_division=0)
        elif metric == "f1":
            score = f1_score(y_true_regime, y_pred_regime, zero_division=0)
        else:
            raise ValueError(
                f"Invalid metric: {metric}. Must be 'accuracy', 'precision', 'recall', or 'f1'"
            )

        regime_performance[regime] = float(score)

    return regime_performance


def filter_by_confidence(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    confidence_threshold: float = 0.6,
    return_mask: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Filter predictions by confidence threshold.

    Only keeps predictions with confidence above threshold.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted class labels
    probabilities : np.ndarray
        Predicted probabilities
    confidence_threshold : float, default=0.6
        Minimum confidence (distance from 0.5 scaled to [0, 1])
    return_mask : bool, default=False
        Whether to return the filter mask

    Returns
    -------
    filtered_predictions : np.ndarray
        Predictions above confidence threshold
    filtered_probabilities : np.ndarray
        Probabilities above confidence threshold
    mask : np.ndarray, optional
        Boolean mask of kept predictions (returned if return_mask=True)

    Examples
    --------
    >>> # Only keep high-confidence predictions
    >>> filtered_pred, filtered_proba = filter_by_confidence(
    ...     predictions, probabilities, confidence_threshold=0.6
    ... )
    >>> print(f"Kept {len(filtered_pred)} out of {len(predictions)} predictions")
    """
    # Calculate confidence (distance from 0.5, scaled to [0, 1])
    confidence = np.abs(probabilities - 0.5) * 2

    # Create filter mask
    mask = confidence >= confidence_threshold

    # Filter
    filtered_predictions = predictions[mask]
    filtered_probabilities = probabilities[mask]

    if return_mask:
        return filtered_predictions, filtered_probabilities, mask
    else:
        return filtered_predictions, filtered_probabilities


def get_regime_statistics(
    regimes: Union[pd.Series, np.ndarray]
) -> pd.DataFrame:
    """
    Get statistics about regime distribution.

    Parameters
    ----------
    regimes : pd.Series or np.ndarray
        Regime labels

    Returns
    -------
    pd.DataFrame
        DataFrame with regime statistics:
        - regime: regime name
        - count: number of samples
        - percentage: percentage of total samples

    Examples
    --------
    >>> regime_stats = get_regime_statistics(regimes_test)
    >>> print(regime_stats)
    """
    # Convert to series if needed
    if isinstance(regimes, np.ndarray):
        regimes_series = pd.Series(regimes)
    else:
        regimes_series = regimes

    # Count regimes
    regime_counts = regimes_series.value_counts()

    # Create DataFrame
    stats_df = pd.DataFrame({
        "regime": regime_counts.index,
        "count": regime_counts.values,
        "percentage": 100 * regime_counts.values / len(regimes_series),
    })

    return stats_df.reset_index(drop=True)
