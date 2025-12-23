"""
Model Comparison Module

This module provides functions for comparing multiple models:
- Comprehensive performance metrics
- ROC curve comparison and visualization
- Statistical significance testing (bootstrap)
- Cross-model evaluation and ranking

Extracted from notebook 80_model_comparison.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    cohen_kappa_score,
    log_loss,
)
import warnings


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : np.ndarray, optional
        Predicted probabilities for class 1

    Returns
    -------
    Dict[str, float]
        Dictionary with metrics:
        - accuracy: Classification accuracy
        - precision: Precision score
        - recall: Recall score
        - f1: F1 score
        - kappa: Cohen's Kappa
        - auc: ROC-AUC (if y_proba provided)
        - logloss: Log loss (if y_proba provided)

    Examples
    --------
    >>> metrics = compute_metrics(y_test, y_pred, y_proba)
    >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    >>> print(f"F1: {metrics['f1']:.3f}")
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }

    if y_proba is not None:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_proba))
            metrics["logloss"] = float(log_loss(y_true, y_proba))
        except ValueError as e:
            warnings.warn(f"Could not compute AUC/logloss: {e}")
            metrics["auc"] = 0.0
            metrics["logloss"] = 0.0

    return metrics


def evaluate_model(
    model: BaseEstimator,
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation on train and test sets.

    Parameters
    ----------
    model : BaseEstimator
        Trained sklearn model
    X_train : pd.DataFrame or np.ndarray
        Training features
    X_test : pd.DataFrame or np.ndarray
        Test features
    y_train : np.ndarray
        Training labels
    y_test : np.ndarray
        Test labels
    model_name : str
        Name of the model

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - model: model name
        - train_*: training metrics
        - test_*: test metrics
        - accuracy_gap: train - test accuracy (overfitting indicator)
        - f1_gap: train - test F1 gap
        - y_test_pred: test predictions
        - y_test_proba: test probabilities

    Examples
    --------
    >>> results = evaluate_model(
    ...     rf_model, X_train, X_test, y_train, y_test, "Random Forest"
    ... )
    >>> print(f"Test Accuracy: {results['test_accuracy']:.3f}")
    >>> print(f"Overfitting Gap: {results['accuracy_gap']:.3f}")
    """
    # Get predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]

    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Compute train metrics
    train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba)

    # Compute test metrics
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)

    # Combine results
    results = {
        "model": model_name,
        # Train metrics
        "train_accuracy": train_metrics["accuracy"],
        "train_f1": train_metrics["f1"],
        "train_auc": train_metrics["auc"],
        # Test metrics
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_auc": test_metrics["auc"],
        "test_kappa": test_metrics["kappa"],
        "test_logloss": test_metrics["logloss"],
        # Overfitting indicators
        "accuracy_gap": train_metrics["accuracy"] - test_metrics["accuracy"],
        "f1_gap": train_metrics["f1"] - test_metrics["f1"],
        # Predictions
        "y_test_pred": y_test_pred,
        "y_test_proba": y_test_proba,
    }

    return results


def compare_models(
    models: Dict[str, BaseEstimator],
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
    rank_by: str = "test_accuracy",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Compare multiple models on same dataset.

    Parameters
    ----------
    models : Dict[str, BaseEstimator]
        Dictionary mapping model names to trained models
    X_train : pd.DataFrame or np.ndarray
        Training features
    X_test : pd.DataFrame or np.ndarray
        Test features
    y_train : np.ndarray
        Training labels
    y_test : np.ndarray
        Test labels
    rank_by : str, default='test_accuracy'
        Metric to rank models by

    Returns
    -------
    comparison_df : pd.DataFrame
        Comparison table with all metrics
    model_results : Dict[str, Dict[str, Any]]
        Detailed results for each model

    Examples
    --------
    >>> models = {'RF': rf_model, 'GB': gb_model, 'LR': lr_model}
    >>> comparison_df, results = compare_models(
    ...     models, X_train, X_test, y_train, y_test
    ... )
    >>> print(comparison_df)
    """
    # Evaluate each model
    model_results = {}

    for model_name, model in models.items():
        model_results[model_name] = evaluate_model(
            model, X_train, X_test, y_train, y_test, model_name
        )

    # Create comparison DataFrame
    comparison_data = []
    for model_name, results in model_results.items():
        comparison_data.append({
            "model": model_name,
            "train_accuracy": results["train_accuracy"],
            "test_accuracy": results["test_accuracy"],
            "test_precision": results["test_precision"],
            "test_recall": results["test_recall"],
            "test_f1": results["test_f1"],
            "test_auc": results["test_auc"],
            "test_kappa": results["test_kappa"],
            "test_logloss": results["test_logloss"],
            "accuracy_gap": results["accuracy_gap"],
            "f1_gap": results["f1_gap"],
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Sort by ranking metric
    if rank_by in comparison_df.columns:
        comparison_df = comparison_df.sort_values(rank_by, ascending=False)

    comparison_df = comparison_df.reset_index(drop=True)

    return comparison_df, model_results


def bootstrap_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    metric: str = "accuracy",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Bootstrap test for statistical significance between two models.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred1 : np.ndarray
        Predictions from model 1
    y_pred2 : np.ndarray
        Predictions from model 2
    metric : str, default='accuracy'
        Metric to compare ('accuracy', 'f1', 'precision', 'recall')
    n_bootstrap : int, default=1000
        Number of bootstrap iterations
    alpha : float, default=0.05
        Significance level for confidence interval

    Returns
    -------
    Dict[str, float]
        Test results:
        - mean_diff: Mean metric difference (model2 - model1)
        - ci_lower: Lower bound of CI
        - ci_upper: Upper bound of CI
        - p_value: P-value for one-sided test (model2 > model1)
        - significant: Whether difference is significant at alpha

    Examples
    --------
    >>> test_results = bootstrap_test(
    ...     y_test, y_pred_rf, y_pred_gb, metric='accuracy'
    ... )
    >>> print(f"P-value: {test_results['p_value']:.4f}")
    >>> print(f"Significant: {test_results['significant']}")
    """
    if len(y_true) != len(y_pred1) or len(y_true) != len(y_pred2):
        raise ValueError("All arrays must have the same length")

    # Select metric function
    if metric == "accuracy":
        metric_func = accuracy_score
    elif metric == "f1":
        metric_func = lambda yt, yp: f1_score(yt, yp, zero_division=0)
    elif metric == "precision":
        metric_func = lambda yt, yp: precision_score(yt, yp, zero_division=0)
    elif metric == "recall":
        metric_func = lambda yt, yp: recall_score(yt, yp, zero_division=0)
    else:
        raise ValueError(f"Invalid metric: {metric}")

    n_samples = len(y_true)
    diffs = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)

        # Compute metrics
        metric1 = metric_func(y_true[indices], y_pred1[indices])
        metric2 = metric_func(y_true[indices], y_pred2[indices])

        diffs.append(metric2 - metric1)

    diffs = np.array(diffs)

    # Calculate statistics
    mean_diff = np.mean(diffs)
    ci_lower = np.percentile(diffs, 100 * alpha / 2)
    ci_upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    p_value = np.mean(diffs <= 0)  # One-sided: model2 > model1

    return {
        "mean_diff": float(mean_diff),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
    }


def plot_roc_comparison(
    model_results: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.

    Parameters
    ----------
    model_results : Dict[str, Dict[str, Any]]
        Results from compare_models() or evaluate_model()
    y_test : np.ndarray
        True test labels
    show : bool, default=True
        Whether to display the plot
    save_path : str, optional
        Path to save the figure
    title : str, optional
        Custom title for the plot
    figsize : Tuple[int, int], default=(10, 8)
        Figure size

    Returns
    -------
    plt.Figure
        The matplotlib figure object

    Examples
    --------
    >>> fig = plot_roc_comparison(
    ...     model_results, y_test, save_path="roc_curves.png"
    ... )
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Default colors
    colors = plt.cm.Set1(np.linspace(0, 1, len(model_results)))

    for (model_name, results), color in zip(model_results.items(), colors):
        # Compute ROC curve
        y_proba = results["y_test_proba"]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = results.get("test_auc", roc_auc_score(y_test, y_proba))

        # Plot
        ax.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC = {auc:.3f})",
            linewidth=2.5,
            color=color,
            alpha=0.8,
        )

    # Plot diagonal (random classifier)
    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        linewidth=2,
        label="Random (AUC = 0.500)",
        alpha=0.5,
    )

    # Formatting
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.set_title("ROC Curves - Model Comparison", fontsize=12, fontweight="bold")

    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved ROC curves to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_confusion_matrices(
    model_results: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    normalize: bool = True,
    n_cols: int = 2,
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Plot confusion matrices for multiple models in a grid.

    Parameters
    ----------
    model_results : Dict[str, Dict[str, Any]]
        Results from compare_models()
    y_test : np.ndarray
        True test labels
    normalize : bool, default=True
        Whether to normalize confusion matrix
    n_cols : int, default=2
        Number of columns in grid
    show : bool, default=True
        Whether to display the plot
    save_path : str, optional
        Path to save the figure
    title : str, optional
        Overall title for the figure
    figsize : Tuple[int, int], optional
        Figure size (auto-calculated if None)

    Returns
    -------
    plt.Figure
        The matplotlib figure object

    Examples
    --------
    >>> fig = plot_confusion_matrices(
    ...     model_results, y_test, save_path="confusion_matrices.png"
    ... )
    """
    import seaborn as sns

    n_models = len(model_results)
    n_rows = int(np.ceil(n_models / n_cols))

    if figsize is None:
        figsize = (7 * n_cols, 6 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for idx, (model_name, results) in enumerate(model_results.items()):
        y_pred = results["y_test_pred"]
        cm = confusion_matrix(y_test, y_pred)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2%"
        else:
            fmt = "d"

        # Plot
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            cbar=True,
            ax=axes[idx],
            square=True,
            xticklabels=["Down", "Up"],
            yticklabels=["Down", "Up"],
        )

        axes[idx].set_xlabel("Predicted", fontsize=10)
        axes[idx].set_ylabel("Actual", fontsize=10)
        axes[idx].set_title(
            f'{model_name}\n(Accuracy: {results["test_accuracy"]:.3f})',
            fontsize=11,
            fontweight="bold",
        )

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    # Add overall title
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    else:
        fig.suptitle(
            "Confusion Matrices - Model Comparison",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved confusion matrices to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_metric_comparison(
    comparison_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    n_cols: int = 3,
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Plot comparison of multiple metrics across models.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison DataFrame from compare_models()
    metrics : List[str], optional
        List of metrics to plot (if None, plots common metrics)
    n_cols : int, default=3
        Number of columns in grid
    show : bool, default=True
        Whether to display the plot
    save_path : str, optional
        Path to save the figure
    title : str, optional
        Overall title
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        The matplotlib figure object

    Examples
    --------
    >>> fig = plot_metric_comparison(
    ...     comparison_df, metrics=['test_accuracy', 'test_f1', 'test_auc']
    ... )
    """
    # Default metrics
    if metrics is None:
        metrics = [
            "test_accuracy",
            "test_f1",
            "test_auc",
            "test_precision",
            "test_recall",
            "test_kappa",
        ]

    # Filter to available metrics
    available_metrics = [m for m in metrics if m in comparison_df.columns]

    n_metrics = len(available_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))

    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    # Color maps for different metrics
    cmaps = {
        "test_accuracy": "Blues",
        "test_f1": "Greens",
        "test_auc": "Purples",
        "test_precision": "Oranges",
        "test_recall": "Reds",
        "test_kappa": "YlOrRd",
    }

    for idx, metric in enumerate(available_metrics):
        values = comparison_df[metric]
        models = comparison_df["model"]

        # Choose colormap
        cmap = cmaps.get(metric, "viridis")
        colors = plt.cm.get_cmap(cmap)(np.linspace(0.4, 0.8, len(values)))

        # Create bar plot
        bars = axes[idx].bar(
            range(len(values)),
            values,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        )

        axes[idx].set_xticks(range(len(values)))
        axes[idx].set_xticklabels(models, rotation=45, ha="right", fontsize=9)
        axes[idx].set_ylabel(metric.replace("_", " ").title(), fontsize=10)
        axes[idx].set_title(
            metric.replace("_", " ").title(), fontsize=11, fontweight="bold"
        )
        axes[idx].grid(axis="y", alpha=0.3)

        # Annotate bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    # Add overall title
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.00)
    else:
        fig.suptitle(
            "Model Performance Comparison",
            fontsize=14,
            fontweight="bold",
            y=1.00,
        )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved metric comparison to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def find_best_model(
    comparison_df: pd.DataFrame,
    metric: str = "test_accuracy",
) -> Tuple[str, float]:
    """
    Find the best performing model by a given metric.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison DataFrame from compare_models()
    metric : str, default='test_accuracy'
        Metric to optimize

    Returns
    -------
    best_model : str
        Name of best model
    best_score : float
        Score of best model

    Examples
    --------
    >>> best_model, best_score = find_best_model(comparison_df, 'test_f1')
    >>> print(f"Best model: {best_model} ({best_score:.3f})")
    """
    if metric not in comparison_df.columns:
        raise ValueError(f"Metric '{metric}' not found in comparison DataFrame")

    idx = comparison_df[metric].idxmax()
    best_model = comparison_df.loc[idx, "model"]
    best_score = comparison_df.loc[idx, metric]

    return best_model, float(best_score)
