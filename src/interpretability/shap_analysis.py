"""
SHAP Analysis Module

This module provides functions for computing and visualizing SHAP (SHapley Additive exPlanations)
values for model interpretability:
- SHAP value computation (TreeExplainer for tree-based models)
- Summary plots (beeswarm) showing global importance and directionality
- Dependence plots showing feature effects and interactions
- Waterfall plots for individual prediction explanations
- Interaction values for feature synergies

Extracted from notebooks 60_feature_importance.ipynb and 65_model_interpretability.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Union, Any
from sklearn.base import BaseEstimator
import warnings

try:
    import shap
except ImportError:
    raise ImportError(
        "SHAP package is required for this module. Install with: pip install shap"
    )


def compute_shap_values(
    model: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[List[str]] = None,
    class_index: int = 1,
    max_samples: Optional[int] = None,
    check_additivity: bool = False,
) -> Tuple[np.ndarray, shap.Explainer]:
    """
    Compute SHAP values for a trained model.

    Uses TreeExplainer for tree-based models (fast) and can handle other model types
    with KernelExplainer (slower).

    Parameters
    ----------
    model : BaseEstimator
        Trained sklearn model
    X : pd.DataFrame or np.ndarray
        Feature matrix (typically test set)
    feature_names : List[str], optional
        List of feature names (required if X is np.ndarray)
    class_index : int, default=1
        For binary classification, which class to explain (0 or 1)
    max_samples : int, optional
        Maximum number of samples to compute SHAP values for (for speed)
        If None, uses all samples
    check_additivity : bool, default=False
        Whether to check additivity of SHAP values (slow, for debugging)

    Returns
    -------
    shap_values : np.ndarray
        SHAP values of shape (n_samples, n_features)
    explainer : shap.Explainer
        The SHAP explainer object (can be reused)

    Examples
    --------
    >>> shap_values, explainer = compute_shap_values(
    ...     rf_model, X_test, feature_names, class_index=1
    ... )
    >>> print(f"SHAP values shape: {shap_values.shape}")
    """
    # Convert to DataFrame if needed
    if isinstance(X, np.ndarray):
        if feature_names is None:
            raise ValueError("feature_names must be provided when X is np.ndarray")
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()
        if feature_names is None:
            feature_names = X_df.columns.tolist()

    # Limit samples if requested
    if max_samples is not None and len(X_df) > max_samples:
        X_df = X_df.iloc[:max_samples]

    # Try TreeExplainer first (for tree-based models)
    try:
        explainer = shap.TreeExplainer(model, check_additivity=check_additivity)
        shap_values_raw = explainer.shap_values(X_df, check_additivity=check_additivity)
    except Exception as e:
        # Fall back to KernelExplainer for other model types
        warnings.warn(
            f"TreeExplainer failed ({str(e)}), falling back to KernelExplainer. "
            "This will be slower."
        )
        # Sample background data for KernelExplainer (use 100 samples)
        background_size = min(100, len(X_df))
        background = shap.sample(X_df, background_size)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values_raw = explainer.shap_values(X_df)

    # Handle different SHAP output formats
    if isinstance(shap_values_raw, list):
        # Binary classification: list of [class_0, class_1]
        shap_values = shap_values_raw[class_index]
    else:
        shap_values = shap_values_raw

    # Handle 3D output (n_samples, n_features, n_classes)
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, class_index]

    # Validate shape
    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP values shape mismatch: got {shap_values.shape[1]} features, "
            f"expected {len(feature_names)}"
        )

    return shap_values, explainer


def compute_shap_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Compute global feature importance from SHAP values.

    Importance is computed as the mean absolute SHAP value for each feature.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values of shape (n_samples, n_features)
    feature_names : List[str]
        List of feature names

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature', 'importance'] sorted by importance (descending)

    Examples
    --------
    >>> importance_df = compute_shap_importance(shap_values, feature_names)
    >>> print(importance_df.head(20))
    """
    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            f"Shape mismatch: SHAP values have {shap_values.shape[1]} features, "
            f"but {len(feature_names)} feature names provided"
        )

    # Compute mean absolute SHAP value for each feature
    importance = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    })

    # Sort by importance (descending)
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df


def plot_shap_summary(
    shap_values: np.ndarray,
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[List[str]] = None,
    max_display: int = 20,
    plot_type: str = "dot",
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Create SHAP summary plot (beeswarm plot).

    Shows:
    - Feature importance (vertical axis ordering)
    - Effect direction (horizontal axis: positive = increases prediction)
    - Feature value (color: red = high value, blue = low value)

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values of shape (n_samples, n_features)
    X : pd.DataFrame or np.ndarray
        Feature matrix (same samples as shap_values)
    feature_names : List[str], optional
        List of feature names (required if X is np.ndarray)
    max_display : int, default=20
        Maximum number of features to display
    plot_type : str, default='dot'
        Type of plot ('dot', 'bar', 'violin')
    show : bool, default=True
        Whether to display the plot
    save_path : str, optional
        Path to save the figure
    title : str, optional
        Custom title for the plot
    figsize : Tuple[int, int], default=(12, 8)
        Figure size

    Returns
    -------
    plt.Figure
        The matplotlib figure object

    Examples
    --------
    >>> fig = plot_shap_summary(
    ...     shap_values, X_test, feature_names,
    ...     max_display=20, save_path="shap_summary.png"
    ... )
    """
    # Convert to DataFrame if needed
    if isinstance(X, np.ndarray):
        if feature_names is None:
            raise ValueError("feature_names must be provided when X is np.ndarray")
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()
        if feature_names is None:
            feature_names = X_df.columns.tolist()

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Create SHAP summary plot
    shap.summary_plot(
        shap_values,
        X_df,
        feature_names=feature_names,
        plot_type=plot_type,
        max_display=max_display,
        show=False,
    )

    # Add custom title if provided
    if title:
        plt.title(title, fontsize=14, fontweight="bold", pad=20)
    else:
        plt.title(
            "SHAP Summary Plot\n(Feature Importance & Effect Direction)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved SHAP summary plot to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_shap_dependence(
    shap_values: np.ndarray,
    X: Union[pd.DataFrame, np.ndarray],
    feature: Union[str, int],
    feature_names: Optional[List[str]] = None,
    interaction_index: Union[str, int, None] = "auto",
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Create SHAP dependence plot for a specific feature.

    Shows how a feature's value affects predictions:
    - X-axis: feature value
    - Y-axis: SHAP value (impact on prediction)
    - Color: interaction feature (automatically selected or specified)

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values of shape (n_samples, n_features)
    X : pd.DataFrame or np.ndarray
        Feature matrix
    feature : str or int
        Feature name or index to plot
    feature_names : List[str], optional
        List of feature names (required if X is np.ndarray)
    interaction_index : str, int, or None, default='auto'
        Feature to use for coloring points (auto-detects interaction)
    show : bool, default=True
        Whether to display the plot
    save_path : str, optional
        Path to save the figure
    title : str, optional
        Custom title for the plot
    ax : plt.Axes, optional
        Matplotlib axes to plot on

    Returns
    -------
    plt.Figure
        The matplotlib figure object

    Examples
    --------
    >>> fig = plot_shap_dependence(
    ...     shap_values, X_test, "impact_permanent_impact_5_mean",
    ...     feature_names, save_path="shap_dependence.png"
    ... )
    """
    # Convert to DataFrame if needed
    if isinstance(X, np.ndarray):
        if feature_names is None:
            raise ValueError("feature_names must be provided when X is np.ndarray")
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()
        if feature_names is None:
            feature_names = X_df.columns.tolist()

    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Create SHAP dependence plot
    shap.dependence_plot(
        feature,
        shap_values,
        X_df,
        feature_names=feature_names,
        interaction_index=interaction_index,
        show=False,
        ax=ax,
    )

    # Add custom title if provided
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        feature_name = feature if isinstance(feature, str) else feature_names[feature]
        ax.set_title(
            f"SHAP Dependence Plot: {feature_name}",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved SHAP dependence plot to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_shap_dependence_multiple(
    shap_values: np.ndarray,
    X: Union[pd.DataFrame, np.ndarray],
    features: List[Union[str, int]],
    feature_names: Optional[List[str]] = None,
    n_cols: int = 3,
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Create SHAP dependence plots for multiple features in a grid.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values of shape (n_samples, n_features)
    X : pd.DataFrame or np.ndarray
        Feature matrix
    features : List[str or int]
        List of feature names or indices to plot
    feature_names : List[str], optional
        List of feature names (required if X is np.ndarray)
    n_cols : int, default=3
        Number of columns in the grid
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
    >>> top_features = ["feature1", "feature2", "feature3", "feature4"]
    >>> fig = plot_shap_dependence_multiple(
    ...     shap_values, X_test, top_features, feature_names
    ... )
    """
    # Convert to DataFrame if needed
    if isinstance(X, np.ndarray):
        if feature_names is None:
            raise ValueError("feature_names must be provided when X is np.ndarray")
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()
        if feature_names is None:
            feature_names = X_df.columns.tolist()

    # Calculate grid dimensions
    n_features = len(features)
    n_rows = int(np.ceil(n_features / n_cols))

    # Auto-calculate figsize if not provided
    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()  # Flatten to 1D array

    # Plot each feature
    for idx, feature in enumerate(features):
        shap.dependence_plot(
            feature,
            shap_values,
            X_df,
            feature_names=feature_names,
            show=False,
            ax=axes[idx],
        )
        feature_name = feature if isinstance(feature, str) else feature_names[feature]
        axes[idx].set_title(f"{feature_name}", fontsize=10, fontweight="bold")

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.00)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved SHAP dependence plots to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_shap_waterfall(
    shap_values: np.ndarray,
    X: Union[pd.DataFrame, np.ndarray],
    sample_index: int,
    feature_names: Optional[List[str]] = None,
    expected_value: Optional[float] = None,
    max_display: int = 15,
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Create SHAP waterfall plot for a single prediction.

    Shows how each feature contributes to moving the prediction from the
    base value (expected value) to the final prediction.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values of shape (n_samples, n_features)
    X : pd.DataFrame or np.ndarray
        Feature matrix
    sample_index : int
        Index of the sample to explain
    feature_names : List[str], optional
        List of feature names (required if X is np.ndarray)
    expected_value : float, optional
        Expected value (base value) for the model
        If None, uses 0 as default
    max_display : int, default=15
        Maximum number of features to display
    show : bool, default=True
        Whether to display the plot
    save_path : str, optional
        Path to save the figure
    title : str, optional
        Custom title for the plot

    Returns
    -------
    plt.Figure
        The matplotlib figure object

    Examples
    --------
    >>> fig = plot_shap_waterfall(
    ...     shap_values, X_test, sample_index=0,
    ...     feature_names=feature_names, expected_value=explainer.expected_value[1]
    ... )
    """
    # Convert to DataFrame if needed
    if isinstance(X, np.ndarray):
        if feature_names is None:
            raise ValueError("feature_names must be provided when X is np.ndarray")
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()
        if feature_names is None:
            feature_names = X_df.columns.tolist()

    # Validate sample index
    if sample_index >= len(shap_values):
        raise ValueError(
            f"sample_index {sample_index} is out of bounds for SHAP values "
            f"with length {len(shap_values)}"
        )

    # Use 0 as default expected value if not provided
    if expected_value is None:
        expected_value = 0.0
        warnings.warn(
            "expected_value not provided, using 0.0 as default. "
            "For better results, provide explainer.expected_value"
        )

    # Create SHAP Explanation object
    shap_explanation = shap.Explanation(
        values=shap_values[sample_index],
        base_values=expected_value,
        data=X_df.iloc[sample_index].values,
        feature_names=feature_names,
    )

    # Create figure
    fig = plt.figure(figsize=(10, 8))

    # Create waterfall plot
    shap.waterfall_plot(shap_explanation, max_display=max_display, show=False)

    # Add custom title if provided
    if title:
        plt.title(title, fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved SHAP waterfall plot to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_shap_waterfall_multiple(
    shap_values: np.ndarray,
    X: Union[pd.DataFrame, np.ndarray],
    sample_indices: List[int],
    feature_names: Optional[List[str]] = None,
    expected_value: Optional[float] = None,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    y_proba: Optional[np.ndarray] = None,
    max_display: int = 15,
    n_cols: int = 2,
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Create SHAP waterfall plots for multiple samples in a grid.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values of shape (n_samples, n_features)
    X : pd.DataFrame or np.ndarray
        Feature matrix
    sample_indices : List[int]
        List of sample indices to explain
    feature_names : List[str], optional
        List of feature names (required if X is np.ndarray)
    expected_value : float, optional
        Expected value (base value) for the model
    y_true : np.ndarray, optional
        True labels (for display in title)
    y_pred : np.ndarray, optional
        Predicted labels (for display in title)
    y_proba : np.ndarray, optional
        Predicted probabilities (for display in title)
    max_display : int, default=15
        Maximum number of features to display per plot
    n_cols : int, default=2
        Number of columns in the grid
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
    >>> # Explain 4 examples: correct up, correct down, false positive, false negative
    >>> fig = plot_shap_waterfall_multiple(
    ...     shap_values, X_test, [0, 10, 50, 100],
    ...     feature_names, expected_value, y_test, y_pred, y_proba
    ... )
    """
    # Convert to DataFrame if needed
    if isinstance(X, np.ndarray):
        if feature_names is None:
            raise ValueError("feature_names must be provided when X is np.ndarray")
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()
        if feature_names is None:
            feature_names = X_df.columns.tolist()

    # Use 0 as default expected value if not provided
    if expected_value is None:
        expected_value = 0.0
        warnings.warn("expected_value not provided, using 0.0 as default")

    # Calculate grid dimensions
    n_samples = len(sample_indices)
    n_rows = int(np.ceil(n_samples / n_cols))

    # Auto-calculate figsize if not provided
    if figsize is None:
        figsize = (8 * n_cols, 6 * n_rows)

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()  # Flatten to 1D array

    # Plot each sample
    for idx, sample_idx in enumerate(sample_indices):
        plt.sca(axes[idx])

        # Create SHAP Explanation object
        shap_explanation = shap.Explanation(
            values=shap_values[sample_idx],
            base_values=expected_value,
            data=X_df.iloc[sample_idx].values,
            feature_names=feature_names,
        )

        # Create waterfall plot
        shap.waterfall_plot(shap_explanation, max_display=max_display, show=False)

        # Create title with labels and predictions if provided
        subtitle_parts = [f"Sample {sample_idx}"]
        if y_true is not None and y_pred is not None:
            subtitle_parts.append(f"Actual={y_true[sample_idx]}, Pred={y_pred[sample_idx]}")
        if y_proba is not None:
            subtitle_parts.append(f"Prob={y_proba[sample_idx]:.3f}")

        axes[idx].set_title(
            ", ".join(subtitle_parts),
            fontsize=10,
            fontweight="bold",
        )

    # Hide unused subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].set_visible(False)

    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.00)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved SHAP waterfall plots to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def compute_shap_interaction_values(
    model: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[List[str]] = None,
    class_index: int = 1,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute SHAP interaction values to identify feature synergies.

    SHAP interaction values show how pairs of features work together to
    influence predictions.

    Parameters
    ----------
    model : BaseEstimator
        Trained sklearn tree-based model
    X : pd.DataFrame or np.ndarray
        Feature matrix
    feature_names : List[str], optional
        List of feature names (required if X is np.ndarray)
    class_index : int, default=1
        For binary classification, which class to explain (0 or 1)
    max_samples : int, optional
        Maximum number of samples (interaction computation is expensive)
        Recommended: 100-200 samples

    Returns
    -------
    interaction_values : np.ndarray
        SHAP interaction values of shape (n_samples, n_features, n_features)
    interaction_df : pd.DataFrame
        Top feature interactions sorted by strength

    Examples
    --------
    >>> interaction_values, top_interactions = compute_shap_interaction_values(
    ...     rf_model, X_test, feature_names, max_samples=100
    ... )
    >>> print(top_interactions.head(20))
    """
    # Convert to DataFrame if needed
    if isinstance(X, np.ndarray):
        if feature_names is None:
            raise ValueError("feature_names must be provided when X is np.ndarray")
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()
        if feature_names is None:
            feature_names = X_df.columns.tolist()

    # Limit samples (interaction computation is expensive)
    if max_samples is not None and len(X_df) > max_samples:
        X_df = X_df.iloc[:max_samples]
        warnings.warn(
            f"Computing SHAP interactions for {max_samples} samples "
            f"(out of {len(X)}). This may take several minutes."
        )

    # Create explainer and compute interaction values
    try:
        explainer = shap.TreeExplainer(model)
        interaction_values_raw = explainer.shap_interaction_values(X_df)
    except Exception as e:
        raise RuntimeError(
            f"Failed to compute SHAP interaction values: {str(e)}. "
            "This typically requires a tree-based model."
        )

    # Handle different output formats
    if isinstance(interaction_values_raw, list):
        interaction_values = interaction_values_raw[class_index]
    else:
        interaction_values = interaction_values_raw

    # Handle 4D output (n_samples, n_features, n_features, n_classes)
    if len(interaction_values.shape) == 4:
        interaction_values = interaction_values[:, :, :, class_index]

    # Compute mean absolute interaction strength for each feature pair
    n_features = len(feature_names)
    top_interactions = []

    for i in range(n_features):
        for j in range(i + 1, n_features):  # Only upper triangle
            interaction_strength = np.abs(interaction_values[:, i, j]).mean()
            top_interactions.append({
                "feature_1": feature_names[i],
                "feature_2": feature_names[j],
                "interaction_strength": float(interaction_strength),
            })

    # Create DataFrame and sort
    interaction_df = pd.DataFrame(top_interactions).sort_values(
        "interaction_strength", ascending=False
    ).reset_index(drop=True)

    return interaction_values, interaction_df


def plot_shap_interaction_heatmap(
    interaction_values: np.ndarray,
    feature_names: List[str],
    top_k: Optional[int] = 20,
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 12),
) -> plt.Figure:
    """
    Create heatmap of SHAP interaction values.

    Parameters
    ----------
    interaction_values : np.ndarray
        SHAP interaction values of shape (n_samples, n_features, n_features)
    feature_names : List[str]
        List of feature names
    top_k : int, optional
        If provided, only show top K features (by main effect)
    show : bool, default=True
        Whether to display the plot
    save_path : str, optional
        Path to save the figure
    title : str, optional
        Custom title for the plot
    figsize : Tuple[int, int], default=(14, 12)
        Figure size

    Returns
    -------
    plt.Figure
        The matplotlib figure object

    Examples
    --------
    >>> fig = plot_shap_interaction_heatmap(
    ...     interaction_values, feature_names, top_k=20,
    ...     save_path="shap_interaction_heatmap.png"
    ... )
    """
    import seaborn as sns

    # Compute mean absolute interaction matrix
    n_features = len(feature_names)
    interaction_matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(n_features):
            interaction_matrix[i, j] = np.abs(interaction_values[:, i, j]).mean()

    # Select top K features if requested
    if top_k is not None and top_k < n_features:
        # Compute main effects (diagonal)
        main_effects = np.diag(interaction_matrix)
        top_indices = np.argsort(main_effects)[-top_k:][::-1]

        # Subset matrix and feature names
        interaction_matrix = interaction_matrix[np.ix_(top_indices, top_indices)]
        selected_features = [feature_names[i] for i in top_indices]
    else:
        selected_features = feature_names

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        interaction_matrix,
        xticklabels=selected_features,
        yticklabels=selected_features,
        cmap="YlOrRd",
        cbar_kws={"label": "Interaction Strength"},
        square=True,
        ax=ax,
    )

    # Customize
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    else:
        ax.set_title(
            f"SHAP Feature Interaction Matrix\nTop {len(selected_features)} Features",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved SHAP interaction heatmap to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig
