"""
Partial Dependence Analysis Module

This module provides functions for computing and visualizing partial dependence plots (PDPs)
and Individual Conditional Expectation (ICE) curves:
- 1D partial dependence plots (marginal effect of single features)
- 2D partial dependence plots (joint effects of feature pairs)
- ICE curves (individual sample effects, showing heterogeneity)
- Combined PDP+ICE plots

Partial dependence shows the average effect of a feature on predictions,
while ICE curves show how individual samples respond to feature changes.

Extracted from notebook 65_model_interpretability.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Union, Any
from sklearn.base import BaseEstimator
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import warnings


def compute_pdp_1d(
    model: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    features: Union[List[str], List[int]],
    feature_names: Optional[List[str]] = None,
    grid_resolution: int = 100,
    percentiles: Tuple[float, float] = (0.05, 0.95),
    kind: str = "average",
) -> Dict[str, Any]:
    """
    Compute 1D partial dependence for specified features.

    Partial dependence shows the marginal effect of a feature on predictions,
    averaging over all other features.

    Parameters
    ----------
    model : BaseEstimator
        Trained sklearn model
    X : pd.DataFrame or np.ndarray
        Feature matrix
    features : List[str] or List[int]
        List of feature names or indices to compute PDP for
    feature_names : List[str], optional
        List of feature names (required if X is np.ndarray or features are indices)
    grid_resolution : int, default=100
        Number of points in the grid
    percentiles : Tuple[float, float], default=(0.05, 0.95)
        Lower and upper percentiles for grid range
    kind : str, default='average'
        Type of PDP: 'average' (mean), 'individual' (ICE), or 'both'

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'pdp_values': partial dependence values for each feature
        - 'grid_values': grid values for each feature
        - 'feature_names': names of features
        - 'kind': type of PDP computed

    Examples
    --------
    >>> pdp_results = compute_pdp_1d(
    ...     rf_model, X_test, ['feature1', 'feature2', 'feature3'],
    ...     grid_resolution=100
    ... )
    >>> for fname, pdp_vals in zip(pdp_results['feature_names'], pdp_results['pdp_values']):
    ...     print(f"{fname}: range [{pdp_vals.min():.3f}, {pdp_vals.max():.3f}]")
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

    # Convert feature names to indices if needed
    feature_indices = []
    for feature in features:
        if isinstance(feature, str):
            feature_indices.append(feature_names.index(feature))
        else:
            feature_indices.append(feature)

    # Compute partial dependence
    pd_result = partial_dependence(
        model,
        X_df,
        features=feature_indices,
        grid_resolution=grid_resolution,
        percentiles=percentiles,
        kind=kind,
    )

    # Extract results
    if kind == "average":
        pdp_values = pd_result["average"]
    elif kind == "individual":
        pdp_values = pd_result["individual"]
    else:  # both
        pdp_values = pd_result

    grid_values = pd_result["grid_values"]

    # Get feature names
    result_feature_names = [
        feature if isinstance(feature, str) else feature_names[feature]
        for feature in features
    ]

    return {
        "pdp_values": pdp_values,
        "grid_values": grid_values,
        "feature_names": result_feature_names,
        "kind": kind,
    }


def compute_pdp_2d(
    model: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    feature_pairs: List[Tuple[Union[str, int], Union[str, int]]],
    feature_names: Optional[List[str]] = None,
    grid_resolution: int = 50,
    percentiles: Tuple[float, float] = (0.05, 0.95),
) -> Dict[str, Any]:
    """
    Compute 2D partial dependence for feature pairs.

    Shows the joint effect of two features on predictions, revealing interactions.

    Parameters
    ----------
    model : BaseEstimator
        Trained sklearn model
    X : pd.DataFrame or np.ndarray
        Feature matrix
    feature_pairs : List[Tuple[str or int, str or int]]
        List of feature pairs (each pair is a tuple of feature names or indices)
    feature_names : List[str], optional
        List of feature names (required if X is np.ndarray)
    grid_resolution : int, default=50
        Number of points in each dimension of the grid
    percentiles : Tuple[float, float], default=(0.05, 0.95)
        Lower and upper percentiles for grid range

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'pdp_values': 2D partial dependence values for each pair
        - 'grid_values': grid values for each pair (list of [grid_x, grid_y])
        - 'feature_pairs': names of feature pairs

    Examples
    --------
    >>> pairs = [('feature1', 'feature2'), ('feature1', 'feature3')]
    >>> pdp_2d_results = compute_pdp_2d(
    ...     rf_model, X_test, pairs, grid_resolution=50
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

    # Convert feature pairs to indices
    feature_pair_indices = []
    for feat1, feat2 in feature_pairs:
        idx1 = feature_names.index(feat1) if isinstance(feat1, str) else feat1
        idx2 = feature_names.index(feat2) if isinstance(feat2, str) else feat2
        feature_pair_indices.append((idx1, idx2))

    # Compute 2D partial dependence
    pd_result = partial_dependence(
        model,
        X_df,
        features=feature_pair_indices,
        grid_resolution=grid_resolution,
        percentiles=percentiles,
        kind="average",
    )

    # Get feature pair names
    result_pairs = []
    for feat1, feat2 in feature_pairs:
        name1 = feat1 if isinstance(feat1, str) else feature_names[feat1]
        name2 = feat2 if isinstance(feat2, str) else feature_names[feat2]
        result_pairs.append((name1, name2))

    return {
        "pdp_values": pd_result["average"],
        "grid_values": pd_result["grid_values"],
        "feature_pairs": result_pairs,
    }


def plot_pdp_1d(
    model: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    features: Union[List[str], List[int]],
    feature_names: Optional[List[str]] = None,
    grid_resolution: int = 100,
    n_cols: int = 3,
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Plot 1D partial dependence for multiple features.

    Parameters
    ----------
    model : BaseEstimator
        Trained sklearn model
    X : pd.DataFrame or np.ndarray
        Feature matrix
    features : List[str] or List[int]
        List of feature names or indices
    feature_names : List[str], optional
        List of feature names (required if X is np.ndarray)
    grid_resolution : int, default=100
        Number of points in the grid
    n_cols : int, default=3
        Number of columns in the plot grid
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
    >>> fig = plot_pdp_1d(
    ...     rf_model, X_test, ['feature1', 'feature2', 'feature3'],
    ...     feature_names, save_path="pdp_1d.png"
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

    # Convert features to indices
    feature_indices = []
    for feature in features:
        if isinstance(feature, str):
            feature_indices.append(feature_names.index(feature))
        else:
            feature_indices.append(feature)

    # Calculate grid dimensions
    n_features = len(features)
    n_rows = int(np.ceil(n_features / n_cols))

    # Auto-calculate figsize if not provided
    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    # Create partial dependence display
    display = PartialDependenceDisplay.from_estimator(
        model,
        X_df,
        features=feature_indices,
        feature_names=feature_names,
        kind="average",
        grid_resolution=grid_resolution,
        ax=axes[:n_features],
        n_jobs=-1,
    )

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.00)
    else:
        fig.suptitle(
            "Partial Dependence Plots (1D)",
            fontsize=14,
            fontweight="bold",
            y=1.00,
        )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved 1D partial dependence plots to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_pdp_2d(
    model: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    feature_pairs: List[Tuple[Union[str, int], Union[str, int]]],
    feature_names: Optional[List[str]] = None,
    grid_resolution: int = 50,
    n_cols: int = 3,
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Plot 2D partial dependence for feature pairs.

    Parameters
    ----------
    model : BaseEstimator
        Trained sklearn model
    X : pd.DataFrame or np.ndarray
        Feature matrix
    feature_pairs : List[Tuple[str or int, str or int]]
        List of feature pairs to plot
    feature_names : List[str], optional
        List of feature names (required if X is np.ndarray)
    grid_resolution : int, default=50
        Number of points in each dimension
    n_cols : int, default=3
        Number of columns in the plot grid
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
    >>> pairs = [('feature1', 'feature2'), ('feature1', 'feature3')]
    >>> fig = plot_pdp_2d(
    ...     rf_model, X_test, pairs, feature_names,
    ...     save_path="pdp_2d.png"
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

    # Convert feature pairs to indices
    feature_pair_indices = []
    for feat1, feat2 in feature_pairs:
        idx1 = feature_names.index(feat1) if isinstance(feat1, str) else feat1
        idx2 = feature_names.index(feat2) if isinstance(feat2, str) else feat2
        feature_pair_indices.append((idx1, idx2))

    # Calculate grid dimensions
    n_pairs = len(feature_pairs)
    n_rows = int(np.ceil(n_pairs / n_cols))

    # Auto-calculate figsize if not provided
    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    # Create 2D partial dependence display
    display = PartialDependenceDisplay.from_estimator(
        model,
        X_df,
        features=feature_pair_indices,
        feature_names=feature_names,
        kind="average",
        grid_resolution=grid_resolution,
        ax=axes[:n_pairs],
        n_jobs=-1,
    )

    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.00)
    else:
        fig.suptitle(
            "2D Partial Dependence - Feature Interactions",
            fontsize=14,
            fontweight="bold",
            y=1.00,
        )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved 2D partial dependence plots to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_ice_curves(
    model: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    features: Union[List[str], List[int]],
    feature_names: Optional[List[str]] = None,
    n_ice_samples: Optional[int] = 50,
    grid_resolution: int = 100,
    centered: bool = False,
    n_cols: int = 3,
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Plot Individual Conditional Expectation (ICE) curves.

    ICE plots show how individual samples respond to feature changes,
    revealing heterogeneity in feature effects.

    Parameters
    ----------
    model : BaseEstimator
        Trained sklearn model
    X : pd.DataFrame or np.ndarray
        Feature matrix
    features : List[str] or List[int]
        List of feature names or indices
    feature_names : List[str], optional
        List of feature names (required if X is np.ndarray)
    n_ice_samples : int, optional, default=50
        Number of samples to plot ICE curves for
        If None, uses all samples (can be slow for large datasets)
    grid_resolution : int, default=100
        Number of points in the grid
    centered : bool, default=False
        Whether to center ICE curves (c-ICE)
    n_cols : int, default=3
        Number of columns in the plot grid
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
    >>> fig = plot_ice_curves(
    ...     rf_model, X_test, ['feature1', 'feature2', 'feature3'],
    ...     feature_names, n_ice_samples=50, save_path="ice_plots.png"
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

    # Sample data if needed
    if n_ice_samples is not None and len(X_df) > n_ice_samples:
        sample_indices = np.random.choice(len(X_df), size=n_ice_samples, replace=False)
        X_ice = X_df.iloc[sample_indices]
    else:
        X_ice = X_df

    # Convert features to indices
    feature_indices = []
    for feature in features:
        if isinstance(feature, str):
            feature_indices.append(feature_names.index(feature))
        else:
            feature_indices.append(feature)

    # Calculate grid dimensions
    n_features = len(features)
    n_rows = int(np.ceil(n_features / n_cols))

    # Auto-calculate figsize if not provided
    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    # Determine ICE kind
    ice_kind = "individual" if not centered else "both"

    # Create ICE display
    display = PartialDependenceDisplay.from_estimator(
        model,
        X_ice,
        features=feature_indices,
        feature_names=feature_names,
        kind=ice_kind,
        grid_resolution=grid_resolution,
        ax=axes[:n_features],
        n_jobs=-1,
        centered=centered,
    )

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.00)
    else:
        ice_type = "Centered ICE" if centered else "ICE"
        fig.suptitle(
            f"{ice_type} Curves (n={len(X_ice)} samples)",
            fontsize=14,
            fontweight="bold",
            y=1.00,
        )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved ICE curves to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_pdp_with_ice(
    model: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    features: Union[List[str], List[int]],
    feature_names: Optional[List[str]] = None,
    n_ice_samples: Optional[int] = 50,
    grid_resolution: int = 100,
    n_cols: int = 3,
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Plot partial dependence with ICE curves overlaid.

    Shows both the average effect (PDP) and individual effects (ICE) together.

    Parameters
    ----------
    model : BaseEstimator
        Trained sklearn model
    X : pd.DataFrame or np.ndarray
        Feature matrix
    features : List[str] or List[int]
        List of feature names or indices
    feature_names : List[str], optional
        List of feature names (required if X is np.ndarray)
    n_ice_samples : int, optional, default=50
        Number of samples to plot ICE curves for
    grid_resolution : int, default=100
        Number of points in the grid
    n_cols : int, default=3
        Number of columns in the plot grid
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
    >>> fig = plot_pdp_with_ice(
    ...     rf_model, X_test, ['feature1', 'feature2'],
    ...     feature_names, n_ice_samples=30
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

    # Sample data if needed
    if n_ice_samples is not None and len(X_df) > n_ice_samples:
        sample_indices = np.random.choice(len(X_df), size=n_ice_samples, replace=False)
        X_plot = X_df.iloc[sample_indices]
    else:
        X_plot = X_df

    # Convert features to indices
    feature_indices = []
    for feature in features:
        if isinstance(feature, str):
            feature_indices.append(feature_names.index(feature))
        else:
            feature_indices.append(feature)

    # Calculate grid dimensions
    n_features = len(features)
    n_rows = int(np.ceil(n_features / n_cols))

    # Auto-calculate figsize if not provided
    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    # Create combined PDP + ICE display
    display = PartialDependenceDisplay.from_estimator(
        model,
        X_plot,
        features=feature_indices,
        feature_names=feature_names,
        kind="both",  # Both PDP and ICE
        grid_resolution=grid_resolution,
        ax=axes[:n_features],
        n_jobs=-1,
    )

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.00)
    else:
        fig.suptitle(
            f"Partial Dependence with ICE Curves (n={len(X_plot)} samples)",
            fontsize=14,
            fontweight="bold",
            y=1.00,
        )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved PDP+ICE plots to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def analyze_feature_heterogeneity(
    model: BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray],
    features: Union[List[str], List[int]],
    feature_names: Optional[List[str]] = None,
    n_ice_samples: Optional[int] = 100,
    grid_resolution: int = 100,
) -> pd.DataFrame:
    """
    Analyze heterogeneity in feature effects using ICE curves.

    Computes statistics on ICE curve variance to identify features with
    heterogeneous effects (where different samples respond differently).

    Parameters
    ----------
    model : BaseEstimator
        Trained sklearn model
    X : pd.DataFrame or np.ndarray
        Feature matrix
    features : List[str] or List[int]
        List of feature names or indices
    feature_names : List[str], optional
        List of feature names (required if X is np.ndarray)
    n_ice_samples : int, optional, default=100
        Number of samples to compute ICE curves for
    grid_resolution : int, default=100
        Number of points in the grid

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature', 'ice_variance', 'ice_range']
        sorted by variance (descending)

    Examples
    --------
    >>> heterogeneity = analyze_feature_heterogeneity(
    ...     rf_model, X_test, feature_names, n_ice_samples=200
    ... )
    >>> print("Features with most heterogeneous effects:")
    >>> print(heterogeneity.head(10))
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

    # Sample data if needed
    if n_ice_samples is not None and len(X_df) > n_ice_samples:
        sample_indices = np.random.choice(len(X_df), size=n_ice_samples, replace=False)
        X_ice = X_df.iloc[sample_indices]
    else:
        X_ice = X_df

    # Analyze each feature
    results = []

    for feature in features:
        # Convert to index
        if isinstance(feature, str):
            feature_idx = feature_names.index(feature)
            feature_name = feature
        else:
            feature_idx = feature
            feature_name = feature_names[feature]

        # Compute ICE curves
        pd_result = partial_dependence(
            model,
            X_ice,
            features=[feature_idx],
            grid_resolution=grid_resolution,
            kind="individual",
        )

        # Get individual curves
        ice_curves = pd_result["individual"][0]  # Shape: (n_samples, grid_resolution)

        # Compute variance across samples at each grid point
        variance_per_point = np.var(ice_curves, axis=0)
        mean_variance = np.mean(variance_per_point)

        # Compute range of effects
        ice_range = ice_curves.max() - ice_curves.min()

        results.append({
            "feature": feature_name,
            "ice_variance": float(mean_variance),
            "ice_range": float(ice_range),
        })

    # Create DataFrame and sort
    heterogeneity_df = pd.DataFrame(results).sort_values(
        "ice_variance", ascending=False
    ).reset_index(drop=True)

    return heterogeneity_df


def compare_pdp_across_models(
    models: Dict[str, BaseEstimator],
    X: Union[pd.DataFrame, np.ndarray],
    feature: Union[str, int],
    feature_names: Optional[List[str]] = None,
    grid_resolution: int = 100,
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Compare partial dependence for a feature across multiple models.

    Parameters
    ----------
    models : Dict[str, BaseEstimator]
        Dictionary mapping model names to trained models
    X : pd.DataFrame or np.ndarray
        Feature matrix
    feature : str or int
        Feature name or index
    feature_names : List[str], optional
        List of feature names (required if X is np.ndarray)
    grid_resolution : int, default=100
        Number of points in the grid
    show : bool, default=True
        Whether to display the plot
    save_path : str, optional
        Path to save the figure
    title : str, optional
        Custom title for the plot
    figsize : Tuple[int, int], default=(10, 6)
        Figure size

    Returns
    -------
    plt.Figure
        The matplotlib figure object

    Examples
    --------
    >>> models = {'RF': rf_model, 'GB': gb_model, 'DT': dt_model}
    >>> fig = compare_pdp_across_models(
    ...     models, X_test, 'feature1', feature_names
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

    # Convert feature to index
    if isinstance(feature, str):
        feature_idx = feature_names.index(feature)
        feature_name = feature
    else:
        feature_idx = feature
        feature_name = feature_names[feature]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Compute and plot PDP for each model
    for model_name, model in models.items():
        pd_result = partial_dependence(
            model,
            X_df,
            features=[feature_idx],
            grid_resolution=grid_resolution,
            kind="average",
        )

        # Extract values
        pdp_values = pd_result["average"][0]
        grid_values = pd_result["grid_values"][0]

        # Plot
        ax.plot(grid_values, pdp_values, label=model_name, linewidth=2, marker='o', markersize=3)

    # Customize plot
    ax.set_xlabel(feature_name, fontsize=12)
    ax.set_ylabel("Partial Dependence", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    else:
        ax.set_title(
            f"Partial Dependence Comparison: {feature_name}",
            fontsize=14,
            fontweight="bold",
        )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved PDP comparison to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig
