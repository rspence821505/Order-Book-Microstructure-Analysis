"""
Decision Path Analysis Module

This module provides functions for extracting and visualizing decision paths from tree-based models:
- Tree structure visualization
- Decision path extraction for individual samples
- IF-THEN rule extraction from trees
- Path-based prediction explanation

Extracted from notebook 65_model_interpretability.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Union, Any
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import plot_tree, export_text, _tree
import warnings


def visualize_tree(
    model: Union[DecisionTreeClassifier, DecisionTreeRegressor],
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    max_depth: Optional[int] = 3,
    filled: bool = True,
    rounded: bool = True,
    fontsize: int = 8,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (25, 15),
    show: bool = True,
) -> plt.Figure:
    """
    Visualize decision tree structure.

    Creates a graphical representation of the decision tree showing splits,
    thresholds, and leaf values.

    Parameters
    ----------
    model : DecisionTreeClassifier or DecisionTreeRegressor
        Trained decision tree model
    feature_names : List[str]
        List of feature names
    class_names : List[str], optional
        List of class names for classification trees (e.g., ['Down', 'Up'])
    max_depth : int, optional, default=3
        Maximum depth to display (None for full tree)
        Recommended to limit for readability
    filled : bool, default=True
        Whether to color nodes by class/value
    rounded : bool, default=True
        Whether to use rounded boxes
    fontsize : int, default=8
        Font size for text in tree
    save_path : str, optional
        Path to save the figure
    title : str, optional
        Custom title for the plot
    figsize : Tuple[int, int], default=(25, 15)
        Figure size
    show : bool, default=True
        Whether to display the plot

    Returns
    -------
    plt.Figure
        The matplotlib figure object

    Examples
    --------
    >>> fig = visualize_tree(
    ...     dt_model, feature_names, class_names=['Down', 'Up'],
    ...     max_depth=3, save_path="tree_structure.png"
    ... )
    """
    if not isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
        raise TypeError(
            f"Model must be DecisionTreeClassifier or DecisionTreeRegressor, "
            f"got {type(model).__name__}"
        )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot tree
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=filled,
        rounded=rounded,
        fontsize=fontsize,
        ax=ax,
        max_depth=max_depth,
    )

    # Add title
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    else:
        depth_str = f"Max Depth {max_depth}" if max_depth else "Full Tree"
        ax.set_title(
            f"Decision Tree Structure ({depth_str})",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved decision tree visualization to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def export_tree_rules(
    model: Union[DecisionTreeClassifier, DecisionTreeRegressor],
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    max_depth: Optional[int] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Export decision tree as text-based IF-THEN rules.

    Parameters
    ----------
    model : DecisionTreeClassifier or DecisionTreeRegressor
        Trained decision tree model
    feature_names : List[str]
        List of feature names
    class_names : List[str], optional
        List of class names for classification trees
    max_depth : int, optional
        Maximum depth to export (None for full tree)
    save_path : str, optional
        Path to save the rules text file

    Returns
    -------
    str
        Text representation of tree rules

    Examples
    --------
    >>> rules = export_tree_rules(
    ...     dt_model, feature_names, class_names=['Down', 'Up'],
    ...     max_depth=4, save_path="tree_rules.txt"
    ... )
    >>> print(rules[:500])
    """
    if not isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
        raise TypeError(
            f"Model must be DecisionTreeClassifier or DecisionTreeRegressor, "
            f"got {type(model).__name__}"
        )

    # Export tree rules
    tree_rules = export_text(
        model,
        feature_names=feature_names,
        max_depth=max_depth,
        class_names=class_names,
    )

    # Save if path provided
    if save_path:
        with open(save_path, "w") as f:
            f.write(tree_rules)
        print(f"✓ Saved decision tree rules to {save_path}")

    return tree_rules


def extract_decision_paths(
    model: Union[DecisionTreeClassifier, DecisionTreeRegressor],
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: List[str],
    sample_indices: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract decision paths for specific samples.

    For each sample, returns the sequence of decision rules (splits) that
    lead to the final prediction.

    Parameters
    ----------
    model : DecisionTreeClassifier or DecisionTreeRegressor
        Trained decision tree model
    X : pd.DataFrame or np.ndarray
        Feature matrix
    feature_names : List[str]
        List of feature names
    sample_indices : List[int], optional
        Specific sample indices to extract paths for
        If None, extracts for all samples

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries, one per sample, containing:
        - 'sample_idx': sample index
        - 'rules': list of decision rules (strings)
        - 'leaf_id': ID of the leaf node
        - 'prediction': model prediction for this sample
        - 'n_splits': number of splits in the path

    Examples
    --------
    >>> paths = extract_decision_paths(
    ...     dt_model, X_test, feature_names, sample_indices=[0, 10, 50]
    ... )
    >>> for path in paths:
    ...     print(f"Sample {path['sample_idx']}: {path['n_splits']} splits")
    ...     for rule in path['rules']:
    ...         print(f"  - {rule}")
    """
    if not isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
        raise TypeError(
            f"Model must be DecisionTreeClassifier or DecisionTreeRegressor, "
            f"got {type(model).__name__}"
        )

    # Convert to DataFrame if needed
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()

    # Use all samples if indices not specified
    if sample_indices is None:
        sample_indices = list(range(len(X_df)))

    # Get tree structure
    tree_ = model.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    # Get decision path and leaf IDs for all samples
    node_indicator = model.decision_path(X_df)
    leaf_ids = model.apply(X_df)

    # Get predictions
    if isinstance(model, DecisionTreeClassifier):
        predictions = model.predict(X_df)
        probabilities = model.predict_proba(X_df)
    else:
        predictions = model.predict(X_df)
        probabilities = None

    # Extract paths for each sample
    paths = []

    for sample_idx in sample_indices:
        if sample_idx >= len(X_df):
            warnings.warn(f"Sample index {sample_idx} out of bounds, skipping")
            continue

        # Get nodes in the path for this sample
        node_index = node_indicator.indices[
            node_indicator.indptr[sample_idx] : node_indicator.indptr[sample_idx + 1]
        ]

        # Extract rules
        rules = []
        for node_id in node_index:
            # Skip leaf nodes
            if leaf_ids[sample_idx] == node_id:
                continue

            # Determine if we went left or right
            threshold = tree_.threshold[node_id]
            feature_idx = tree_.feature[node_id]
            feature_value = X_df.iloc[sample_idx, feature_idx]

            if feature_value <= threshold:
                inequality = "<="
            else:
                inequality = ">"

            # Create rule string
            rule = f"{feature_name[node_id]} {inequality} {threshold:.4f}"
            rules.append(rule)

        # Build path info
        path_info = {
            "sample_idx": int(sample_idx),
            "rules": rules,
            "leaf_id": int(leaf_ids[sample_idx]),
            "prediction": predictions[sample_idx],
            "n_splits": len(rules),
        }

        if probabilities is not None:
            path_info["probabilities"] = probabilities[sample_idx].tolist()

        paths.append(path_info)

    return paths


def path_to_rules(
    path: Dict[str, Any],
    class_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> str:
    """
    Convert a decision path to a human-readable rule string.

    Parameters
    ----------
    path : Dict[str, Any]
        Path dictionary from extract_decision_paths()
    class_names : List[str], optional
        Class names for classification (e.g., ['Down', 'Up'])
    verbose : bool, default=True
        Whether to include additional information (probabilities, etc.)

    Returns
    -------
    str
        Human-readable rule string

    Examples
    --------
    >>> paths = extract_decision_paths(dt_model, X_test, feature_names, [0])
    >>> rule_str = path_to_rules(paths[0], class_names=['Down', 'Up'])
    >>> print(rule_str)
    """
    lines = []

    # Header
    lines.append(f"Sample {path['sample_idx']}:")

    # Rules
    for i, rule in enumerate(path["rules"], 1):
        lines.append(f"  {i}. IF {rule}")

    # Prediction
    prediction = path["prediction"]
    if class_names is not None and isinstance(prediction, (int, np.integer)):
        prediction_str = class_names[int(prediction)]
    else:
        prediction_str = str(prediction)

    lines.append(f"   THEN predict: {prediction_str}")

    # Probabilities if available and verbose
    if verbose and "probabilities" in path:
        probs = path["probabilities"]
        if class_names is not None and len(class_names) == len(probs):
            prob_strs = [f"{name}={prob:.3f}" for name, prob in zip(class_names, probs)]
            lines.append(f"   Probabilities: {', '.join(prob_strs)}")
        else:
            lines.append(f"   Probabilities: {probs}")

    return "\n".join(lines)


def print_decision_paths(
    paths: List[Dict[str, Any]],
    class_names: Optional[List[str]] = None,
    max_paths: Optional[int] = None,
) -> None:
    """
    Print decision paths in a readable format.

    Parameters
    ----------
    paths : List[Dict[str, Any]]
        List of path dictionaries from extract_decision_paths()
    class_names : List[str], optional
        Class names for classification
    max_paths : int, optional
        Maximum number of paths to print

    Examples
    --------
    >>> paths = extract_decision_paths(dt_model, X_test, feature_names)
    >>> print_decision_paths(paths, class_names=['Down', 'Up'], max_paths=5)
    """
    if max_paths is not None:
        paths = paths[:max_paths]

    print("=" * 80)
    print(f"DECISION PATHS ({len(paths)} samples)")
    print("=" * 80)

    for path in paths:
        print("\n" + path_to_rules(path, class_names))

    print("\n" + "=" * 80)


def get_feature_usage_in_tree(
    model: Union[DecisionTreeClassifier, DecisionTreeRegressor],
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Analyze which features are used in the decision tree and how often.

    Parameters
    ----------
    model : DecisionTreeClassifier or DecisionTreeRegressor
        Trained decision tree model
    feature_names : List[str]
        List of feature names

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature', 'n_splits', 'importance']
        showing feature usage statistics

    Examples
    --------
    >>> usage_df = get_feature_usage_in_tree(dt_model, feature_names)
    >>> print(usage_df.head(10))
    """
    if not isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
        raise TypeError(
            f"Model must be DecisionTreeClassifier or DecisionTreeRegressor, "
            f"got {type(model).__name__}"
        )

    tree_ = model.tree_
    n_features = len(feature_names)

    # Count splits per feature
    split_counts = np.zeros(n_features, dtype=int)

    for node_id in range(tree_.node_count):
        feature_idx = tree_.feature[node_id]
        if feature_idx != _tree.TREE_UNDEFINED:
            split_counts[feature_idx] += 1

    # Create DataFrame
    usage_df = pd.DataFrame({
        "feature": feature_names,
        "n_splits": split_counts,
        "importance": model.feature_importances_,
    })

    # Sort by number of splits
    usage_df = usage_df.sort_values("n_splits", ascending=False).reset_index(drop=True)

    return usage_df


def find_similar_paths(
    paths: List[Dict[str, Any]],
    reference_idx: int,
    similarity_threshold: float = 0.5,
) -> List[Tuple[int, float]]:
    """
    Find samples with similar decision paths to a reference sample.

    Similarity is based on the Jaccard similarity of the sets of rules.

    Parameters
    ----------
    paths : List[Dict[str, Any]]
        List of path dictionaries from extract_decision_paths()
    reference_idx : int
        Index of the reference path in the paths list
    similarity_threshold : float, default=0.5
        Minimum similarity score (0-1) to be considered similar

    Returns
    -------
    List[Tuple[int, float]]
        List of (path_index, similarity_score) tuples, sorted by similarity (descending)

    Examples
    --------
    >>> paths = extract_decision_paths(dt_model, X_test, feature_names)
    >>> similar = find_similar_paths(paths, reference_idx=0, similarity_threshold=0.6)
    >>> print(f"Found {len(similar)} similar paths")
    """
    if reference_idx >= len(paths):
        raise ValueError(f"reference_idx {reference_idx} out of bounds")

    reference_rules = set(paths[reference_idx]["rules"])

    similar_paths = []

    for i, path in enumerate(paths):
        if i == reference_idx:
            continue

        # Compute Jaccard similarity
        path_rules = set(path["rules"])
        intersection = len(reference_rules & path_rules)
        union = len(reference_rules | path_rules)

        if union == 0:
            similarity = 0.0
        else:
            similarity = intersection / union

        if similarity >= similarity_threshold:
            similar_paths.append((i, similarity))

    # Sort by similarity (descending)
    similar_paths.sort(key=lambda x: x[1], reverse=True)

    return similar_paths


def analyze_decision_tree_depth(
    model: Union[DecisionTreeClassifier, DecisionTreeRegressor],
    X: Union[pd.DataFrame, np.ndarray],
) -> Dict[str, Any]:
    """
    Analyze the depth distribution of samples in the decision tree.

    Parameters
    ----------
    model : DecisionTreeClassifier or DecisionTreeRegressor
        Trained decision tree model
    X : pd.DataFrame or np.ndarray
        Feature matrix

    Returns
    -------
    Dict[str, Any]
        Dictionary containing depth statistics:
        - 'mean_depth': average path depth
        - 'median_depth': median path depth
        - 'min_depth': minimum path depth
        - 'max_depth': maximum path depth
        - 'depth_distribution': counts per depth level

    Examples
    --------
    >>> depth_stats = analyze_decision_tree_depth(dt_model, X_test)
    >>> print(f"Average depth: {depth_stats['mean_depth']:.2f}")
    >>> print(f"Max depth: {depth_stats['max_depth']}")
    """
    if not isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
        raise TypeError(
            f"Model must be DecisionTreeClassifier or DecisionTreeRegressor, "
            f"got {type(model).__name__}"
        )

    # Get decision path
    node_indicator = model.decision_path(X)

    # Compute depth for each sample (number of nodes in path - 1)
    depths = np.diff(node_indicator.indptr) - 1

    # Compute statistics
    depth_stats = {
        "mean_depth": float(np.mean(depths)),
        "median_depth": float(np.median(depths)),
        "min_depth": int(np.min(depths)),
        "max_depth": int(np.max(depths)),
        "std_depth": float(np.std(depths)),
        "depth_distribution": dict(zip(*np.unique(depths, return_counts=True))),
    }

    return depth_stats


def visualize_depth_distribution(
    depth_stats: Dict[str, Any],
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Visualize the distribution of path depths in the decision tree.

    Parameters
    ----------
    depth_stats : Dict[str, Any]
        Depth statistics from analyze_decision_tree_depth()
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
    >>> depth_stats = analyze_decision_tree_depth(dt_model, X_test)
    >>> fig = visualize_depth_distribution(
    ...     depth_stats, save_path="depth_distribution.png"
    ... )
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract depth distribution
    depths = sorted(depth_stats["depth_distribution"].keys())
    counts = [depth_stats["depth_distribution"][d] for d in depths]

    # Create bar plot
    ax.bar(depths, counts, alpha=0.7, edgecolor="black")

    # Add mean line
    mean_depth = depth_stats["mean_depth"]
    ax.axvline(
        mean_depth,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_depth:.2f}",
    )

    # Labels and title
    ax.set_xlabel("Path Depth", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    else:
        ax.set_title(
            "Decision Tree Path Depth Distribution",
            fontsize=14,
            fontweight="bold",
        )

    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Add statistics text box
    stats_text = (
        f"Mean: {depth_stats['mean_depth']:.2f}\n"
        f"Median: {depth_stats['median_depth']:.0f}\n"
        f"Min: {depth_stats['min_depth']}\n"
        f"Max: {depth_stats['max_depth']}\n"
        f"Std: {depth_stats['std_depth']:.2f}"
    )

    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved depth distribution plot to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def extract_leaf_statistics(
    model: Union[DecisionTreeClassifier, DecisionTreeRegressor],
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Extract statistics for each leaf node in the tree.

    Parameters
    ----------
    model : DecisionTreeClassifier or DecisionTreeRegressor
        Trained decision tree model
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : np.ndarray, optional
        True labels (for computing leaf accuracy)

    Returns
    -------
    pd.DataFrame
        DataFrame with leaf statistics:
        - 'leaf_id': ID of the leaf node
        - 'n_samples': number of samples in this leaf
        - 'prediction': leaf prediction
        - 'accuracy': accuracy in this leaf (if y provided)

    Examples
    --------
    >>> leaf_stats = extract_leaf_statistics(dt_model, X_test, y_test)
    >>> print(leaf_stats.sort_values('n_samples', ascending=False).head(10))
    """
    if not isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
        raise TypeError(
            f"Model must be DecisionTreeClassifier or DecisionTreeRegressor, "
            f"got {type(model).__name__}"
        )

    # Get leaf IDs for all samples
    leaf_ids = model.apply(X)
    predictions = model.predict(X)

    # Count samples per leaf
    unique_leaves, leaf_counts = np.unique(leaf_ids, return_counts=True)

    # Build statistics
    leaf_stats = []

    for leaf_id, n_samples in zip(unique_leaves, leaf_counts):
        # Get samples in this leaf
        mask = leaf_ids == leaf_id
        leaf_prediction = predictions[mask][0]  # All predictions in leaf are same

        stat = {
            "leaf_id": int(leaf_id),
            "n_samples": int(n_samples),
            "prediction": leaf_prediction,
        }

        # Compute accuracy if y provided
        if y is not None:
            y_leaf = y[mask]
            pred_leaf = predictions[mask]
            accuracy = np.mean(y_leaf == pred_leaf)
            stat["accuracy"] = float(accuracy)

        leaf_stats.append(stat)

    # Create DataFrame
    leaf_df = pd.DataFrame(leaf_stats)
    leaf_df = leaf_df.sort_values("n_samples", ascending=False).reset_index(drop=True)

    return leaf_df
