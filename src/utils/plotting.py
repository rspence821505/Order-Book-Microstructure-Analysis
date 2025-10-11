"""
Plotting utilities for LOB feature visualization and analysis.

This module provides functions for visualizing limit order book features,
including distributions, correlations, time series, and feature relationships.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict

# Set default style
# Visualization settings
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.figsize": (16, 10),
    }
)


# Import the feature categorization function
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from features.basic_features import get_feature_columns


def plot_feature_distributions(
    df: pd.DataFrame,
    feature_cols: list[str] = None,
    max_features: int = 20,
    save_path: str = None,
):
    """
    Plot distribution histograms for engineered features.

    Args:
        df: DataFrame with features
        feature_cols: List of features to plot (if None, uses all engineered features)
        max_features: Maximum number of features to plot at once
        save_path: Optional path to save figure
    """
    if feature_cols is None:
        feature_groups = get_feature_columns(df)
        feature_cols = feature_groups["all_engineered"]

    # Limit to max_features
    feature_cols = feature_cols[:max_features]

    n_features = len(feature_cols)
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, col in enumerate(feature_cols):
        ax = axes[idx]
        data = df[col].dropna()

        # Plot histogram
        ax.hist(data, bins=50, alpha=0.7, color="steelblue", edgecolor="black")

        # Add mean and median lines
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.4f}",
        )
        ax.axvline(
            median_val,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_val:.4f}",
        )

        ax.set_title(f"{col}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Saved to: {save_path}")

    plt.show()


def plot_feature_boxplots(
    df: pd.DataFrame,
    feature_cols: list[str] = None,
    max_features: int = 20,
    save_path: str = None,
):
    """
    Plot box plots to visualize feature distributions and outliers.

    Args:
        df: DataFrame with features
        feature_cols: List of features to plot
        max_features: Maximum features to display
        save_path: Optional save path
    """
    if feature_cols is None:
        feature_groups = get_feature_columns(df)
        feature_cols = feature_groups["all_engineered"]

    feature_cols = feature_cols[:max_features]

    fig, ax = plt.subplots(figsize=(18, 8))

    # Prepare data for boxplot
    data_to_plot = [df[col].dropna() for col in feature_cols]

    bp = ax.boxplot(
        data_to_plot,
        labels=feature_cols,
        patch_artist=True,
        showmeans=True,
        meanline=True,
    )

    # Color the boxes
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)

    ax.set_xlabel("Features", fontsize=12, fontweight="bold")
    ax.set_ylabel("Value", fontsize=12, fontweight="bold")
    ax.set_title("Feature Distributions (Box Plots)", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Saved to: {save_path}")

    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    feature_cols: list[str] = None,
    method: str = "pearson",
    save_path: str = None,
):
    """
    Plot correlation heatmap for engineered features.

    Args:
        df: DataFrame with features
        feature_cols: List of features to correlate
        method: 'pearson', 'spearman', or 'kendall'
        save_path: Optional save path
    """
    if feature_cols is None:
        feature_groups = get_feature_columns(df)
        feature_cols = feature_groups["all_engineered"]

    # Compute correlation matrix
    corr_matrix = df[feature_cols].corr(method=method)

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 18))

    # Plot heatmap
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": f"{method.capitalize()} Correlation"},
        ax=ax,
    )

    ax.set_title(
        f"Feature Correlation Matrix ({method.capitalize()})",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Saved to: {save_path}")

    plt.show()

    return corr_matrix


def plot_feature_group_comparison(
    df: pd.DataFrame, summary_stats: pd.DataFrame, save_path: str = None
):
    """
    Plot comparison of key statistics across feature groups.

    Args:
        df: DataFrame with features
        summary_stats: Output from compute_summary_stats
        save_path: Optional save path
    """
    feature_groups = get_feature_columns(df)

    # Create group labels for each feature
    group_labels = []
    for feat in summary_stats.index:
        if feat in feature_groups["spread"]:
            group_labels.append("Spread")
        elif feat in feature_groups["returns"]:
            group_labels.append("Returns")
        elif feat in feature_groups["imbalance"]:
            group_labels.append("Imbalance")
        elif feat in feature_groups["depth"]:
            group_labels.append("Depth")
        elif feat in feature_groups["queue"]:
            group_labels.append("Queue")
        elif feat in feature_groups["time"]:
            group_labels.append("Time")
        elif feat in feature_groups["price"]:
            group_labels.append("Price")
        elif feat in feature_groups["volume"]:
            group_labels.append("Volume")
        else:
            group_labels.append("Other")

    summary_stats["group"] = group_labels

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Mean by group
    group_means = summary_stats.groupby("group")["mean"].mean().sort_values()
    axes[0, 0].barh(range(len(group_means)), group_means.values, color="steelblue")
    axes[0, 0].set_yticks(range(len(group_means)))
    axes[0, 0].set_yticklabels(group_means.index)
    axes[0, 0].set_xlabel("Average Mean Value")
    axes[0, 0].set_title("Average Mean by Feature Group", fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3, axis="x")

    # 2. Std by group
    group_std = summary_stats.groupby("group")["std"].mean().sort_values()
    axes[0, 1].barh(range(len(group_std)), group_std.values, color="coral")
    axes[0, 1].set_yticks(range(len(group_std)))
    axes[0, 1].set_yticklabels(group_std.index)
    axes[0, 1].set_xlabel("Average Std Dev")
    axes[0, 1].set_title("Average Std Dev by Feature Group", fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3, axis="x")

    # 3. Skewness by group
    group_skew = summary_stats.groupby("group")["skewness"].mean().sort_values()
    axes[0, 2].barh(range(len(group_skew)), group_skew.values, color="lightgreen")
    axes[0, 2].set_yticks(range(len(group_skew)))
    axes[0, 2].set_yticklabels(group_skew.index)
    axes[0, 2].set_xlabel("Average Skewness")
    axes[0, 2].set_title("Average Skewness by Feature Group", fontweight="bold")
    axes[0, 2].axvline(x=0, color="black", linestyle="--", linewidth=1)
    axes[0, 2].grid(True, alpha=0.3, axis="x")

    # 4. Kurtosis by group
    group_kurt = summary_stats.groupby("group")["kurtosis"].mean().sort_values()
    axes[1, 0].barh(range(len(group_kurt)), group_kurt.values, color="orchid")
    axes[1, 0].set_yticks(range(len(group_kurt)))
    axes[1, 0].set_yticklabels(group_kurt.index)
    axes[1, 0].set_xlabel("Average Kurtosis")
    axes[1, 0].set_title("Average Kurtosis by Feature Group", fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3, axis="x")

    # 5. Missing % by group
    group_missing = summary_stats.groupby("group")["missing_pct"].mean().sort_values()
    axes[1, 1].barh(range(len(group_missing)), group_missing.values, color="salmon")
    axes[1, 1].set_yticks(range(len(group_missing)))
    axes[1, 1].set_yticklabels(group_missing.index)
    axes[1, 1].set_xlabel("Average Missing %")
    axes[1, 1].set_title("Average Missing Data % by Feature Group", fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3, axis="x")

    # 6. Outlier % by group
    group_outliers = summary_stats.groupby("group")["outlier_pct"].mean().sort_values()
    axes[1, 2].barh(range(len(group_outliers)), group_outliers.values, color="gold")
    axes[1, 2].set_yticks(range(len(group_outliers)))
    axes[1, 2].set_yticklabels(group_outliers.index)
    axes[1, 2].set_xlabel("Average Outlier %")
    axes[1, 2].set_title("Average Outlier % by Feature Group", fontweight="bold")
    axes[1, 2].grid(True, alpha=0.3, axis="x")

    plt.suptitle("Feature Statistics by Group", fontsize=16, fontweight="bold", y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Saved to: {save_path}")

    plt.show()


def plot_time_series_features(
    df: pd.DataFrame,
    feature_cols: list[str] = None,
    n_samples: int = 500,
    save_path: str = None,
):
    """
    Plot time series of selected features.

    Args:
        df: DataFrame with timestamp and features
        feature_cols: Features to plot (max 6 recommended)
        n_samples: Number of recent samples to plot
        save_path: Optional save path
    """
    if feature_cols is None:
        # Default to key features
        feature_cols = [
            "mid_price",
            "spread_abs",
            "imbalance_top",
            "total_depth",
            "log_return_lag_1",
            "depth_imbalance_5",
        ]

    # Limit to max 6 features and last n_samples
    feature_cols = feature_cols[:6]
    df_plot = df.tail(n_samples)

    n_features = len(feature_cols)
    fig, axes = plt.subplots(n_features, 1, figsize=(16, 3 * n_features))

    if n_features == 1:
        axes = [axes]

    for idx, col in enumerate(feature_cols):
        ax = axes[idx]
        ax.plot(df_plot.index, df_plot[col], linewidth=1, alpha=0.8)
        ax.set_ylabel(col, fontweight="bold")
        ax.set_title(
            f"{col} over time (last {n_samples} observations)",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        # Add mean line
        mean_val = df_plot[col].mean()
        ax.axhline(
            y=mean_val,
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
            label=f"Mean: {mean_val:.4f}",
        )
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Observation Index", fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Saved to: {save_path}")

    plt.show()


def plot_feature_relationships(
    df: pd.DataFrame, feature_pairs: list[tuple] = None, save_path: str = None
):
    """
    Plot scatter plots showing relationships between feature pairs.

    Args:
        df: DataFrame with features
        feature_pairs: List of (x_feature, y_feature) tuples
        save_path: Optional save path
    """
    if feature_pairs is None:
        # Default interesting pairs
        feature_pairs = [
            ("spread_abs", "imbalance_top"),
            ("total_depth", "spread_abs"),
            ("log_return_lag_1", "imbalance_top"),
            ("depth_imbalance_5", "log_return_lag_1"),
            ("mid_price", "weighted_mid"),
            ("bid_depth_5", "ask_depth_5"),
        ]

    n_pairs = len(feature_pairs)
    n_cols = 3
    n_rows = int(np.ceil(n_pairs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_pairs > 1 else [axes]

    for idx, (x_col, y_col) in enumerate(feature_pairs):
        ax = axes[idx]

        # Sample if dataset is large
        if len(df) > 5000:
            df_sample = df.sample(n=5000, random_state=42)
        else:
            df_sample = df

        ax.scatter(
            df_sample[x_col], df_sample[y_col], alpha=0.3, s=10, color="steelblue"
        )

        # Add correlation
        corr = df[x_col].corr(df[y_col])
        ax.set_xlabel(x_col, fontweight="bold")
        ax.set_ylabel(y_col, fontweight="bold")
        ax.set_title(
            f"{x_col} vs {y_col}\nCorrelation: {corr:.3f}",
            fontsize=10,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Saved to: {save_path}")

    plt.show()


def create_comprehensive_feature_report(
    df: pd.DataFrame, summary_stats: pd.DataFrame, output_dir: str = "reports/figures"
):
    """
    Create a comprehensive set of visualizations for all engineered features.

    Args:
        df: LOB DataFrame with features
        summary_stats: Output from compute_summary_stats
        output_dir: Directory to save all figures
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    feature_groups = get_feature_columns(df)

    print("\n" + "=" * 80)
    print("CREATING COMPREHENSIVE FEATURE VISUALIZATIONS")
    print("=" * 80)

    # 1. Distributions for all features (in batches)
    print("\n📊 Creating distribution plots...")
    all_features = feature_groups["all_engineered"]
    batch_size = 20
    for i in range(0, len(all_features), batch_size):
        batch = all_features[i : i + batch_size]
        plot_feature_distributions(
            df,
            batch,
            max_features=batch_size,
            save_path=f"{output_dir}/distributions_batch_{i//batch_size + 1}.png",
        )

    # 2. Feature group comparison
    print("\n📊 Creating feature group comparison...")
    plot_feature_group_comparison(
        df, summary_stats, save_path=f"{output_dir}/feature_group_comparison.png"
    )

    # 3. Correlation heatmap
    print("\n📊 Creating correlation heatmap...")
    plot_correlation_heatmap(
        df, all_features, save_path=f"{output_dir}/correlation_heatmap.png"
    )

    # 4. Time series of key features
    print("\n📊 Creating time series plots...")
    plot_time_series_features(
        df,
        feature_cols=[
            "mid_price",
            "spread_abs",
            "imbalance_top",
            "total_depth",
            "log_return_lag_1",
            "depth_imbalance_5",
        ],
        save_path=f"{output_dir}/time_series_key_features.png",
    )

    # 5. Feature relationships
    print("\n📊 Creating relationship plots...")
    plot_feature_relationships(df, save_path=f"{output_dir}/feature_relationships.png")

    # 6. Individual feature group visualizations
    for group_name in ["spread", "imbalance", "depth", "returns"]:
        if group_name in feature_groups and len(feature_groups[group_name]) > 0:
            print(f"\n📊 Creating {group_name} features visualization...")
            plot_feature_distributions(
                df,
                feature_groups[group_name],
                max_features=20,
                save_path=f"{output_dir}/{group_name}_features.png",
            )

    print("\n" + "=" * 80)
    print(f"✅ All visualizations saved to: {output_dir}/")
    print("=" * 80 + "\n")
