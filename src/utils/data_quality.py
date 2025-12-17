"""
Data quality and cleaning utilities.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional


def remove_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    threshold: float = 1.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Remove outliers from specified columns.

    Args:
        df: Input DataFrame
        columns: List of columns to check for outliers. If None, checks all numeric columns.
        method: Method to use ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
            - For IQR: multiplier for IQR (typically 1.5 or 3.0)
            - For z-score: number of standard deviations (typically 3.0)
        verbose: Print information about outliers removed

    Returns:
        DataFrame with outliers removed
    """
    df = df.copy()
    initial_len = len(df)

    # If no columns specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude timestamp-related columns
        columns = [c for c in columns if "timestamp" not in c.lower()]

    if method == "iqr":
        # IQR method (more robust to extreme outliers)
        for col in columns:
            if col not in df.columns:
                continue

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # Mark outliers
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            n_outliers = outlier_mask.sum()

            if verbose and n_outliers > 0:
                print(
                    f"  {col}: removing {n_outliers} outliers ({n_outliers/len(df)*100:.2f}%)"
                )
                print(f"    Bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")

            # Remove rows with outliers
            df = df[~outlier_mask]

    elif method == "zscore":
        # Z-score method (assumes normal distribution)
        for col in columns:
            if col not in df.columns:
                continue

            mean = df[col].mean()
            std = df[col].std()

            if std == 0:
                continue

            z_scores = np.abs((df[col] - mean) / std)
            outlier_mask = z_scores > threshold
            n_outliers = outlier_mask.sum()

            if verbose and n_outliers > 0:
                print(
                    f"  {col}: removing {n_outliers} outliers ({n_outliers/len(df)*100:.2f}%)"
                )

            df = df[~outlier_mask]

    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")

    final_len = len(df)
    removed = initial_len - final_len

    if verbose:
        print(f"\nTotal rows removed: {removed} ({removed/initial_len*100:.2f}%)")
        print(f"Remaining rows: {final_len}")

    return df.reset_index(drop=True)


def detect_gaps(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    max_gap_seconds: float = 60.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Detect time gaps in data.

    Args:
        df: Input DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        max_gap_seconds: Maximum acceptable gap in seconds
        verbose: Print information about gaps

    Returns:
        DataFrame with gap information (start, end, duration)
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in DataFrame")

    # Ensure timestamp is datetime
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)

    # Calculate time differences
    time_diffs = df[timestamp_col].diff()

    # Find gaps
    gaps = time_diffs[time_diffs.dt.total_seconds() > max_gap_seconds]

    if len(gaps) == 0:
        if verbose:
            print(f"No gaps > {max_gap_seconds}s found")
        return pd.DataFrame()

    gap_info = []
    for idx in gaps.index:
        gap_start = df.loc[idx - 1, timestamp_col]
        gap_end = df.loc[idx, timestamp_col]
        gap_duration = (gap_end - gap_start).total_seconds()

        gap_info.append(
            {
                "gap_start": gap_start,
                "gap_end": gap_end,
                "duration_seconds": gap_duration,
                "index_before": idx - 1,
                "index_after": idx,
            }
        )

    gaps_df = pd.DataFrame(gap_info)

    if verbose:
        print(f"Found {len(gaps_df)} gaps > {max_gap_seconds}s")
        print(f"Total gap time: {gaps_df['duration_seconds'].sum():.0f}s")
        print(f"Max gap: {gaps_df['duration_seconds'].max():.0f}s")

    return gaps_df


def check_data_quality(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    numeric_cols: Optional[List[str]] = None,
) -> dict:
    """
    Comprehensive data quality check.

    Args:
        df: Input DataFrame
        timestamp_col: Name of timestamp column
        numeric_cols: List of numeric columns to check

    Returns:
        Dictionary with quality metrics
    """
    quality_report = {}

    # Basic info
    quality_report["n_rows"] = len(df)
    quality_report["n_columns"] = len(df.columns)

    # Missing values
    quality_report["missing_values"] = df.isnull().sum().to_dict()
    quality_report["missing_pct"] = (df.isnull().sum() / len(df) * 100).to_dict()

    # Duplicates (exclude columns with unhashable types like arrays)
    try:
        # Try to check duplicates on all columns
        quality_report["n_duplicates"] = df.duplicated().sum()
    except TypeError:
        # If error, exclude columns with unhashable types (e.g., arrays, lists)
        hashable_cols = []
        for col in df.columns:
            try:
                # Test if column values are hashable
                hash(df[col].iloc[0])
                hashable_cols.append(col)
            except (TypeError, AttributeError):
                # Skip unhashable columns (arrays, lists, etc.)
                continue

        if hashable_cols:
            quality_report["n_duplicates"] = df[hashable_cols].duplicated().sum()
        else:
            quality_report["n_duplicates"] = 0

    # Timestamp checks
    if timestamp_col in df.columns:
        df_sorted = df.sort_values(timestamp_col)
        quality_report["timestamp_range"] = {
            "start": df_sorted[timestamp_col].min(),
            "end": df_sorted[timestamp_col].max(),
            "duration_hours": (
                df_sorted[timestamp_col].max() - df_sorted[timestamp_col].min()
            ).total_seconds()
            / 3600,
        }

        # Check for time gaps
        time_diffs = df_sorted[timestamp_col].diff()
        quality_report["time_gaps"] = {
            "median_seconds": time_diffs.median().total_seconds(),
            "mean_seconds": time_diffs.mean().total_seconds(),
            "max_seconds": time_diffs.max().total_seconds(),
        }

    # Numeric column stats
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    quality_report["numeric_stats"] = {}
    for col in numeric_cols:
        if col in df.columns:
            quality_report["numeric_stats"][col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "zeros": (df[col] == 0).sum(),
                "negatives": (df[col] < 0).sum(),
            }

    return quality_report


def print_quality_report(quality_report: dict):
    """Pretty print quality report."""
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)

    print(f"\n Basic Info:")
    print(f"  Rows: {quality_report['n_rows']:,}")
    print(f"  Columns: {quality_report['n_columns']}")
    print(f"  Duplicates: {quality_report['n_duplicates']}")

    print(f"\n Missing Values:")
    missing = {k: v for k, v in quality_report["missing_values"].items() if v > 0}
    if missing:
        for col, count in missing.items():
            pct = quality_report["missing_pct"][col]
            print(f"  {col}: {count} ({pct:.2f}%)")
    else:
        print(" No missing values")

    if "timestamp_range" in quality_report:
        print(f"\n Timestamp Info:")
        tr = quality_report["timestamp_range"]
        print(f"  Start: {tr['start']}")
        print(f"  End: {tr['end']}")
        print(f"  Duration: {tr['duration_hours']:.2f} hours")

        tg = quality_report["time_gaps"]
        print(f"  Median gap: {tg['median_seconds']:.2f}s")
        print(f"  Max gap: {tg['max_seconds']:.2f}s")

    print("\n" + "=" * 60)
