"""
Trade and Quote Data Synchronizer

This module provides functionality for time-aligning and synchronizing trade data with
NBBO quote data and L2 order book snapshots for U.S. equity microstructure analysis.

Proper synchronization is critical for:
    - Computing accurate trade aggressiveness (requires quote at trade time)
    - Analyzing LOB state before/after trades
    - Building features that combine trade and order book information
    - Ensuring temporal consistency in microstructure analysis

Author: Rylan Spence
Date: 2025
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Literal, Union
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSynchronizer:
    """
    Synchronize trade, quote, and order book data streams with various alignment strategies.

    Supports multiple synchronization methods:
        - as-of merge: Match each trade with most recent quote before trade
        - nearest merge: Match each trade with temporally nearest quote
        - forward-fill: Fill quote data forward in time
        - sampling: Resample all data to common frequency

    Attributes:
        trades (pd.DataFrame): Trade tick data
        quotes (pd.DataFrame): NBBO quote data
        lob (pd.DataFrame): L2 order book snapshots
        tolerance (pd.Timedelta): Maximum time difference for matching

    Example:
        >>> sync = DataSynchronizer(trades=trades, quotes=quotes)
        >>> aligned_data = sync.align_trades_with_quotes(method='asof')
        >>> print(f"Aligned {len(aligned_data)} records")
    """

    def __init__(
        self,
        trades: Optional[pd.DataFrame] = None,
        quotes: Optional[pd.DataFrame] = None,
        lob: Optional[pd.DataFrame] = None,
        tolerance: str = '10ms'
    ):
        """
        Initialize the data synchronizer.

        Args:
            trades: Trade data with 'timestamp' column
            quotes: NBBO quote data with 'timestamp' column
            lob: L2 order book snapshots with 'timestamp' column
            tolerance: Maximum time difference for matching (pandas offset string)
        """
        self.trades = trades
        self.quotes = quotes
        self.lob = lob
        self.tolerance = pd.Timedelta(tolerance)

        # Validate data
        self._validate_data()

    def _validate_data(self):
        """Validate that all provided DataFrames have required timestamp column."""
        for name, df in [('trades', self.trades), ('quotes', self.quotes), ('lob', self.lob)]:
            if df is not None:
                if not isinstance(df, pd.DataFrame):
                    raise TypeError(f"{name} must be a pandas DataFrame")
                if 'timestamp' not in df.columns:
                    raise ValueError(f"{name} DataFrame must have 'timestamp' column")
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    logger.warning(f"{name} timestamp column is not datetime64, converting...")
                    df['timestamp'] = pd.to_datetime(df['timestamp'])

    def align_trades_with_quotes(
        self,
        method: Literal['asof', 'nearest', 'interpolate'] = 'asof',
        direction: Literal['backward', 'forward', 'nearest'] = 'backward',
        suffix: Tuple[str, str] = ('_trade', '_quote')
    ) -> pd.DataFrame:
        """
        Align trade data with NBBO quote data.

        This is the most common synchronization operation in equity microstructure analysis,
        enabling:
            - Lee-Ready trade classification
            - Spread decomposition analysis
            - Price impact measurement

        Args:
            method: Alignment method:
                - 'asof': Use most recent quote relative to trade (default)
                - 'nearest': Use temporally nearest quote (forward or backward)
                - 'interpolate': Linearly interpolate quote values
            direction: For 'asof' method:
                - 'backward': Use quote before trade (most common)
                - 'forward': Use quote after trade
                - 'nearest': Use closest quote
            suffix: Tuple of suffixes for overlapping columns (left, right)

        Returns:
            DataFrame with aligned trade and quote data

        Example:
            >>> aligned = sync.align_trades_with_quotes(method='asof', direction='backward')
            >>> # Each trade now has bid_price, ask_price from most recent quote
        """
        if self.trades is None or self.quotes is None:
            raise ValueError("Both trades and quotes must be provided")

        logger.info(
            f"Aligning {len(self.trades):,} trades with {len(self.quotes):,} quotes "
            f"(method: {method}, direction: {direction})"
        )

        # Ensure both are sorted by timestamp
        trades_sorted = self.trades.sort_values('timestamp').copy()
        quotes_sorted = self.quotes.sort_values('timestamp').copy()

        # Set timestamp as index for merge_asof
        trades_sorted = trades_sorted.set_index('timestamp')
        quotes_sorted = quotes_sorted.set_index('timestamp')

        if method == 'asof':
            # Use pandas merge_asof for efficient as-of join
            aligned = pd.merge_asof(
                trades_sorted,
                quotes_sorted,
                left_index=True,
                right_index=True,
                direction=direction,
                tolerance=self.tolerance,
                suffixes=suffix
            )

        elif method == 'nearest':
            # Use merge_asof with nearest direction
            aligned = pd.merge_asof(
                trades_sorted,
                quotes_sorted,
                left_index=True,
                right_index=True,
                direction='nearest',
                tolerance=self.tolerance,
                suffixes=suffix
            )

        elif method == 'interpolate':
            # Interpolate quote values at trade timestamps
            # Reindex quotes to union of timestamps, then interpolate
            all_timestamps = trades_sorted.index.union(quotes_sorted.index)
            quotes_reindexed = quotes_sorted.reindex(all_timestamps)

            # Interpolate numeric columns
            numeric_cols = quotes_reindexed.select_dtypes(include=[np.number]).columns
            quotes_reindexed[numeric_cols] = quotes_reindexed[numeric_cols].interpolate(
                method='time'
            )

            # Merge with trades
            aligned = pd.merge(
                trades_sorted,
                quotes_reindexed,
                left_index=True,
                right_index=True,
                how='left',
                suffixes=suffix
            )

        else:
            raise ValueError(f"Unknown alignment method: {method}")

        aligned = aligned.reset_index()

        # Count successfully matched records
        quote_cols = [col for col in aligned.columns if col.endswith(suffix[1])]
        if quote_cols:
            matched = aligned[quote_cols[0]].notna().sum()
            match_rate = matched / len(aligned) * 100
            logger.info(
                f"Matched {matched:,} / {len(aligned):,} trades with quotes ({match_rate:.2f}%)"
            )

        return aligned

    def align_trades_with_lob(
        self,
        method: Literal['asof', 'nearest'] = 'asof',
        direction: Literal['backward', 'forward', 'nearest'] = 'backward',
        suffix: Tuple[str, str] = ('_trade', '_lob')
    ) -> pd.DataFrame:
        """
        Align trade data with L2 order book snapshots.

        Useful for analyzing:
            - LOB state immediately before trade execution
            - Order book depth at trade time
            - Price impact on LOB levels

        Args:
            method: Alignment method ('asof' or 'nearest')
            direction: For 'asof': 'backward', 'forward', or 'nearest'
            suffix: Tuple of suffixes for overlapping columns

        Returns:
            DataFrame with aligned trade and LOB data
        """
        if self.trades is None or self.lob is None:
            raise ValueError("Both trades and LOB data must be provided")

        logger.info(
            f"Aligning {len(self.trades):,} trades with {len(self.lob):,} LOB snapshots "
            f"(method: {method})"
        )

        trades_sorted = self.trades.sort_values('timestamp').set_index('timestamp')
        lob_sorted = self.lob.sort_values('timestamp').set_index('timestamp')

        if method == 'asof':
            aligned = pd.merge_asof(
                trades_sorted,
                lob_sorted,
                left_index=True,
                right_index=True,
                direction=direction,
                tolerance=self.tolerance,
                suffixes=suffix
            )
        elif method == 'nearest':
            aligned = pd.merge_asof(
                trades_sorted,
                lob_sorted,
                left_index=True,
                right_index=True,
                direction='nearest',
                tolerance=self.tolerance,
                suffixes=suffix
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        aligned = aligned.reset_index()

        lob_cols = [col for col in aligned.columns if col.endswith(suffix[1])]
        if lob_cols:
            matched = aligned[lob_cols[0]].notna().sum()
            match_rate = matched / len(aligned) * 100
            logger.info(
                f"Matched {matched:,} / {len(aligned):,} trades with LOB ({match_rate:.2f}%)"
            )

        return aligned

    def align_quotes_with_lob(
        self,
        method: Literal['asof', 'nearest'] = 'asof',
        direction: Literal['backward', 'forward', 'nearest'] = 'backward'
    ) -> pd.DataFrame:
        """
        Align NBBO quotes with L2 order book snapshots.

        Useful for:
            - Comparing top-of-book (NBBO) with full LOB depth
            - Validating NBBO consistency with aggregated LOB
            - Analyzing depth beyond best bid/offer

        Args:
            method: Alignment method ('asof' or 'nearest')
            direction: For 'asof': 'backward', 'forward', or 'nearest'

        Returns:
            DataFrame with aligned quote and LOB data
        """
        if self.quotes is None or self.lob is None:
            raise ValueError("Both quotes and LOB data must be provided")

        logger.info(
            f"Aligning {len(self.quotes):,} quotes with {len(self.lob):,} LOB snapshots"
        )

        quotes_sorted = self.quotes.sort_values('timestamp').set_index('timestamp')
        lob_sorted = self.lob.sort_values('timestamp').set_index('timestamp')

        if method == 'asof':
            aligned = pd.merge_asof(
                quotes_sorted,
                lob_sorted,
                left_index=True,
                right_index=True,
                direction=direction,
                tolerance=self.tolerance,
                suffixes=('_quote', '_lob')
            )
        else:
            aligned = pd.merge_asof(
                quotes_sorted,
                lob_sorted,
                left_index=True,
                right_index=True,
                direction='nearest',
                tolerance=self.tolerance,
                suffixes=('_quote', '_lob')
            )

        return aligned.reset_index()

    def create_unified_timeline(
        self,
        freq: str = '1S',
        method: str = 'ffill'
    ) -> pd.DataFrame:
        """
        Create a unified timeline by resampling all data sources to a common frequency.

        This approach:
            - Avoids complex temporal joins
            - Produces regular time series suitable for time-series models
            - Simplifies feature engineering on synchronized data

        Args:
            freq: Resampling frequency (pandas offset alias):
                  '1S' = 1 second, '100ms' = 100 milliseconds, '1T' = 1 minute
            method: Fill method for missing values:
                    'ffill' = forward fill (use last known value)
                    'bfill' = backward fill
                    'interpolate' = linear interpolation

        Returns:
            DataFrame with unified timeline at specified frequency

        Example:
            >>> # Resample all data to 1-second intervals
            >>> unified = sync.create_unified_timeline(freq='1S', method='ffill')
        """
        if self.trades is None and self.quotes is None and self.lob is None:
            raise ValueError("At least one data source must be provided")

        logger.info(f"Creating unified timeline at {freq} frequency")

        # Determine time range
        min_times = []
        max_times = []

        for df in [self.trades, self.quotes, self.lob]:
            if df is not None:
                min_times.append(df['timestamp'].min())
                max_times.append(df['timestamp'].max())

        time_range = pd.date_range(
            start=min(min_times),
            end=max(max_times),
            freq=freq
        )

        unified = pd.DataFrame({'timestamp': time_range})
        unified = unified.set_index('timestamp')

        # Merge each data source
        for name, df in [('trades', self.trades), ('quotes', self.quotes), ('lob', self.lob)]:
            if df is not None:
                df_indexed = df.set_index('timestamp')

                # Aggregate trades within each interval (count, sum volume)
                if name == 'trades':
                    resampled = df_indexed.resample(freq).agg({
                        'price': 'mean',
                        'size': 'sum',
                    })
                    resampled = resampled.rename(columns={
                        'price': 'avg_trade_price',
                        'size': 'total_trade_volume'
                    })
                else:
                    # For quotes and LOB, use last value in interval
                    resampled = df_indexed.resample(freq).last()

                # Merge with unified timeline
                unified = unified.join(resampled, how='left', rsuffix=f'_{name}')

        # Fill missing values
        if method == 'ffill':
            unified = unified.fillna(method='ffill')
        elif method == 'bfill':
            unified = unified.fillna(method='bfill')
        elif method == 'interpolate':
            numeric_cols = unified.select_dtypes(include=[np.number]).columns
            unified[numeric_cols] = unified[numeric_cols].interpolate(method='time')

        unified = unified.reset_index()

        logger.info(f"Created unified timeline with {len(unified):,} intervals")
        return unified

    def compute_trade_impact(
        self,
        aligned_data: pd.DataFrame,
        horizons: list = [1, 5, 10]
    ) -> pd.DataFrame:
        """
        Compute trade price impact at multiple time horizons.

        Price impact measures how much the mid-price moves after a trade,
        indicating the trade's effect on the market.

        Args:
            aligned_data: DataFrame with aligned trades and quotes (from align_trades_with_quotes)
            horizons: List of time horizons (in ticks) to measure impact

        Returns:
            DataFrame with impact columns: impact_1tick, impact_5tick, impact_10tick

        Formula:
            impact_N = (mid_price[t+N] - mid_price[t]) * trade_side[t]

        Where trade_side = +1 for buys, -1 for sells
        """
        if 'mid_price' not in aligned_data.columns:
            raise ValueError("aligned_data must have 'mid_price' column from quotes")

        aligned_data = aligned_data.copy()

        for horizon in horizons:
            # Shift mid_price forward by horizon
            future_mid = aligned_data['mid_price'].shift(-horizon)

            # Compute impact (signed by trade direction)
            if 'trade_side' in aligned_data.columns:
                aligned_data[f'impact_{horizon}tick'] = (
                    (future_mid - aligned_data['mid_price']) * aligned_data['trade_side']
                )
            else:
                aligned_data[f'impact_{horizon}tick'] = future_mid - aligned_data['mid_price']

        logger.info(f"Computed price impact at {len(horizons)} horizons")
        return aligned_data

    def filter_by_time_gap(
        self,
        aligned_data: pd.DataFrame,
        max_gap: str = '100ms',
        time_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Filter aligned data to remove records with large time gaps between sources.

        Large gaps indicate stale data or misalignment and should be excluded.

        Args:
            aligned_data: Aligned DataFrame with timestamp column
            max_gap: Maximum allowed time gap (pandas offset string)
            time_col: Name of timestamp column

        Returns:
            Filtered DataFrame
        """
        max_gap_td = pd.Timedelta(max_gap)

        # Compute time gap between consecutive records
        aligned_data = aligned_data.copy()
        aligned_data['time_gap'] = aligned_data[time_col].diff()

        # Filter out large gaps
        valid_mask = (aligned_data['time_gap'] <= max_gap_td) | (aligned_data['time_gap'].isna())
        filtered = aligned_data[valid_mask].drop(columns=['time_gap'])

        removed = len(aligned_data) - len(filtered)
        logger.info(
            f"Filtered {removed:,} records with time gaps > {max_gap} "
            f"({len(filtered)/len(aligned_data)*100:.2f}% retained)"
        )

        return filtered


def align_trades_quotes(
    trades: pd.DataFrame,
    quotes: pd.DataFrame,
    method: str = 'asof',
    tolerance: str = '10ms'
) -> pd.DataFrame:
    """
    Convenience function to align trades with quotes.

    Args:
        trades: Trade DataFrame with 'timestamp' column
        quotes: Quote DataFrame with 'timestamp' column
        method: Alignment method ('asof' or 'nearest')
        tolerance: Maximum time difference for matching

    Returns:
        Aligned DataFrame

    Example:
        >>> aligned = align_trades_quotes(trades, quotes, method='asof')
        >>> print(aligned[['timestamp', 'price', 'bid_price', 'ask_price']].head())
    """
    sync = DataSynchronizer(trades=trades, quotes=quotes, tolerance=tolerance)
    return sync.align_trades_with_quotes(method=method)


def align_trades_lob(
    trades: pd.DataFrame,
    lob: pd.DataFrame,
    method: str = 'asof',
    tolerance: str = '10ms'
) -> pd.DataFrame:
    """
    Convenience function to align trades with LOB snapshots.

    Args:
        trades: Trade DataFrame with 'timestamp' column
        lob: LOB DataFrame with 'timestamp' column
        method: Alignment method ('asof' or 'nearest')
        tolerance: Maximum time difference for matching

    Returns:
        Aligned DataFrame
    """
    sync = DataSynchronizer(trades=trades, lob=lob, tolerance=tolerance)
    return sync.align_trades_with_lob(method=method)


def create_unified_dataset(
    trades: Optional[pd.DataFrame] = None,
    quotes: Optional[pd.DataFrame] = None,
    lob: Optional[pd.DataFrame] = None,
    freq: str = '1S',
    method: str = 'ffill'
) -> pd.DataFrame:
    """
    Convenience function to create unified timeline from all data sources.

    Args:
        trades: Trade DataFrame
        quotes: Quote DataFrame
        lob: LOB DataFrame
        freq: Resampling frequency
        method: Fill method for missing values

    Returns:
        Unified DataFrame at specified frequency

    Example:
        >>> unified = create_unified_dataset(
        ...     trades=trades,
        ...     quotes=quotes,
        ...     freq='1S',
        ...     method='ffill'
        ... )
    """
    sync = DataSynchronizer(trades=trades, quotes=quotes, lob=lob)
    return sync.create_unified_timeline(freq=freq, method=method)
