"""
Polygon.io L2 Order Book Snapshot Loader

This module provides functionality for loading and processing Level 2 (L2) order book snapshots
from Polygon.io's Stocks Developer API for U.S. equity microstructure analysis.

Polygon.io's L2 snapshots provide aggregated order book depth at multiple price levels,
though typically less granular than full LOB data from LOBSTER or exchange direct feeds.

Author: Rylan Spence
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Union, Dict, Tuple
from datetime import datetime
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolygonLOBLoader:
    """
    Load and process Level 2 order book snapshots from Polygon.io for U.S. equity microstructure analysis.

    Polygon.io provides L2 aggregated snapshots via:
        - REST API: /v3/snapshot/locale/us/markets/stocks/tickers/{ticker}
        - WebSocket: Aggregated book updates

    Note: Polygon.io L2 data typically provides 5-20 levels of depth (bid/ask),
    which is less granular than full exchange feeds but sufficient for many microstructure analyses.

    Attributes:
        data_dir (Path): Directory containing Parquet snapshot files
        api_key (str): Polygon.io API key for direct fetching
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOG')
        depth (int): Number of order book levels to load (default: 10)

    Example:
        >>> loader = PolygonLOBLoader(data_dir='data/raw/polygon_snapshots', ticker='AAPL')
        >>> lob = loader.load_and_parse(depth=10)
        >>> print(f"Loaded {len(lob)} order book snapshots with {loader.depth} levels")
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/raw/polygon_snapshots",
        api_key: Optional[str] = None,
        ticker: str = "AAPL",
        depth: int = 10
    ):
        """
        Initialize the Polygon.io L2 snapshot loader.

        Args:
            data_dir: Directory containing Parquet snapshot files
            api_key: Polygon.io API key (required for API fetching)
            ticker: Stock ticker symbol to load
            depth: Number of order book levels to process (1-20, depending on data availability)
        """
        self.data_dir = Path(data_dir)
        self.api_key = api_key
        self.ticker = ticker.upper()
        self.depth = depth
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_files(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        pattern: str = "*.parquet"
    ) -> pd.DataFrame:
        """
        Load L2 order book snapshots from local Parquet files.

        Expected Parquet schema:
            - timestamp (datetime64[ns])
            - bid_price_0, bid_price_1, ..., bid_price_{N-1} (float64)
            - bid_size_0, bid_size_1, ..., bid_size_{N-1} (int64)
            - ask_price_0, ask_price_1, ..., ask_price_{N-1} (float64)
            - ask_size_0, ask_size_1, ..., ask_size_{N-1} (int64)

        Note: Uses 0-indexed levels (bid_price_0 = best bid)

        Args:
            start_date: Filter snapshots after this date (ISO format: 'YYYY-MM-DD')
            end_date: Filter snapshots before this date (ISO format: 'YYYY-MM-DD')
            pattern: Glob pattern to match files (default: '*.parquet')

        Returns:
            DataFrame with timestamp and LOB levels (0-indexed)

        Raises:
            FileNotFoundError: If no matching files found
        """
        files = list(self.data_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(f"No files matching pattern '{pattern}' in {self.data_dir}")

        logger.info(f"Loading {len(files)} Parquet files from {self.data_dir}")

        dfs = []
        for file_path in sorted(files):
            try:
                df = pd.read_parquet(file_path)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        if not dfs:
            raise ValueError("No data successfully loaded from Parquet files")

        lob = pd.concat(dfs, ignore_index=True)

        # Ensure timestamp column is datetime
        if 'timestamp' in lob.columns:
            lob['timestamp'] = pd.to_datetime(lob['timestamp'])
            lob = lob.sort_values('timestamp').reset_index(drop=True)

        # Apply date filters
        if start_date:
            lob = lob[lob['timestamp'] >= pd.Timestamp(start_date)]
        if end_date:
            lob = lob[lob['timestamp'] <= pd.Timestamp(end_date)]

        logger.info(f"Loaded {len(lob):,} L2 snapshots for {self.ticker}")
        return lob

    def fetch_snapshot_from_api(self) -> pd.DataFrame:
        """
        Fetch current L2 order book snapshot from Polygon.io API (single snapshot, not historical).

        Uses the /v3/snapshot/locale/us/markets/stocks/tickers/{ticker} endpoint.

        Note: This endpoint provides a single current snapshot, not historical snapshots.
        For historical L2 data, you need a higher-tier Polygon.io subscription with
        access to historical aggregated book data or WebSocket recordings.

        Returns:
            DataFrame with single-row snapshot

        Raises:
            ValueError: If API key not provided
            requests.HTTPError: If API request fails

        API Response Format (simplified):
            {
                "status": "OK",
                "ticker": {
                    "ticker": "AAPL",
                    "updated": 1609459200000000000,
                    "quote": {
                        "bid": 132.68,
                        "ask": 132.70,
                        "bidSize": 300,
                        "askSize": 200,
                        ...
                    },
                    "lastTrade": {...},
                    ...
                }
            }
        """
        if not self.api_key:
            raise ValueError("API key required for fetching from Polygon.io")

        url = f"https://api.polygon.io/v3/snapshot/locale/us/markets/stocks/tickers/{self.ticker}"
        params = {"apiKey": self.api_key}

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "OK":
                raise ValueError(f"API returned status: {data.get('status')}")

            ticker_data = data.get("ticker", {})
            quote = ticker_data.get("quote", {})

            # Extract top-of-book (Level 1) data
            snapshot = {
                'timestamp': pd.to_datetime(ticker_data.get('updated', 0), unit='ns'),
                'bid_price_0': float(quote.get('bid', 0)),
                'bid_size_0': int(quote.get('bidSize', 0)),
                'ask_price_0': float(quote.get('ask', 0)),
                'ask_size_0': int(quote.get('askSize', 0)),
            }

            logger.info(f"Fetched current snapshot for {self.ticker}")
            return pd.DataFrame([snapshot])

        except requests.HTTPError as e:
            logger.error(f"HTTP error fetching snapshot: {e}")
            raise

    def standardize_column_names(
        self,
        lob: pd.DataFrame,
        from_1_indexed: bool = False
    ) -> pd.DataFrame:
        """
        Standardize column names to 0-indexed format.

        Some data sources use 1-indexed levels (bid_price_1 = best bid),
        while this loader uses 0-indexed (bid_price_0 = best bid).

        Args:
            lob: DataFrame with LOB columns
            from_1_indexed: If True, convert from 1-indexed to 0-indexed

        Returns:
            DataFrame with standardized 0-indexed column names
        """
        if not from_1_indexed:
            return lob

        lob = lob.copy()
        rename_map = {}

        # Find all bid/ask price and size columns
        for col in lob.columns:
            if col.startswith(('bid_price_', 'bid_size_', 'ask_price_', 'ask_size_')):
                parts = col.split('_')
                if len(parts) == 3 and parts[2].isdigit():
                    old_index = int(parts[2])
                    new_index = old_index - 1  # Convert to 0-indexed
                    new_col = f"{parts[0]}_{parts[1]}_{new_index}"
                    rename_map[col] = new_col

        lob = lob.rename(columns=rename_map)
        logger.info(f"Standardized {len(rename_map)} columns from 1-indexed to 0-indexed")
        return lob

    def parse_lob_levels(
        self,
        lob: pd.DataFrame,
        max_depth: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Parse and validate LOB levels, ensuring data integrity.

        Validates:
            - Bid prices are strictly decreasing (bid_0 > bid_1 > bid_2 > ...)
            - Ask prices are strictly increasing (ask_0 < ask_1 < ask_2 < ...)
            - All sizes are non-negative
            - No missing prices at inner levels when outer levels exist

        Args:
            lob: DataFrame with bid_price_N, bid_size_N, ask_price_N, ask_size_N columns
            max_depth: Maximum depth to validate (default: self.depth)

        Returns:
            DataFrame with validated LOB levels
        """
        if max_depth is None:
            max_depth = self.depth

        lob = lob.copy()

        # Validate bid prices are decreasing
        for i in range(max_depth - 1):
            bid_col = f'bid_price_{i}'
            next_bid_col = f'bid_price_{i+1}'

            if bid_col in lob.columns and next_bid_col in lob.columns:
                # Allow NaN for missing levels
                invalid = (
                    lob[bid_col].notna() & lob[next_bid_col].notna() &
                    (lob[bid_col] <= lob[next_bid_col])
                )
                if invalid.any():
                    logger.warning(
                        f"Found {invalid.sum()} snapshots with invalid bid ordering at level {i}"
                    )
                    # Set invalid entries to NaN
                    lob.loc[invalid, next_bid_col] = np.nan
                    lob.loc[invalid, f'bid_size_{i+1}'] = np.nan

        # Validate ask prices are increasing
        for i in range(max_depth - 1):
            ask_col = f'ask_price_{i}'
            next_ask_col = f'ask_price_{i+1}'

            if ask_col in lob.columns and next_ask_col in lob.columns:
                invalid = (
                    lob[ask_col].notna() & lob[next_ask_col].notna() &
                    (lob[ask_col] >= lob[next_ask_col])
                )
                if invalid.any():
                    logger.warning(
                        f"Found {invalid.sum()} snapshots with invalid ask ordering at level {i}"
                    )
                    lob.loc[invalid, next_ask_col] = np.nan
                    lob.loc[invalid, f'ask_size_{i+1}'] = np.nan

        logger.info(f"Validated {max_depth} LOB levels for {len(lob):,} snapshots")
        return lob

    def compute_lob_features(self, lob: pd.DataFrame) -> pd.DataFrame:
        """
        Compute basic LOB-derived microstructure features.

        Features:
            - mid_price: (best_bid + best_ask) / 2
            - quoted_spread: best_ask - best_bid
            - relative_spread: spread / mid_price
            - depth_at_best: bid_size_0 + ask_size_0
            - depth_imbalance: (bid_size_0 - ask_size_0) / (bid_size_0 + ask_size_0)
            - cumulative_depth_{N}: sum of depth across N levels

        Args:
            lob: DataFrame with LOB columns

        Returns:
            DataFrame with additional feature columns
        """
        lob = lob.copy()

        # Basic price features
        lob['mid_price'] = (lob['bid_price_0'] + lob['ask_price_0']) / 2.0
        lob['quoted_spread'] = lob['ask_price_0'] - lob['bid_price_0']
        lob['relative_spread'] = lob['quoted_spread'] / lob['mid_price']

        # Top-of-book depth
        lob['depth_at_best'] = lob['bid_size_0'] + lob['ask_size_0']
        total_best = lob['bid_size_0'] + lob['ask_size_0']
        lob['depth_imbalance'] = (lob['bid_size_0'] - lob['ask_size_0']) / total_best.replace(0, np.nan)

        # Cumulative depth across levels
        for n in [5, 10]:
            if n <= self.depth:
                bid_depth = sum([lob.get(f'bid_size_{i}', 0) for i in range(n)])
                ask_depth = sum([lob.get(f'ask_size_{i}', 0) for i in range(n)])
                lob[f'cumulative_bid_depth_{n}'] = bid_depth
                lob[f'cumulative_ask_depth_{n}'] = ask_depth
                lob[f'cumulative_depth_{n}'] = bid_depth + ask_depth

        logger.info(f"Computed LOB features for {len(lob):,} snapshots")
        return lob

    def compute_book_slope(
        self,
        lob: pd.DataFrame,
        side: str = 'bid',
        max_levels: Optional[int] = None
    ) -> pd.Series:
        """
        Compute order book slope (volume decay rate away from best price).

        Fits an exponential decay model: Q(p) = Q_0 * exp(-lambda * |p - p_best|)
        where lambda is the slope parameter (higher = faster decay).

        Args:
            lob: DataFrame with LOB levels
            side: 'bid' or 'ask'
            max_levels: Number of levels to use for slope calculation (default: self.depth)

        Returns:
            Series with slope values (lambda parameter)
        """
        if max_levels is None:
            max_levels = min(self.depth, 5)  # Use top 5 levels for stability

        slopes = []

        for idx, row in lob.iterrows():
            prices = []
            sizes = []

            best_price = row.get(f'{side}_price_0', np.nan)
            if pd.isna(best_price):
                slopes.append(np.nan)
                continue

            for i in range(max_levels):
                price = row.get(f'{side}_price_{i}', np.nan)
                size = row.get(f'{side}_size_{i}', np.nan)

                if pd.notna(price) and pd.notna(size) and size > 0:
                    prices.append(price)
                    sizes.append(size)

            if len(prices) < 2:
                slopes.append(np.nan)
                continue

            # Compute price distances from best
            price_distances = np.abs(np.array(prices) - best_price)

            # Log-linear regression: log(size) ~ -lambda * distance
            try:
                log_sizes = np.log(np.array(sizes) + 1)  # Add 1 to avoid log(0)
                slope, _ = np.polyfit(price_distances, log_sizes, 1)
                slopes.append(-slope)  # Negate to get decay rate
            except:
                slopes.append(np.nan)

        return pd.Series(slopes, index=lob.index, name=f'{side}_slope')

    def load_and_parse(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        compute_features: bool = True,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Main entry point: load, validate, and process L2 order book snapshots.

        Args:
            start_date: Filter snapshots after this date (ISO format)
            end_date: Filter snapshots before this date (ISO format)
            compute_features: If True, compute LOB-derived features
            validate: If True, validate LOB level ordering

        Returns:
            Fully processed DataFrame ready for microstructure analysis

        Example:
            >>> loader = PolygonLOBLoader(ticker='AAPL', depth=10)
            >>> lob = loader.load_and_parse(
            ...     start_date='2024-01-01',
            ...     compute_features=True,
            ...     validate=True
            ... )
        """
        # Step 1: Load LOB data
        lob = self.load_files(start_date=start_date, end_date=end_date)

        # Step 2: Validate LOB levels
        if validate:
            lob = self.parse_lob_levels(lob, max_depth=self.depth)

        # Step 3: Compute LOB features
        if compute_features:
            lob = self.compute_lob_features(lob)

        logger.info(f"Final dataset: {len(lob):,} L2 snapshots for {self.ticker}")
        return lob

    def save_to_parquet(
        self,
        lob: pd.DataFrame,
        filename: Optional[str] = None
    ) -> Path:
        """
        Save processed LOB snapshots to Parquet format.

        Args:
            lob: DataFrame to save
            filename: Output filename (default: {ticker}_lob_{date}.parquet)

        Returns:
            Path to saved file
        """
        if filename is None:
            date_str = lob['timestamp'].min().strftime('%Y%m%d')
            filename = f"{self.ticker}_lob_{date_str}.parquet"

        output_path = self.data_dir / filename
        lob.to_parquet(output_path, index=False, compression='snappy')

        logger.info(f"Saved {len(lob):,} LOB snapshots to {output_path}")
        return output_path


def load_polygon_lob(
    ticker: str = "AAPL",
    data_dir: Union[str, Path] = "data/raw/polygon_snapshots",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    depth: int = 10,
    compute_features: bool = True,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to load Polygon.io L2 order book snapshots.

    Args:
        ticker: Stock ticker symbol
        data_dir: Directory containing LOB Parquet files
        start_date: Filter start date (ISO format)
        end_date: Filter end date (ISO format)
        depth: Number of LOB levels to process
        compute_features: Compute LOB-derived features
        api_key: Polygon.io API key (optional, for direct fetching)

    Returns:
        DataFrame with processed LOB snapshot data

    Example:
        >>> lob = load_polygon_lob('AAPL', start_date='2024-01-01', depth=10)
        >>> print(f"Average spread: {lob['quoted_spread'].mean():.4f}")
    """
    loader = PolygonLOBLoader(
        data_dir=data_dir,
        api_key=api_key,
        ticker=ticker,
        depth=depth
    )
    return loader.load_and_parse(
        start_date=start_date,
        end_date=end_date,
        compute_features=compute_features
    )
