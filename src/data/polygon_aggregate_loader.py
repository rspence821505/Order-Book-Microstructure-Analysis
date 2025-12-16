"""
Polygon.io Aggregate (OHLCV) Data Loader

This module provides functionality for loading historical minute/second bars
from Polygon.io's Stocks Developer API for U.S. equities.

Author: Rylan Spence
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union
from datetime import datetime
import requests
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolygonAggregateLoader:
    """
    Load and process OHLCV aggregate bars from Polygon.io.

    Supports minute, second, hour, and day bars with full historical depth.
    Use aggregates to approximate NBBO features when tick quotes aren't available.

    Attributes:
        data_dir (Path): Directory containing Parquet aggregate files
        api_key (str): Polygon.io API key for direct fetching
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOG')

    Example:
        >>> loader = PolygonAggregateLoader(ticker='AAPL')
        >>> bars = loader.fetch_from_api(date='2024-12-06', timespan='minute')
        >>> print(f"Loaded {len(bars)} minute bars")
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/raw/polygon_aggregates",
        api_key: Optional[str] = None,
        ticker: str = "AAPL"
    ):
        """
        Initialize the Polygon.io aggregate loader.

        Args:
            data_dir: Directory for aggregate files
            api_key: Polygon.io API key (from env or param)
            ticker: Stock ticker symbol to load
        """
        self.data_dir = Path(data_dir)
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        self.ticker = ticker.upper()
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_from_api(
        self,
        date: str,
        timespan: str = 'minute',
        multiplier: int = 1
    ) -> pd.DataFrame:
        """
        Fetch aggregate bars from Polygon.io API.

        Uses /v2/aggs/ticker/{ticker}/range endpoint for historical bars.

        Args:
            date: Date to fetch (ISO format: 'YYYY-MM-DD')
            timespan: Bar timespan - 'second', 'minute', 'hour', 'day'
            multiplier: Multiplier for timespan (e.g., 5 for 5-minute bars)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, vwap, trades

        Raises:
            ValueError: If API key not provided
            requests.HTTPError: If API request fails

        API Response Format:
            {
                "results": [
                    {
                        "t": 1609459200000,  # milliseconds
                        "o": 132.69,
                        "h": 132.75,
                        "l": 132.68,
                        "c": 132.72,
                        "v": 1000,
                        "vw": 132.71,
                        "n": 10  # number of trades
                    },
                    ...
                ],
                "status": "OK"
            }
        """
        if not self.api_key:
            raise ValueError("API key required for fetching from Polygon.io")

        # Endpoint: /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
        base_url = (
            f"https://api.polygon.io/v2/aggs/ticker/{self.ticker}/range/"
            f"{multiplier}/{timespan}/{date}/{date}"
        )

        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "OK":
                logger.error(f"API error: {data.get('status')}")
                return pd.DataFrame()

            results = data.get("results", [])
            if not results:
                logger.warning(f"No aggregates fetched for {self.ticker} on {date}")
                return pd.DataFrame()

            # Convert to DataFrame
            bars_df = pd.DataFrame(results)
            bars_df = self._standardize_response(bars_df)

            logger.info(
                f"Fetched {len(bars_df):,} {timespan} bars for {self.ticker} on {date}"
            )

            return bars_df

        except requests.HTTPError as e:
            logger.error(f"HTTP error fetching aggregates: {e}")
            raise

    def _standardize_response(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert API response to standardized DataFrame format.

        Args:
            df: Raw DataFrame from API

        Returns:
            Standardized DataFrame with consistent column names
        """
        # Rename columns to standard names
        column_mapping = {
            't': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vw': 'vwap',
            'n': 'trades'
        }

        df = df.rename(columns=column_mapping)

        # Convert timestamp from milliseconds to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        # Ensure numeric types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'trades']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    def compute_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute spread and mid-price approximations from OHLC bars.

        Approximations:
            - Spread ≈ High - Low
            - Mid-price ≈ (High + Low) / 2
            - Relative spread ≈ (High - Low) / Mid-price

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with added columns: estimated_spread, mid_price, relative_spread
        """
        df = df.copy()

        # Estimated spread from high-low range
        df['estimated_spread'] = df['high'] - df['low']

        # Mid-price approximation
        df['mid_price'] = (df['high'] + df['low']) / 2

        # Relative spread (basis points)
        df['relative_spread'] = df['estimated_spread'] / df['mid_price']

        # Typical price (weighted average)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        logger.info("Computed spread features from OHLC bars")

        return df

    def load_files(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        pattern: str = "*.parquet"
    ) -> pd.DataFrame:
        """
        Load aggregate data from local Parquet files.

        Args:
            start_date: Filter after this date (ISO format: 'YYYY-MM-DD')
            end_date: Filter before this date (ISO format: 'YYYY-MM-DD')
            pattern: Glob pattern to match files

        Returns:
            DataFrame with aggregate bars

        Raises:
            FileNotFoundError: If no matching files found
        """
        files = list(self.data_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(
                f"No files matching pattern '{pattern}' in {self.data_dir}"
            )

        logger.info(f"Loading {len(files)} Parquet files from {self.data_dir}")

        dfs = []
        for file_path in sorted(files):
            try:
                df = pd.read_parquet(file_path)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue

        if not dfs:
            raise ValueError("No valid Parquet files loaded")

        # Concatenate all files
        aggregates = pd.concat(dfs, ignore_index=True)

        # Ensure timestamp is datetime
        if 'timestamp' in aggregates.columns:
            aggregates['timestamp'] = pd.to_datetime(aggregates['timestamp'], utc=True)
            aggregates = aggregates.sort_values('timestamp').reset_index(drop=True)

        # Filter by date range
        if start_date:
            start_dt = pd.to_datetime(start_date, utc=True)
            aggregates = aggregates[aggregates['timestamp'] >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(end_date, utc=True)
            aggregates = aggregates[aggregates['timestamp'] <= end_dt]

        logger.info(
            f"Loaded {len(aggregates):,} aggregates for {self.ticker} "
            f"({aggregates['timestamp'].min()} to {aggregates['timestamp'].max()})"
        )

        return aggregates


def load_polygon_aggregates(
    ticker: str = "AAPL",
    data_dir: Union[str, Path] = "data/raw/polygon_aggregates",
    date: Optional[str] = None,
    timespan: str = 'minute',
    multiplier: int = 1,
    compute_features: bool = True,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to load Polygon.io aggregate data.

    Args:
        ticker: Stock ticker symbol
        data_dir: Directory containing aggregate Parquet files
        date: Date to fetch (ISO format: 'YYYY-MM-DD')
        timespan: Bar timespan - 'second', 'minute', 'hour', 'day'
        multiplier: Multiplier for timespan
        compute_features: Compute spread approximations from OHLC
        api_key: Polygon.io API key (optional)

    Returns:
        DataFrame with processed aggregate data

    Example:
        >>> bars = load_polygon_aggregates('AAPL', date='2024-12-06', timespan='minute')
        >>> print(bars[['timestamp', 'open', 'high', 'low', 'close', 'estimated_spread']])
    """
    loader = PolygonAggregateLoader(
        data_dir=data_dir,
        api_key=api_key,
        ticker=ticker
    )

    # Fetch from API if date provided
    if date:
        aggregates = loader.fetch_from_api(
            date=date,
            timespan=timespan,
            multiplier=multiplier
        )
    else:
        # Load from local files
        aggregates = loader.load_files()

    # Compute spread features if requested
    if compute_features and len(aggregates) > 0:
        aggregates = loader.compute_spread_features(aggregates)

    return aggregates
