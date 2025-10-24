"""
Polygon.io NBBO Quote Data Loader

This module provides functionality for loading and processing National Best Bid and Offer (NBBO)
quote data from Polygon.io's Stocks Developer API for U.S. equity microstructure analysis.

NBBO represents the best (highest bid, lowest ask) prices across all U.S. exchanges,
consolidated by the Securities Information Processor (SIP).

Author: Rylan Spence
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Union, Dict
from datetime import datetime
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolygonQuoteLoader:
    """
    Load and process NBBO quote data from Polygon.io for U.S. equity microstructure analysis.

    Supports both historical data loading from Parquet files and direct API fetching
    from Polygon.io's /v3/quotes/{ticker} endpoint.

    NBBO quote data includes:
        - Timestamp (nanosecond precision)
        - Bid price (national best bid)
        - Bid size (shares available at best bid)
        - Ask price (national best ask)
        - Ask size (shares available at best ask)
        - Bid exchange (exchange offering best bid)
        - Ask exchange (exchange offering best ask)

    Attributes:
        data_dir (Path): Directory containing Parquet quote files
        api_key (str): Polygon.io API key for direct fetching
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOG')

    Example:
        >>> loader = PolygonQuoteLoader(data_dir='data/raw/polygon_quotes', ticker='AAPL')
        >>> quotes = loader.load_and_parse()
        >>> print(f"Loaded {len(quotes)} NBBO updates")
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/raw/polygon_quotes",
        api_key: Optional[str] = None,
        ticker: str = "AAPL"
    ):
        """
        Initialize the Polygon.io NBBO quote loader.

        Args:
            data_dir: Directory containing Parquet quote files or where to save fetched data
            api_key: Polygon.io API key (required for API fetching)
            ticker: Stock ticker symbol to load
        """
        self.data_dir = Path(data_dir)
        self.api_key = api_key
        self.ticker = ticker.upper()
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_files(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        pattern: str = "*.parquet"
    ) -> pd.DataFrame:
        """
        Load NBBO quote data from local Parquet files.

        Args:
            start_date: Filter quotes after this date (ISO format: 'YYYY-MM-DD')
            end_date: Filter quotes before this date (ISO format: 'YYYY-MM-DD')
            pattern: Glob pattern to match files (default: '*.parquet')

        Returns:
            DataFrame with columns: timestamp, bid_price, bid_size, ask_price, ask_size,
                                   bid_exchange, ask_exchange

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

        quotes = pd.concat(dfs, ignore_index=True)

        # Ensure timestamp column is datetime
        if 'timestamp' in quotes.columns:
            quotes['timestamp'] = pd.to_datetime(quotes['timestamp'])
            quotes = quotes.sort_values('timestamp').reset_index(drop=True)

        # Apply date filters
        if start_date:
            quotes = quotes[quotes['timestamp'] >= pd.Timestamp(start_date)]
        if end_date:
            quotes = quotes[quotes['timestamp'] <= pd.Timestamp(end_date)]

        logger.info(f"Loaded {len(quotes):,} NBBO quotes for {self.ticker}")
        return quotes

    def fetch_from_api(
        self,
        date: str,
        limit: int = 50000
    ) -> pd.DataFrame:
        """
        Fetch NBBO quote data directly from Polygon.io API.

        Uses the /v3/quotes/{ticker} endpoint to retrieve historical NBBO updates.

        Args:
            date: Date to fetch (ISO format: 'YYYY-MM-DD')
            limit: Maximum number of quotes per request (default: 50000, max: 50000)

        Returns:
            DataFrame with standardized NBBO columns

        Raises:
            ValueError: If API key not provided
            requests.HTTPError: If API request fails

        API Response Format:
            {
                "results": [
                    {
                        "sip_timestamp": 1609459200000000000,  # nanoseconds
                        "participant_timestamp": 1609459200000000000,
                        "ask_price": 132.70,
                        "ask_size": 200,
                        "ask_exchange": 4,  # Exchange ID
                        "bid_price": 132.68,
                        "bid_size": 300,
                        "bid_exchange": 11,
                        "conditions": [1],
                        "indicators": [0],
                        "tape": 3
                    },
                    ...
                ],
                "status": "OK",
                "next_url": "https://..."
            }
        """
        if not self.api_key:
            raise ValueError("API key required for fetching from Polygon.io")

        base_url = f"https://api.polygon.io/v3/quotes/{self.ticker}"

        params = {
            "timestamp": date,
            "order": "asc",
            "limit": limit,
            "sort": "timestamp",
            "apiKey": self.api_key
        }

        all_quotes = []
        next_url = None
        page = 1

        while True:
            try:
                if next_url:
                    response = requests.get(next_url)
                else:
                    response = requests.get(base_url, params=params)

                response.raise_for_status()
                data = response.json()

                if data.get("status") != "OK":
                    logger.error(f"API error: {data.get('status')}")
                    break

                results = data.get("results", [])
                if not results:
                    break

                all_quotes.extend(results)
                logger.info(f"Fetched page {page}: {len(results)} quotes")

                # Check for pagination
                next_url = data.get("next_url")
                if not next_url:
                    break

                # Add API key to next_url
                next_url = f"{next_url}&apiKey={self.api_key}"
                page += 1

            except requests.HTTPError as e:
                logger.error(f"HTTP error fetching quotes: {e}")
                break

        if not all_quotes:
            logger.warning(f"No quotes fetched for {self.ticker} on {date}")
            return pd.DataFrame()

        # Convert to DataFrame with standardized columns
        quotes_df = pd.DataFrame(all_quotes)
        quotes_df = self._standardize_api_response(quotes_df)

        logger.info(f"Fetched {len(quotes_df):,} total NBBO quotes for {date}")
        return quotes_df

    def _standardize_api_response(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize Polygon.io API response to internal NBBO format.

        Args:
            df: Raw DataFrame from API response

        Returns:
            Standardized DataFrame with columns:
                - timestamp (datetime64[ns])
                - bid_price (float64)
                - bid_size (int64)
                - ask_price (float64)
                - ask_size (int64)
                - bid_exchange (int)
                - ask_exchange (int)
                - conditions (list, optional)
        """
        standardized = pd.DataFrame()

        # Use SIP timestamp (consolidated feed timestamp) as primary
        if 'sip_timestamp' in df.columns:
            standardized['timestamp'] = pd.to_datetime(df['sip_timestamp'], unit='ns')
        elif 'participant_timestamp' in df.columns:
            standardized['timestamp'] = pd.to_datetime(df['participant_timestamp'], unit='ns')
        else:
            raise ValueError("No timestamp column found in API response")

        # NBBO bid side
        standardized['bid_price'] = df['bid_price'].astype(float)
        standardized['bid_size'] = df['bid_size'].astype(int)
        standardized['bid_exchange'] = df.get('bid_exchange', 0).astype(int)

        # NBBO ask side
        standardized['ask_price'] = df['ask_price'].astype(float)
        standardized['ask_size'] = df['ask_size'].astype(int)
        standardized['ask_exchange'] = df.get('ask_exchange', 0).astype(int)

        # Optional: conditions and indicators
        if 'conditions' in df.columns:
            standardized['conditions'] = df['conditions']

        return standardized.sort_values('timestamp').reset_index(drop=True)

    def filter_valid_quotes(self, quotes: pd.DataFrame) -> pd.DataFrame:
        """
        Filter for valid NBBO quotes, removing locked/crossed markets and stale quotes.

        Locked market: bid_price == ask_price (spread = 0)
        Crossed market: bid_price > ask_price (spread < 0)

        Both conditions indicate data quality issues or special market conditions
        and should be excluded from microstructure analysis.

        Args:
            quotes: DataFrame with bid_price and ask_price columns

        Returns:
            Filtered DataFrame with valid quotes only
        """
        original_len = len(quotes)

        # Remove quotes with zero or negative spreads
        valid_quotes = quotes[quotes['ask_price'] > quotes['bid_price']].copy()

        # Remove quotes with zero bid or ask prices
        valid_quotes = valid_quotes[
            (valid_quotes['bid_price'] > 0) &
            (valid_quotes['ask_price'] > 0)
        ]

        # Remove quotes with zero sizes (no liquidity)
        valid_quotes = valid_quotes[
            (valid_quotes['bid_size'] > 0) &
            (valid_quotes['ask_size'] > 0)
        ]

        removed = original_len - len(valid_quotes)
        logger.info(
            f"Filtered {original_len:,} quotes to {len(valid_quotes):,} valid quotes "
            f"({removed:,} removed, {len(valid_quotes)/original_len*100:.2f}% retained)"
        )

        return valid_quotes

    def compute_nbbo_features(self, quotes: pd.DataFrame) -> pd.DataFrame:
        """
        Compute basic NBBO-derived microstructure features.

        Features:
            - mid_price: Simple mid-price = (bid + ask) / 2
            - quoted_spread: Absolute spread = ask - bid
            - relative_spread: Spread / mid_price (percentage)
            - spread_bps: Spread in basis points (10,000 * relative_spread)
            - weighted_mid_price: Size-weighted mid = (bid*ask_size + ask*bid_size) / (bid_size + ask_size)
            - order_flow_imbalance: (bid_size - ask_size) / (bid_size + ask_size)

        Args:
            quotes: DataFrame with bid_price, bid_size, ask_price, ask_size

        Returns:
            DataFrame with additional feature columns
        """
        quotes = quotes.copy()

        # Mid-price (simple average)
        quotes['mid_price'] = (quotes['bid_price'] + quotes['ask_price']) / 2.0

        # Quoted spread
        quotes['quoted_spread'] = quotes['ask_price'] - quotes['bid_price']

        # Relative spread (percentage)
        quotes['relative_spread'] = quotes['quoted_spread'] / quotes['mid_price']

        # Spread in basis points
        quotes['spread_bps'] = quotes['relative_spread'] * 10_000

        # Weighted mid-price (size-weighted)
        total_size = quotes['bid_size'] + quotes['ask_size']
        quotes['weighted_mid_price'] = (
            (quotes['bid_price'] * quotes['ask_size'] +
             quotes['ask_price'] * quotes['bid_size']) / total_size
        )

        # Order flow imbalance (top-of-book)
        quotes['order_flow_imbalance'] = (
            (quotes['bid_size'] - quotes['ask_size']) / total_size
        )

        logger.info(f"Computed NBBO features for {len(quotes):,} quotes")
        return quotes

    def resample_quotes(
        self,
        quotes: pd.DataFrame,
        freq: str = '1S',
        method: str = 'last'
    ) -> pd.DataFrame:
        """
        Resample high-frequency NBBO quotes to lower frequency.

        Useful for reducing data volume and aligning with trade data at specific intervals.

        Args:
            quotes: DataFrame with timestamp index
            freq: Resampling frequency (pandas offset alias):
                  '1S' = 1 second, '100ms' = 100 milliseconds, '1T' = 1 minute
            method: Resampling method:
                    'last' = last quote in interval (most recent)
                    'mean' = average across interval
                    'median' = median across interval

        Returns:
            Resampled DataFrame at specified frequency

        Example:
            >>> # Resample to 1-second intervals using last quote
            >>> quotes_1s = loader.resample_quotes(quotes, freq='1S', method='last')
        """
        if 'timestamp' not in quotes.columns:
            raise ValueError("DataFrame must have 'timestamp' column")

        quotes = quotes.set_index('timestamp')

        if method == 'last':
            resampled = quotes.resample(freq).last()
        elif method == 'mean':
            resampled = quotes.resample(freq).mean()
        elif method == 'median':
            resampled = quotes.resample(freq).median()
        else:
            raise ValueError(f"Unknown resampling method: {method}")

        # Drop NaN rows (intervals with no quotes)
        resampled = resampled.dropna()

        logger.info(
            f"Resampled {len(quotes):,} quotes to {len(resampled):,} at {freq} frequency "
            f"(method: {method})"
        )

        return resampled.reset_index()

    def detect_stale_quotes(
        self,
        quotes: pd.DataFrame,
        max_quote_age: float = 1.0
    ) -> pd.DataFrame:
        """
        Detect and flag stale quotes (quotes that haven't updated in a long time).

        Stale quotes can indicate:
            - Low trading activity
            - Data quality issues
            - Market halts

        Args:
            quotes: DataFrame with timestamp column
            max_quote_age: Maximum time (in seconds) between updates before flagging as stale

        Returns:
            DataFrame with additional 'is_stale' boolean column
        """
        quotes = quotes.copy()
        quotes = quotes.sort_values('timestamp')

        # Compute time since last update
        quotes['time_since_last_update'] = quotes['timestamp'].diff().dt.total_seconds()

        # Flag stale quotes
        quotes['is_stale'] = quotes['time_since_last_update'] > max_quote_age

        stale_count = quotes['is_stale'].sum()
        logger.info(
            f"Detected {stale_count:,} stale quotes "
            f"({stale_count/len(quotes)*100:.2f}%) with age > {max_quote_age}s"
        )

        return quotes

    def load_and_parse(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        filter_valid: bool = True,
        compute_features: bool = True,
        resample_freq: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Main entry point: load, filter, and process NBBO quote data.

        Args:
            start_date: Filter quotes after this date (ISO format)
            end_date: Filter quotes before this date (ISO format)
            filter_valid: If True, remove locked/crossed markets
            compute_features: If True, compute NBBO-derived features
            resample_freq: Optional resampling frequency (e.g., '1S', '100ms')

        Returns:
            Fully processed DataFrame ready for microstructure analysis

        Example:
            >>> loader = PolygonQuoteLoader(ticker='AAPL')
            >>> quotes = loader.load_and_parse(
            ...     start_date='2024-01-01',
            ...     filter_valid=True,
            ...     compute_features=True,
            ...     resample_freq='1S'
            ... )
        """
        # Step 1: Load quote data
        quotes = self.load_files(start_date=start_date, end_date=end_date)

        # Step 2: Filter for valid quotes
        if filter_valid:
            quotes = self.filter_valid_quotes(quotes)

        # Step 3: Compute NBBO features
        if compute_features:
            quotes = self.compute_nbbo_features(quotes)

        # Step 4: Resample if requested
        if resample_freq:
            quotes = self.resample_quotes(quotes, freq=resample_freq, method='last')

        logger.info(f"Final dataset: {len(quotes):,} NBBO quotes for {self.ticker}")
        return quotes

    def save_to_parquet(
        self,
        quotes: pd.DataFrame,
        filename: Optional[str] = None
    ) -> Path:
        """
        Save processed NBBO quotes to Parquet format.

        Args:
            quotes: DataFrame to save
            filename: Output filename (default: {ticker}_quotes_{date}.parquet)

        Returns:
            Path to saved file
        """
        if filename is None:
            date_str = quotes['timestamp'].min().strftime('%Y%m%d')
            filename = f"{self.ticker}_quotes_{date_str}.parquet"

        output_path = self.data_dir / filename
        quotes.to_parquet(output_path, index=False, compression='snappy')

        logger.info(f"Saved {len(quotes):,} quotes to {output_path}")
        return output_path


def load_polygon_quotes(
    ticker: str = "AAPL",
    data_dir: Union[str, Path] = "data/raw/polygon_quotes",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    filter_valid: bool = True,
    compute_features: bool = True,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to load Polygon.io NBBO quote data.

    Args:
        ticker: Stock ticker symbol
        data_dir: Directory containing quote Parquet files
        start_date: Filter start date (ISO format)
        end_date: Filter end date (ISO format)
        filter_valid: Remove locked/crossed markets
        compute_features: Compute NBBO-derived features
        api_key: Polygon.io API key (optional, for direct fetching)

    Returns:
        DataFrame with processed NBBO quote data

    Example:
        >>> quotes = load_polygon_quotes('AAPL', start_date='2024-01-01')
        >>> print(f"Average spread: {quotes['quoted_spread'].mean():.4f}")
    """
    loader = PolygonQuoteLoader(data_dir=data_dir, api_key=api_key, ticker=ticker)
    return loader.load_and_parse(
        start_date=start_date,
        end_date=end_date,
        filter_valid=filter_valid,
        compute_features=compute_features
    )
