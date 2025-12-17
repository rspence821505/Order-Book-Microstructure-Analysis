"""
Polygon.io Trade Data Loader

This module provides functionality for loading and processing trade tick data
from Polygon.io's Stocks Developer API for U.S. equities.

Author: Rylan Spence
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Union
from datetime import datetime
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolygonTradeLoader:
    """
    Load and process trade tick data from Polygon.io for U.S. equity microstructure analysis.

    Supports both historical data loading from Parquet files and direct API fetching
    from Polygon.io's /v3/trades/{ticker} endpoint.

    Trade data includes:
        - Timestamp (nanosecond precision)
        - Price (transaction price)
        - Size (number of shares traded)
        - Exchange (venue ID: NASDAQ, NYSE, ARCA, BATS, etc.)
        - Conditions (array of trade condition codes)
        - Trade ID (unique identifier)

    Attributes:
        data_dir (Path): Directory containing Parquet trade files
        api_key (str): Polygon.io API key for direct fetching
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOG')

    Example:
        >>> loader = PolygonTradeLoader(data_dir='data/raw/polygon_trades', ticker='AAPL')
        >>> trades = loader.load_and_parse(filter_regular=True)
        >>> print(f"Loaded {len(trades)} trades")
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/raw/polygon_trades",
        api_key: Optional[str] = None,
        ticker: str = "AAPL"
    ):
        """
        Initialize the Polygon.io trade loader.

        Args:
            data_dir: Directory containing Parquet trade files or where to save fetched data
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
        Load trade data from local Parquet files.

        Args:
            start_date: Filter trades after this date (ISO format: 'YYYY-MM-DD')
            end_date: Filter trades before this date (ISO format: 'YYYY-MM-DD')
            pattern: Glob pattern to match files (default: '*.parquet')

        Returns:
            DataFrame with columns: timestamp, price, size, exchange, conditions, trade_id

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

        trades = pd.concat(dfs, ignore_index=True)

        # Ensure timestamp column is datetime
        if 'timestamp' in trades.columns:
            trades['timestamp'] = pd.to_datetime(trades['timestamp'])
            trades = trades.sort_values('timestamp').reset_index(drop=True)

        # Apply date filters
        if start_date:
            trades = trades[trades['timestamp'] >= pd.Timestamp(start_date)]
        if end_date:
            trades = trades[trades['timestamp'] <= pd.Timestamp(end_date)]

        logger.info(f"Loaded {len(trades):,} trades for {self.ticker}")
        return trades

    def fetch_from_api(
        self,
        date: str,
        limit: int = 50000
    ) -> pd.DataFrame:
        """
        Fetch trade data directly from Polygon.io API.

        Uses the /v3/trades/{ticker} endpoint to retrieve historical trade ticks.

        Args:
            date: Date to fetch (ISO format: 'YYYY-MM-DD')
            limit: Maximum number of trades per request (default: 50000, max: 50000)

        Returns:
            DataFrame with standardized trade columns

        Raises:
            ValueError: If API key not provided
            requests.HTTPError: If API request fails

        API Response Format:
            {
                "results": [
                    {
                        "sip_timestamp": 1609459200000000000,  # nanoseconds
                        "participant_timestamp": 1609459200000000000,
                        "price": 132.69,
                        "size": 100,
                        "exchange": 4,  # Exchange ID
                        "conditions": [12, 37],  # Trade condition codes
                        "id": "12345",
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

        base_url = f"https://api.polygon.io/v3/trades/{self.ticker}"

        params = {
            "timestamp": date,
            "order": "asc",
            "limit": limit,
            "sort": "timestamp",
            "apiKey": self.api_key
        }

        all_trades = []
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

                all_trades.extend(results)
                logger.info(f"Fetched page {page}: {len(results)} trades")

                # Check for pagination
                next_url = data.get("next_url")
                if not next_url:
                    break

                # Add API key to next_url
                next_url = f"{next_url}&apiKey={self.api_key}"
                page += 1

            except requests.HTTPError as e:
                logger.error(f"HTTP error fetching trades: {e}")
                break

        if not all_trades:
            logger.warning(f"No trades fetched for {self.ticker} on {date}")
            return pd.DataFrame()

        # Convert to DataFrame with standardized columns
        trades_df = pd.DataFrame(all_trades)
        trades_df = self._standardize_api_response(trades_df)

        logger.info(f"Fetched {len(trades_df):,} total trades for {date}")
        return trades_df

    def _standardize_api_response(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize Polygon.io API response to internal format.

        Args:
            df: Raw DataFrame from API response

        Returns:
            Standardized DataFrame with columns:
                - timestamp (datetime64[ns])
                - price (float64)
                - size (int64)
                - exchange (int)
                - conditions (list)
                - trade_id (str)
        """
        standardized = pd.DataFrame()

        # Use SIP timestamp (consolidated feed timestamp) as primary
        if 'sip_timestamp' in df.columns:
            standardized['timestamp'] = pd.to_datetime(df['sip_timestamp'], unit='ns')
        elif 'participant_timestamp' in df.columns:
            standardized['timestamp'] = pd.to_datetime(df['participant_timestamp'], unit='ns')
        else:
            raise ValueError("No timestamp column found in API response")

        # Core trade data
        standardized['price'] = df['price'].astype(float)
        standardized['size'] = df['size'].astype(int)
        standardized['exchange'] = df['exchange'].astype(int)
        standardized['conditions'] = df['conditions']  # Keep as list
        standardized['trade_id'] = df.get('id', '').astype(str)

        # Optional: add participant timestamp if available
        if 'participant_timestamp' in df.columns:
            standardized['participant_timestamp'] = pd.to_datetime(
                df['participant_timestamp'], unit='ns'
            )

        return standardized.sort_values('timestamp').reset_index(drop=True)

    def filter_regular_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        Filter for regular trades, excluding odd-lots, out-of-sequence, and other non-standard trades.

        Polygon.io Trade Conditions Reference:
            - Regular trades typically have conditions: []  or [12] (Form T late report)
            - Exclude: Odd-lot (condition 37), Out-of-sequence (52), Opening/Closing auctions, etc.

        Common conditions to EXCLUDE:
            - 7: Qualified Contingent Trade (QCT)
            - 9: Corrected Trade
            - 14, 15: Sold (out of sequence)
            - 16: Stopped stock (regular trade)
            - 37: Odd-lot trade
            - 38: Official closing price
            - 52: Out of sequence

        Args:
            trades: DataFrame with 'conditions' column (list of integers)

        Returns:
            Filtered DataFrame with regular trades only
        """
        # Define conditions to exclude
        exclude_conditions = {
            7, 9, 14, 15, 16, 37, 38, 52,  # Core exclusions
            53, 54, 55, 56,  # Cross trade variations
            6,  # Average price trade
            4,  # Derivatively priced
        }

        def is_regular(condition_list):
            # Handle None first
            if condition_list is None:
                return True
            # Handle numpy arrays or lists BEFORE pd.isna check
            if isinstance(condition_list, (list, np.ndarray)):
                if len(condition_list) == 0:
                    return True
                # Check if any excluded condition is present
                return not any(c in exclude_conditions for c in condition_list)
            # Handle scalar NaN (after checking for arrays)
            if pd.isna(condition_list):
                return True
            # Single scalar value
            return condition_list not in exclude_conditions

        mask = trades['conditions'].apply(is_regular)
        filtered = trades[mask].copy()

        logger.info(
            f"Filtered {len(trades):,} trades to {len(filtered):,} regular trades "
            f"({len(filtered)/len(trades)*100:.1f}% retained)"
        )

        return filtered

    def classify_aggressiveness(
        self,
        trades: pd.DataFrame,
        quotes: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Classify trade aggressiveness using the Lee-Ready algorithm.

        The Lee-Ready algorithm classifies trades as:
            - Aggressive BUY (1): trade at ask or above mid-price
            - Aggressive SELL (-1): trade at bid or below mid-price
            - Unclassified (0): trade exactly at mid-price (rare)

        Requires synchronized NBBO quote data with bid/ask prices.

        Args:
            trades: DataFrame with 'timestamp' and 'price' columns
            quotes: DataFrame with 'timestamp', 'bid_price', 'ask_price' columns
                   If None, aggressiveness cannot be computed

        Returns:
            DataFrame with additional columns:
                - mid_price (float): NBBO mid-price at trade time
                - trade_side (int): 1 = buy, -1 = sell, 0 = unknown
                - is_aggressive_buy (bool): True if trade is aggressive buy
                - is_aggressive_sell (bool): True if trade is aggressive sell
        """
        trades = trades.copy()

        if quotes is None or quotes.empty:
            logger.warning("No quote data provided - cannot classify trade aggressiveness")
            trades['trade_side'] = 0
            trades['is_aggressive_buy'] = False
            trades['is_aggressive_sell'] = False
            return trades

        # Merge trades with nearest prior quote (as-of merge)
        trades = trades.set_index('timestamp')
        quotes = quotes.set_index('timestamp')

        # Forward-fill quotes to align with trade timestamps
        merged = pd.merge_asof(
            trades,
            quotes[['bid_price', 'ask_price']],
            left_index=True,
            right_index=True,
            direction='backward'  # Use most recent quote before trade
        )

        # Compute mid-price
        merged['mid_price'] = (merged['bid_price'] + merged['ask_price']) / 2.0

        # Lee-Ready classification
        merged['trade_side'] = np.where(
            merged['price'] > merged['mid_price'],
            1,  # Buy
            np.where(
                merged['price'] < merged['mid_price'],
                -1,  # Sell
                0   # At mid-price (use tick test if needed)
            )
        )

        # Boolean indicators
        merged['is_aggressive_buy'] = (merged['trade_side'] == 1)
        merged['is_aggressive_sell'] = (merged['trade_side'] == -1)

        merged = merged.reset_index()

        logger.info(
            f"Classified {len(merged):,} trades: "
            f"{(merged['is_aggressive_buy']).sum():,} buys, "
            f"{(merged['is_aggressive_sell']).sum():,} sells"
        )

        return merged

    def load_and_parse(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        filter_regular: bool = True,
        quotes: Optional[pd.DataFrame] = None,
        classify_trades: bool = True
    ) -> pd.DataFrame:
        """
        Main entry point: load, filter, and classify trade data.

        Args:
            start_date: Filter trades after this date (ISO format)
            end_date: Filter trades before this date (ISO format)
            filter_regular: If True, exclude odd-lots and non-standard trades
            quotes: Optional NBBO quote data for trade classification
            classify_trades: If True, apply Lee-Ready classification (requires quotes)

        Returns:
            Fully processed DataFrame ready for microstructure analysis

        Example:
            >>> loader = PolygonTradeLoader(ticker='AAPL')
            >>> trades = loader.load_and_parse(
            ...     start_date='2024-01-01',
            ...     filter_regular=True,
            ...     classify_trades=True
            ... )
        """
        # Step 1: Load trade data
        trades = self.load_files(start_date=start_date, end_date=end_date)

        # Step 2: Filter for regular trades
        if filter_regular:
            trades = self.filter_regular_trades(trades)

        # Step 3: Classify aggressiveness
        if classify_trades and quotes is not None:
            trades = self.classify_aggressiveness(trades, quotes)

        logger.info(f"Final dataset: {len(trades):,} trades for {self.ticker}")
        return trades

    def save_to_parquet(
        self,
        trades: pd.DataFrame,
        filename: Optional[str] = None
    ) -> Path:
        """
        Save processed trades to Parquet format.

        Args:
            trades: DataFrame to save
            filename: Output filename (default: {ticker}_trades_{date}.parquet)

        Returns:
            Path to saved file
        """
        if filename is None:
            date_str = trades['timestamp'].min().strftime('%Y%m%d')
            filename = f"{self.ticker}_trades_{date_str}.parquet"

        output_path = self.data_dir / filename
        trades.to_parquet(output_path, index=False, compression='snappy')

        logger.info(f"Saved {len(trades):,} trades to {output_path}")
        return output_path


def load_polygon_trades(
    ticker: str = "AAPL",
    data_dir: Union[str, Path] = "data/raw/polygon_trades",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    filter_regular: bool = True,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to load Polygon.io trade data.

    Args:
        ticker: Stock ticker symbol
        data_dir: Directory containing trade Parquet files
        start_date: Filter start date (ISO format)
        end_date: Filter end date (ISO format)
        filter_regular: Exclude odd-lots and non-standard trades
        api_key: Polygon.io API key (optional, for direct fetching)

    Returns:
        DataFrame with processed trade data

    Example:
        >>> trades = load_polygon_trades('AAPL', start_date='2024-01-01')
        >>> print(trades.head())
    """
    loader = PolygonTradeLoader(data_dir=data_dir, api_key=api_key, ticker=ticker)
    return loader.load_and_parse(
        start_date=start_date,
        end_date=end_date,
        filter_regular=filter_regular,
        classify_trades=False  # Requires quotes
    )
