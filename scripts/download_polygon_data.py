#!/usr/bin/env python3
"""
Download historical data from Polygon.io for U.S. equities.

Usage:
    # Download 1 day of AAPL data (all types)
    python scripts/download_polygon_data.py --ticker AAPL --start-date 2025-01-15

    # Download multiple days with specific data types
    python scripts/download_polygon_data.py --ticker AAPL --start-date 2025-01-13 --end-date 2025-01-17 --data-type trades

    # Download for multiple tickers
    python scripts/download_polygon_data.py --ticker AAPL GOOG NVDA --start-date 2025-01-15
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.polygon_trade_loader import PolygonTradeLoader
from src.data.polygon_quote_loader import PolygonQuoteLoader
from src.data.polygon_lob_loader import PolygonLOBLoader
from src.data.polygon_aggregate_loader import PolygonAggregateLoader


def load_api_key(api_key_file: Path) -> str:
    """Load API key from file or environment variable."""
    # Try environment variable first
    api_key = os.getenv("POLYGON_API_KEY")
    if api_key:
        return api_key

    # Try loading from file
    if api_key_file.exists():
        with open(api_key_file, "r") as f:
            api_key = f.read().strip()
        return api_key

    raise ValueError(
        f"API key not found. Either:\n"
        f"1. Set POLYGON_API_KEY environment variable, or\n"
        f"2. Create file: {api_key_file}"
    )


def download_trades(ticker: str, start_date: str, end_date: str, output_dir: Path, api_key: str):
    """Download trade tick data from Polygon.io API."""
    print(f"\n{'='*60}")
    print(f"Downloading TRADES for {ticker}")
    print(f"{'='*60}")

    try:
        # Create loader with API key
        loader = PolygonTradeLoader(
            data_dir=output_dir,
            api_key=api_key,
            ticker=ticker
        )

        # Fetch data from API for the specified date
        # Note: For simplicity, we're fetching only the start_date
        # For multi-day ranges, loop through dates
        trades_df = loader.fetch_from_api(date=start_date)

        if trades_df is None or len(trades_df) == 0:
            print(f"  ⚠️  No trade data available for {ticker} on {start_date}")
            return

        # Filter for regular trades (exclude odd-lots, etc.)
        trades_df = loader.filter_regular_trades(trades_df)

        # Save to parquet
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{ticker}_{start_date}_trades.parquet"
        output_path = output_dir / filename
        trades_df.to_parquet(output_path, index=False)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved {len(trades_df):,} trades")
        print(f"  ✓ File: {output_path}")
        print(f"  ✓ Size: {file_size_mb:.2f} MB")

    except Exception as e:
        print(f"  ✗ Error downloading trades: {e}")
        import traceback
        traceback.print_exc()


def download_quotes(ticker: str, start_date: str, end_date: str, output_dir: Path, api_key: str):
    """Download NBBO quote data from Polygon.io API."""
    print(f"\n{'='*60}")
    print(f"Downloading QUOTES (NBBO) for {ticker}")
    print(f"{'='*60}")

    try:
        # Create loader with API key
        loader = PolygonQuoteLoader(
            data_dir=output_dir,
            api_key=api_key,
            ticker=ticker
        )

        # Fetch data from API for the specified date
        quotes_df = loader.fetch_from_api(date=start_date)

        if quotes_df is None or len(quotes_df) == 0:
            print(f"  ⚠️  No quote data available for {ticker} on {start_date}")
            return

        # Compute features (spread, mid-price, etc.)
        quotes_df = loader.compute_nbbo_features(quotes_df)

        # Save to parquet
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{ticker}_{start_date}_quotes.parquet"
        output_path = output_dir / filename
        quotes_df.to_parquet(output_path, index=False)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved {len(quotes_df):,} quotes")
        print(f"  ✓ File: {output_path}")
        print(f"  ✓ Size: {file_size_mb:.2f} MB")

    except Exception as e:
        print(f"  ✗ Error downloading quotes: {e}")
        import traceback
        traceback.print_exc()


def download_lob(
    ticker: str, start_date: str, end_date: str, output_dir: Path, api_key: str, depth: int = 10
):
    """Download L2 order book snapshots from Polygon.io API."""
    print(f"\n{'='*60}")
    print(f"Downloading LOB SNAPSHOTS for {ticker} (depth={depth})")
    print(f"{'='*60}")

    try:
        # Create loader with API key
        loader = PolygonLOBLoader(
            data_dir=output_dir,
            api_key=api_key,
            ticker=ticker,
            depth=depth
        )

        # Fetch snapshot from API
        lob_df = loader.fetch_snapshot_from_api()

        if lob_df is None or len(lob_df) == 0:
            print(f"  ⚠️  No LOB data available for {ticker}")
            return

        # Compute book shape features
        lob_df = loader.compute_book_features(lob_df)

        # Save to parquet
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{ticker}_{start_date}_lob.parquet"
        output_path = output_dir / filename
        lob_df.to_parquet(output_path, index=False)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved {len(lob_df):,} snapshots")
        print(f"  ✓ File: {output_path}")
        print(f"  ✓ Size: {file_size_mb:.2f} MB")

    except Exception as e:
        print(f"  ✗ Error downloading LOB: {e}")
        import traceback
        traceback.print_exc()


def download_aggregates(
    ticker: str, start_date: str, end_date: str, output_dir: Path, api_key: str,
    timespan: str = 'minute', multiplier: int = 1
):
    """Download minute/second aggregate bars from Polygon.io API."""
    print(f"\n{'='*60}")
    print(f"Downloading {multiplier}-{timespan.upper()} AGGREGATES for {ticker}")
    print(f"{'='*60}")

    try:
        # Create loader with API key
        loader = PolygonAggregateLoader(
            data_dir=output_dir,
            api_key=api_key,
            ticker=ticker
        )

        # Fetch data from API for the specified date
        agg_df = loader.fetch_from_api(
            date=start_date,
            timespan=timespan,
            multiplier=multiplier
        )

        if agg_df is None or len(agg_df) == 0:
            print(f"  ⚠️  No aggregate data available for {ticker} on {start_date}")
            return

        # Compute spread features (spread ≈ High - Low)
        agg_df = loader.compute_spread_features(agg_df)

        # Save to parquet
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{ticker}_{start_date}_{multiplier}{timespan}_bars.parquet"
        output_path = output_dir / filename
        agg_df.to_parquet(output_path, index=False)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved {len(agg_df):,} bars")
        print(f"  ✓ File: {output_path}")
        print(f"  ✓ Size: {file_size_mb:.2f} MB")

        # Show sample spread stats
        avg_spread = agg_df['estimated_spread'].mean()
        avg_spread_bps = (agg_df['relative_spread'].mean() * 10000)
        print(f"  ✓ Avg spread: ${avg_spread:.4f} ({avg_spread_bps:.2f} bps)")

    except Exception as e:
        print(f"  ✗ Error downloading aggregates: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Download historical Polygon.io data for U.S. equities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--ticker",
        type=str,
        nargs="+",
        default=["AAPL"],
        help="Stock ticker(s) to download (e.g., AAPL GOOG NVDA)",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date in YYYY-MM-DD format (e.g., 2025-01-15)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format (default: same as start-date)",
    )

    parser.add_argument(
        "--data-type",
        choices=["trades", "quotes", "lob", "aggregates", "all"],
        default="all",
        help="Type of data to download (default: all)",
    )

    parser.add_argument(
        "--timespan",
        choices=["second", "minute", "hour", "day"],
        default="minute",
        help="Aggregate bar timespan (default: minute)",
    )

    parser.add_argument(
        "--multiplier",
        type=int,
        default=1,
        help="Aggregate multiplier (e.g., 5 for 5-minute bars, default: 1)",
    )

    parser.add_argument(
        "--lob-depth",
        type=int,
        default=10,
        help="Number of price levels for LOB snapshots (default: 10)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory (default: data/raw)",
    )

    parser.add_argument(
        "--api-key-file",
        type=str,
        default="polygon_api_key.txt",
        help="Path to API key file (default: polygon_api_key.txt)",
    )

    args = parser.parse_args()

    # Set end_date to start_date if not specified
    if args.end_date is None:
        args.end_date = args.start_date

    # Validate dates
    try:
        datetime.strptime(args.start_date, "%Y-%m-%d")
        datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError:
        print("Error: Dates must be in YYYY-MM-DD format")
        return 1

    # Load API key
    api_key_file = project_root / args.api_key_file
    try:
        api_key = load_api_key(api_key_file)
        os.environ["POLYGON_API_KEY"] = api_key
        print(f"✓ Loaded API key")
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Setup output directories
    output_dir = Path(args.output_dir)
    trades_dir = output_dir / "polygon_trades"
    quotes_dir = output_dir / "polygon_quotes"
    lob_dir = output_dir / "polygon_snapshots"
    aggregates_dir = output_dir / "polygon_aggregates"

    # Print configuration
    print(f"\n{'='*60}")
    print("POLYGON DATA DOWNLOAD")
    print(f"{'='*60}")
    print(f"Ticker(s): {', '.join(args.ticker)}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Data types: {args.data_type}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

    # Download data for each ticker
    for ticker in args.ticker:
        print(f"\n\n{'#'*60}")
        print(f"# Processing {ticker}")
        print(f"{'#'*60}")

        if args.data_type in ["trades", "all"]:
            download_trades(ticker, args.start_date, args.end_date, trades_dir, api_key)

        if args.data_type in ["aggregates", "all"]:
            download_aggregates(
                ticker, args.start_date, args.end_date, aggregates_dir, api_key,
                args.timespan, args.multiplier
            )

        if args.data_type in ["quotes", "all"]:
            download_quotes(ticker, args.start_date, args.end_date, quotes_dir, api_key)

        if args.data_type in ["lob", "all"]:
            download_lob(
                ticker, args.start_date, args.end_date, lob_dir, api_key, args.lob_depth
            )

    print(f"\n\n{'='*60}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Data saved to: {output_dir}")
    print(f"\nNext step: Run notebooks/00_data_collections.ipynb to process the data")

    return 0


if __name__ == "__main__":
    sys.exit(main())
