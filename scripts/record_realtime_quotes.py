#!/usr/bin/env python3
"""
Record real-time (15-min delayed) quotes from Polygon.io WebSocket.

This script streams NBBO quotes for specified tickers and saves them to disk,
allowing you to build up a historical tick-quote dataset over time.

Usage:
    # Record quotes for AAPL (runs until stopped with Ctrl+C)
    python scripts/record_realtime_quotes.py --ticker AAPL

    # Record multiple tickers
    python scripts/record_realtime_quotes.py --ticker AAPL GOOG NVDA

    # Specify output directory
    python scripts/record_realtime_quotes.py --ticker AAPL --output-dir data/raw/polygon_quotes
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import signal
import json
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Check if polygon package is installed
try:
    from polygon import WebSocketClient
    from polygon.websocket.models import WebSocketMessage, Market, Feed
except ImportError:
    print("Error: polygon package not installed!")
    print("Install with: pip install polygon-api-client")
    sys.exit(1)


class QuoteRecorder:
    """
    Record real-time NBBO quotes from Polygon.io WebSocket.

    Automatically saves quotes to Parquet files every N minutes or on shutdown.
    """

    def __init__(
        self,
        api_key: str,
        tickers: List[str],
        output_dir: Path,
        save_interval_minutes: int = 15
    ):
        """
        Initialize the quote recorder.

        Args:
            api_key: Polygon.io API key
            tickers: List of ticker symbols to record
            output_dir: Directory to save quote files
            save_interval_minutes: How often to save data to disk
        """
        self.api_key = api_key
        self.tickers = [t.upper() for t in tickers]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval_minutes

        # Buffer for storing quotes before saving
        self.quote_buffer = []
        self.last_save_time = datetime.now()
        self.running = True

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        print(f"QuoteRecorder initialized")
        print(f"  Tickers: {', '.join(self.tickers)}")
        print(f"  Output: {self.output_dir}")
        print(f"  Save interval: {self.save_interval} minutes")

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n\nReceived shutdown signal, saving data...")
        self.running = False
        self._save_buffer()
        sys.exit(0)

    def _save_buffer(self):
        """Save buffered quotes to Parquet file."""
        if not self.quote_buffer:
            print("No quotes to save")
            return

        # Convert buffer to DataFrame
        df = pd.DataFrame(self.quote_buffer)

        # Generate filename with timestamp
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H%M%S')

        for ticker in self.tickers:
            ticker_quotes = df[df['ticker'] == ticker].copy()

            if len(ticker_quotes) == 0:
                continue

            filename = f"{ticker}_{date_str}_{time_str}_quotes.parquet"
            output_path = self.output_dir / filename

            ticker_quotes.to_parquet(output_path, index=False)

            print(f"  ✓ Saved {len(ticker_quotes):,} quotes for {ticker}")
            print(f"    File: {output_path}")
            print(f"    Size: {output_path.stat().st_size / 1024:.1f} KB")

        # Clear buffer
        self.quote_buffer = []
        self.last_save_time = datetime.now()

    def _handle_quote(self, msg: WebSocketMessage):
        """
        Handle incoming quote message.

        Args:
            msg: WebSocket message containing quote data
        """
        try:
            # Extract quote data
            quote = {
                'ticker': msg.symbol if hasattr(msg, 'symbol') else msg.get('T'),
                'timestamp': pd.to_datetime(msg.timestamp if hasattr(msg, 'timestamp') else msg.get('t'), unit='ns', utc=True),
                'bid_price': msg.bid_price if hasattr(msg, 'bid_price') else msg.get('bp'),
                'ask_price': msg.ask_price if hasattr(msg, 'ask_price') else msg.get('ap'),
                'bid_size': msg.bid_size if hasattr(msg, 'bid_size') else msg.get('bs'),
                'ask_size': msg.ask_size if hasattr(msg, 'ask_size') else msg.get('as'),
                'bid_exchange': msg.bid_exchange if hasattr(msg, 'bid_exchange') else msg.get('bx', None),
                'ask_exchange': msg.ask_exchange if hasattr(msg, 'ask_exchange') else msg.get('ax', None),
            }

            self.quote_buffer.append(quote)

            # Print status every 100 quotes
            if len(self.quote_buffer) % 100 == 0:
                print(f"  Buffered {len(self.quote_buffer):,} quotes...")

            # Check if it's time to save
            if datetime.now() - self.last_save_time > timedelta(minutes=self.save_interval):
                print(f"\n{'='*60}")
                print(f"Saving buffered quotes (interval: {self.save_interval} min)")
                print(f"{'='*60}")
                self._save_buffer()

        except Exception as e:
            print(f"Error handling quote: {e}")

    def start(self):
        """Start recording quotes from WebSocket."""
        print(f"\n{'='*60}")
        print("STARTING QUOTE RECORDING")
        print(f"{'='*60}")
        print(f"Connecting to Polygon WebSocket...")
        print(f"Recording quotes for: {', '.join(self.tickers)}")
        print(f"Press Ctrl+C to stop recording\n")

        try:
            # Create WebSocket client
            # Note: Delayed data feed for Developer tier
            ws_client = WebSocketClient(
                api_key=self.api_key,
                market=Market.Stocks,
                feed=Feed.Delayed  # 15-minute delayed quotes
            )

            # Subscribe to quote events
            ws_client.subscribe(*[f"Q.{ticker}" for ticker in self.tickers])

            # Start listening
            ws_client.run(self._handle_quote)

        except KeyboardInterrupt:
            print("\n\nStopping recording...")
            self._save_buffer()

        except Exception as e:
            print(f"\nError: {e}")
            self._save_buffer()
            raise


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


def main():
    parser = argparse.ArgumentParser(
        description="Record real-time (15-min delayed) quotes from Polygon.io",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--ticker",
        type=str,
        nargs="+",
        default=["AAPL"],
        help="Stock ticker(s) to record (e.g., AAPL GOOG NVDA)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/polygon_quotes",
        help="Output directory (default: data/raw/polygon_quotes)",
    )

    parser.add_argument(
        "--save-interval",
        type=int,
        default=15,
        help="Save buffered quotes every N minutes (default: 15)",
    )

    parser.add_argument(
        "--api-key-file",
        type=str,
        default="polygon_api_key.txt",
        help="Path to API key file (default: polygon_api_key.txt)",
    )

    args = parser.parse_args()

    # Load API key
    api_key_file = project_root / args.api_key_file
    try:
        api_key = load_api_key(api_key_file)
        print(f"✓ Loaded API key")
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Create recorder
    recorder = QuoteRecorder(
        api_key=api_key,
        tickers=args.ticker,
        output_dir=Path(args.output_dir),
        save_interval_minutes=args.save_interval
    )

    # Start recording
    recorder.start()

    return 0


if __name__ == "__main__":
    sys.exit(main())
