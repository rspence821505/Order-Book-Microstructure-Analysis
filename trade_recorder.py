#!/usr/bin/env python3
"""
Trade Recorder for Binance/Binance.US

Records individual trades (executions) from cryptocurrency exchanges via WebSocket
and saves them to Parquet format for time-series analysis, particularly for Hawkes
process modeling and order flow analysis.

Features:
    - Real-time trade capture via WebSocket
    - Configurable recording duration
    - Parquet output with timestamp indexing
    - Trade direction classification (buy/sell aggressor)

Typical Usage:
    # Command line
    python trade_recorder.py --venue binanceus --symbol BTCUSDT --seconds 3600

    # In Jupyter notebook
    await amain()

Output Format:
    Parquet file with columns:
        - timestamp (index): Trade execution time (UTC)
        - price: Trade price
        - quantity: Trade size
        - is_buyer_maker: True if sell aggressor, False if buy aggressor

Use Case:
    Hawkes process modeling of trade arrival dynamics

Author: Rylan Spence
"""

import asyncio
import websockets
import json
import pandas as pd
import pathlib
import argparse
import signal
from datetime import datetime
from typing import Optional

# Venue configurations
VENUES = {
    "binance": {
        "WS": "wss://stream.binance.com:9443/ws",
        "note": "Global (blocked in US)",
    },
    "binanceus": {
        "WS": "wss://stream.binance.us:9443/ws",
        "note": "US-compliant",
    },
}


class TradeRecorder:
    """
    Records individual trade executions from cryptocurrency exchanges.

    Captures real-time trade data including price, quantity, and aggressor side
    for analyzing order flow dynamics and building Hawkes process models.

    Attributes:
        venue (dict): Venue configuration
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
        seconds (int | None): Recording duration (None = indefinite)
        output (pathlib.Path): Output file path
        trades (list): Accumulated trade records
        _stop (bool): Graceful shutdown flag

    Trade Message Format (Binance):
        {
            "e": "trade",
            "E": event_time,
            "s": "BTCUSDT",
            "t": trade_id,
            "p": "50000.00",      # Price
            "q": "0.001",         # Quantity
            "T": trade_time,      # Trade timestamp
            "m": true/false,      # Is buyer the maker?
            "M": true/false       # Ignore (always true)
        }
    """

    def __init__(self, venue: str, symbol: str, seconds: Optional[int], output: str):
        """
        Initialize trade recorder.

        Args:
            venue: Exchange name ('binance' or 'binanceus')
            symbol: Trading pair (e.g., 'BTCUSDT')
            seconds: Recording duration in seconds (None = indefinite)
            output: Output file path for Parquet data

        Raises:
            ValueError: If venue is not recognized
        """
        if venue not in VENUES:
            raise ValueError(f"Unknown venue '{venue}'. Choose from {list(VENUES)}")

        self.venue = VENUES[venue]
        self.symbol = symbol.upper()
        self.seconds = seconds
        self.output = pathlib.Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)

        # Construct WebSocket URL
        self.ws_url = f"{self.venue['WS']}/{self.symbol.lower()}@trade"

        # Trade storage
        self.trades = []
        self._stop = False

    def stop(self, *_):
        """
        Signal handler for graceful shutdown.

        Sets internal flag to stop recording loop.
        """
        self._stop = True

    def _save_trades(self):
        """
        Save accumulated trades to Parquet file.

        Converts trade list to DataFrame and writes to disk.
        Merges with existing file if present to support resuming.
        """
        if not self.trades:
            print("No trades to save.")
            return

        # Convert to DataFrame
        df = pd.DataFrame(self.trades).set_index("timestamp")

        # Merge with existing data if file exists
        if self.output.exists() and self.output.stat().st_size > 0:
            old = pd.read_parquet(self.output)
            df = pd.concat([old, df]).sort_index()
            # Remove duplicates by timestamp
            df = df[~df.index.duplicated(keep="last")]

        # Write to Parquet
        df.to_parquet(self.output)
        print(f"Recorded {len(self.trades)} trades to {self.output}")

    async def run(self):
        """
        Main recording loop.

        Connects to WebSocket, receives trade messages, and accumulates them
        until duration expires or user interrupts.

        Trade Direction Classification:
            - is_buyer_maker=True: Buyer provided liquidity (sell aggressor hit bid)
            - is_buyer_maker=False: Seller provided liquidity (buy aggressor lifted ask)

        This is critical for Hawkes modeling where aggressor side matters.

        Raises:
            RuntimeError: On WebSocket connection failures
        """
        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.stop)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        print(f"[{self.symbol}] Connecting to trade stream...")

        try:
            async with websockets.connect(self.ws_url) as ws:
                print(f"[{self.symbol}] Recording trades...")

                start = asyncio.get_event_loop().time()

                async for msg in ws:
                    # Parse trade message
                    data = json.loads(msg)

                    # Append trade record
                    self.trades.append(
                        {
                            "timestamp": pd.to_datetime(data["T"], unit="ms", utc=True),
                            "price": float(data["p"]),
                            "quantity": float(data["q"]),
                            "is_buyer_maker": data["m"],  # True = sell aggressor
                        }
                    )

                    # Check stopping conditions
                    if self._stop:
                        print("\nGraceful shutdown requested...")
                        break

                    if self.seconds and (
                        asyncio.get_event_loop().time() - start > self.seconds
                    ):
                        print(f"\nRecording duration ({self.seconds}s) reached.")
                        break

                    # Progress indicator (every 100 trades)
                    if len(self.trades) % 100 == 0:
                        print(
                            f"\rTrades captured: {len(self.trades)}", end="", flush=True
                        )

        except websockets.exceptions.WebSocketException as e:
            print(f"\nWebSocket error: {e}")
            raise RuntimeError(f"Failed to connect to trade stream: {e}") from e

        finally:
            # Always save accumulated trades
            print()  # New line after progress indicator
            self._save_trades()


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    p = argparse.ArgumentParser(
        description="Trade Recorder (Binance / Binance.US)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record BTC trades for 1 hour
  python trade_recorder.py --symbol BTCUSDT --seconds 3600
  
  # Record ETH trades indefinitely (until Ctrl+C)
  python trade_recorder.py --symbol ETHUSDT --seconds 0
  
  # Custom output location
  python trade_recorder.py --symbol BTCUSDT --out data/raw/btc_trades.parquet
  
  # Use global Binance (if not in US)
  python trade_recorder.py --venue binance --symbol BTCUSDT

Use Case:
  Phase 2, Week 9: Collect trade data for Hawkes process modeling of
  order arrival dynamics and trade clustering analysis.
        """,
    )
    p.add_argument(
        "--venue",
        default="binanceus",
        choices=list(VENUES),
        help="Exchange venue (default: binanceus)",
    )
    p.add_argument(
        "--symbol", default="BTCUSDT", help="Trading pair symbol (default: BTCUSDT)"
    )
    p.add_argument(
        "--seconds",
        type=int,
        default=3600,
        help="Duration in seconds, 0=indefinite (default: 3600)",
    )
    p.add_argument(
        "--out",
        default="data/raw/trades.parquet",
        help="Output Parquet file path (default: data/raw/trades.parquet)",
    )
    return p.parse_args()


async def amain():
    """
    Async main function for running the recorder.

    Parses arguments and starts the recording process.
    Suitable for both CLI and notebook usage.
    """
    args = parse_args()
    seconds = None if args.seconds == 0 else args.seconds

    recorder = TradeRecorder(
        venue=args.venue, symbol=args.symbol, seconds=seconds, output=args.out
    )

    await recorder.run()


if __name__ == "__main__":
    """
    Entry point for CLI usage.

    Detects whether running in notebook (with existing event loop) or CLI,
    and runs appropriately.
    """
    try:
        # Check if already in async context (e.g., Jupyter notebook)
        asyncio.get_running_loop()
    except RuntimeError:
        # No event loop - normal CLI execution
        asyncio.run(amain())
    else:
        # Event loop exists - probably in notebook
        print("Detected running event loop (notebook). Call:  await amain()")
