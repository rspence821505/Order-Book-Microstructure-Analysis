# # Record both LOB and trades for 1 hour
# python scripts/download_data.py --symbol BTCUSDT --duration 3600 --record-type both

# # Record only LOB snapshots
# python scripts/download_data.py --symbol ETHUSDT --duration 7200 --record-type lob

"""
Main script to download and record LOB + trade data.
"""

import argparse
from pathlib import Path
from datetime import datetime
from recorders.orderbook_recorder import OrderBookRecorder
from recorders.trade_recorder import TradeRecorder


def main():
    parser = argparse.ArgumentParser(description="Download LOB and trade data")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument(
        "--duration", type=int, default=3600, help="Duration in seconds"
    )
    parser.add_argument("--lob-depth", type=int, default=20, help="LOB depth levels")
    parser.add_argument(
        "--lob-interval", type=int, default=100, help="LOB snapshot interval (ms)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/raw", help="Output directory"
    )
    parser.add_argument(
        "--record-type", choices=["lob", "trades", "both"], default="both"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print(f"Starting data recording for {args.symbol}")
    print(f"Duration: {args.duration}s")
    print(f"Output: {output_dir}")

    if args.record_type in ["lob", "both"]:
        print("\n=== Recording Order Book ===")
        lob_recorder = OrderBookRecorder(
            symbol=args.symbol,
            depth=args.lob_depth,
            output_dir=output_dir / "binance_snapshots",
        )
        lob_recorder.record(
            duration_seconds=args.duration, interval_ms=args.lob_interval
        )

    if args.record_type in ["trades", "both"]:
        print("\n=== Recording Trades ===")
        trade_recorder = TradeRecorder(
            symbol=args.symbol, output_dir=output_dir / "binance_trades"
        )
        trade_recorder.record(duration_seconds=args.duration)

    print("\n=== Recording Complete ===")


if __name__ == "__main__":
    main()
