#!/usr/bin/env python3
"""
Order Book Recorder for Binance/Binance.US

This module records real-time limit order book (LOB) snapshots from cryptocurrency
exchanges and saves them to Parquet format for later analysis. It handles WebSocket
streaming, snapshot synchronization, and data persistence.

Features:
    - Real-time order book tracking via WebSocket
    - Automatic snapshot alignment and gap detection
    - Configurable depth (1-1000 levels)
    - Parquet output with incremental writes
    - Support for Binance (global) and Binance.US

Typical Usage:
    # Command line
    python orderbook_recorder.py --venue binanceus --symbol BTCUSDT --depth 20 --seconds 3600

    # In Jupyter notebook
    await amain()

Output Format:
    Parquet file with columns: timestamp (index), bid_px_1, bid_qty_1, ..., ask_px_N, ask_qty_N
    One row per second with top N levels of the order book.

Author: Rylan Spence
"""

import asyncio
import aiohttp
import websockets
import json
import signal
import sys
import time
from datetime import datetime, timezone
from collections import deque
from decimal import Decimal
from typing import Dict, List, Tuple, Optional
import pandas as pd
import pathlib
import argparse
import contextlib

# Venue configurations for supported exchanges
VENUES = {
    "binance": {
        "REST": "https://api.binance.com",
        "WS": "wss://stream.binance.com:9443/ws",
        "ws_stream": "{sym}@depth@100ms",  # Differential depth updates
        "depth_endpoint": "/api/v3/depth",  # Full snapshot endpoint
        "note": "Global (blocked in US)",
    },
    "binanceus": {
        "REST": "https://api.binance.us",
        "WS": "wss://stream.binance.us:9443/ws",
        "ws_stream": "{sym}@depth",  # US uses @depth (100/1000ms server-chosen)
        "depth_endpoint": "/api/v3/depth",
        "note": "US-compliant",
    },
}


def utcnow_ms() -> int:
    """
    Get current UTC time in milliseconds since epoch.

    Returns:
        int: Current timestamp in milliseconds
    """
    return int(time.time() * 1000)


def top_n(
    book_side: Dict[Decimal, Decimal], n: int, reverse: bool
) -> List[Tuple[Decimal, Decimal]]:
    """
    Extract top N price levels from one side of the order book.

    Args:
        book_side: Dictionary mapping price -> quantity
        n: Number of levels to extract
        reverse: If True, sort descending (for bids). If False, ascending (for asks)

    Returns:
        List of (price, quantity) tuples, padded with (NaN, NaN) if fewer than N levels exist

    Example:
        For bids: top_n({100.5: 1.2, 100.0: 2.3}, 3, reverse=True)
        Returns: [(100.5, 1.2), (100.0, 2.3), (NaN, NaN)]
    """
    # Filter out zero-quantity levels and sort by price
    levels = sorted(
        ((p, q) for p, q in book_side.items() if q > 0),
        key=lambda x: x[0],
        reverse=reverse,
    )
    # Return top N levels, padding with NaN if necessary
    return levels[:n] + [(Decimal("NaN"), Decimal("NaN"))] * max(0, n - len(levels))


def apply_update_side(side: Dict[Decimal, Decimal], updates: List[List[str]]):
    """
    Apply differential updates to one side of the order book.

    Updates are in format [[price, quantity], ...] where quantity=0 means delete.

    Args:
        side: Dictionary of current book state (price -> quantity)
        updates: List of [price_str, quantity_str] updates from WebSocket

    Note:
        Modifies 'side' dictionary in place. Zero quantities remove the level.
    """
    for px_s, qty_s in updates:
        px = Decimal(px_s)
        qty = Decimal(qty_s)
        if qty == 0:
            # Quantity of 0 means remove this price level
            side.pop(px, None)
        else:
            # Update or add this price level
            side[px] = qty


class OrderBookRecorder:
    """
    Records real-time order book snapshots from cryptocurrency exchanges.

    This class manages WebSocket connections, maintains synchronized order book state,
    and persists snapshots at 1-second intervals to Parquet format.

    Attributes:
        venue (dict): Venue configuration (URLs, endpoints)
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
        depth (int): Number of price levels to capture (1-1000)
        seconds (int | None): Recording duration in seconds (None = indefinite)
        bids (Dict[Decimal, Decimal]): Current bid side state (price -> qty)
        asks (Dict[Decimal, Decimal]): Current ask side state (price -> qty)
        last_update_id (int | None): Last snapshot update ID for synchronization
        last_u (int | None): Last processed event update ID
        event_buffer (deque): Buffer of WebSocket events awaiting processing
        rows (list[dict]): Accumulated snapshot rows for batch writing

    Synchronization Strategy:
        1. Get initial snapshot via REST API (with lastUpdateId)
        2. Buffer WebSocket events until they align with snapshot
        3. Apply events sequentially, checking for gaps
        4. Re-sync from REST if gap detected
    """

    def __init__(
        self, venue: str, symbol: str, depth: int, out_path: str, seconds: Optional[int]
    ):
        """
        Initialize the order book recorder.

        Args:
            venue: Exchange name ('binance' or 'binanceus')
            symbol: Trading pair (e.g., 'BTCUSDT')
            depth: Number of order book levels to capture (max 1000)
            out_path: Output file path for Parquet data
            seconds: Recording duration in seconds (None = run until stopped)

        Raises:
            ValueError: If venue is not recognized
        """
        if venue not in VENUES:
            raise ValueError(f"Unknown venue '{venue}'. Choose from {list(VENUES)}")

        self.venue = VENUES[venue]
        self.symbol = symbol.upper()
        self.depth = int(depth)
        self.seconds = seconds

        # Construct WebSocket URL based on venue
        self.ws_url = (
            f"{self.venue['WS']}/{self.symbol.lower()}@depth@100ms"
            if venue == "binance"
            else f"{self.venue['WS']}/{self.symbol.lower()}@depth"
        )

        # REST API depth limit (Binance max is 1000)
        self.http_depth_limit = min(self.depth, 1000)
        self.rest_depth_url = self.venue["REST"] + self.venue["depth_endpoint"]

        # Setup output path
        self.out_path = pathlib.Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        # Order book state
        self.bids: Dict[Decimal, Decimal] = {}
        self.asks: Dict[Decimal, Decimal] = {}

        # Synchronization state
        self.last_update_id: Optional[int] = None  # From REST snapshot
        self.last_u: Optional[int] = None  # Last processed update ID

        # Event processing
        self.event_buffer: deque = deque()  # Buffered WebSocket events
        self.rows: list[dict] = []  # Accumulated snapshot rows
        self._stop = False  # Graceful shutdown flag

    def stop(self, *_):
        """
        Signal handler for graceful shutdown.

        Can be called by SIGINT (Ctrl+C) or SIGTERM.
        Sets internal flag to stop recording loop.
        """
        self._stop = True

    async def _get_snapshot(self, session: aiohttp.ClientSession):
        """
        Fetch full order book snapshot from REST API.

        This provides the initial book state and lastUpdateId for synchronization.
        Called at startup and when gap is detected in WebSocket stream.

        Args:
            session: aiohttp client session for HTTP requests

        Raises:
            RuntimeError: If snapshot request fails (network, region blocking, etc.)

        Updates:
            self.bids, self.asks: Full book state from snapshot
            self.last_update_id: Snapshot's lastUpdateId for sync
            self.last_u: Set equal to last_update_id
        """
        params = {"symbol": self.symbol, "limit": self.http_depth_limit}
        async with session.get(self.rest_depth_url, params=params, timeout=20) as resp:
            if resp.status == 451:
                # HTTP 451 = Unavailable For Legal Reasons (region block)
                raise RuntimeError(
                    "HTTP 451 from REST snapshot: venue blocks your region."
                )
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Snapshot failed: HTTP {resp.status}: {text}")
            snap = await resp.json()

        # Parse snapshot and initialize book state
        self.last_update_id = snap["lastUpdateId"]
        self.bids = {Decimal(p): Decimal(q) for p, q in snap["bids"]}
        self.asks = {Decimal(p): Decimal(q) for p, q in snap["asks"]}
        self.last_u = self.last_update_id

    async def _ws_consumer(self, ws):
        """
        Consume WebSocket messages and buffer them for processing.

        Runs as separate async task. Continuously receives differential depth
        updates and adds them to event_buffer for the main loop to process.

        Args:
            ws: WebSocket connection object

        Message Format (Binance):
            {
                "U": first_update_id,      # First update ID in this event
                "u": final_update_id,      # Final update ID in this event
                "pu": prev_final_update_id (optional),  # Previous event's final ID
                "b": [[price, qty], ...],  # Bid updates
                "a": [[price, qty], ...],  # Ask updates
                "E": event_time
            }
        """
        async for msg in ws:
            if isinstance(msg, websockets.frames.Close):
                break
            data = json.loads(msg)

            # Extract relevant fields from depth update message
            U = data.get("U")  # First update ID in this event
            u = data.get("u")  # Final update ID in this event
            pu = data.get("pu")  # Previous final update ID (for continuity check)
            b = data.get("b", [])  # Bid updates
            a = data.get("a", [])  # Ask updates
            E = data.get("E")  # Event time

            # Skip malformed messages
            if U is None or u is None:
                continue

            # Buffer event for main loop to process
            self.event_buffer.append({"U": U, "u": u, "pu": pu, "b": b, "a": a, "E": E})

    def _align_and_apply_buffered(self) -> bool:
        """
        Align WebSocket events with snapshot and apply buffered updates.

        This is the core synchronization logic. It ensures WebSocket updates are
        applied in correct sequence after the REST snapshot, detecting and handling
        gaps in the update stream.

        Synchronization Rules (from Binance API docs):
            1. Buffer events where event_u < snapshot_lastUpdateId (drop old events)
            2. First valid event must satisfy: event_U <= lastUpdateId+1 <= event_u
            3. Subsequent events must be contiguous (via 'pu' or U/u overlap check)

        Returns:
            bool: True if any events were successfully applied, False otherwise

        Side Effects:
            - Updates self.bids and self.asks with new quantities
            - Updates self.last_u to track last processed update ID
            - Removes processed events from event_buffer
        """
        if self.last_update_id is None:
            return False

        applied_any = False
        target = self.last_update_id + 1  # Next expected update ID

        # Drop buffered events older than our snapshot
        while self.event_buffer and self.event_buffer[0]["u"] < target:
            self.event_buffer.popleft()

        if not self.event_buffer:
            return False

        # Check if first buffered event aligns with snapshot
        first = self.event_buffer[0]
        if not (first["U"] <= target <= first["u"]):
            # Not yet aligned - need to wait for more events or re-sync
            return False

        # Apply first aligned event
        apply_update_side(self.bids, first["b"])
        apply_update_side(self.asks, first["a"])
        self.last_u = first["u"]
        self.event_buffer.popleft()
        applied_any = True

        # Apply subsequent events if contiguous
        while self.event_buffer:
            ev = self.event_buffer[0]
            pu = ev.get("pu")

            # Check continuity: either explicit via 'pu' or implicit via U/u overlap
            contiguous = (pu is not None and pu == self.last_u) or (
                pu is None and ev["U"] <= self.last_u + 1 <= ev["u"]
            )

            if not contiguous:
                # Gap detected - stop processing and signal main loop to re-sync
                return applied_any

            # Apply update and continue
            apply_update_side(self.bids, ev["b"])
            apply_update_side(self.asks, ev["a"])
            self.last_u = ev["u"]
            self.event_buffer.popleft()
            applied_any = True

        return applied_any

    def _emit_snapshot_row(self, t_ms: int):
        """
        Capture current order book state as a snapshot row.

        Extracts top N levels from both sides and formats as a flat dictionary
        suitable for DataFrame conversion.

        Args:
            t_ms: Timestamp in milliseconds for this snapshot

        Appends to:
            self.rows: List of snapshot dictionaries

        Row Format:
            {
                'timestamp': datetime,
                'bid_px_1': float, 'bid_qty_1': float,
                'bid_px_2': float, 'bid_qty_2': float,
                ...
                'ask_px_1': float, 'ask_qty_1': float,
                ...
            }
        """
        # Get top N levels from each side
        top_bids = top_n(self.bids, self.depth, reverse=True)
        top_asks = top_n(self.asks, self.depth, reverse=False)

        # Build row dictionary
        row = {"timestamp": pd.to_datetime(t_ms, unit="ms", utc=True)}

        # Add bid levels
        for i, (px, qty) in enumerate(top_bids, start=1):
            row[f"bid_px_{i}"] = (
                float(px) if px == px else None
            )  # px==px checks for NaN
            row[f"bid_qty_{i}"] = float(qty) if qty == qty else None

        # Add ask levels
        for i, (px, qty) in enumerate(top_asks, start=1):
            row[f"ask_px_{i}"] = float(px) if px == px else None
            row[f"ask_qty_{i}"] = float(qty) if qty == qty else None

        self.rows.append(row)

    async def _flush_rows(self):
        """
        Write accumulated snapshot rows to Parquet file.

        Handles incremental writes by:
            1. Converting rows to DataFrame
            2. Reading existing Parquet file if present
            3. Concatenating and deduplicating by timestamp
            4. Writing back to Parquet

        This allows resuming recordings and handling interruptions gracefully.

        Side Effects:
            - Writes to self.out_path
            - Clears self.rows buffer
        """
        if not self.rows:
            return

        # Convert accumulated rows to DataFrame
        df = pd.DataFrame(self.rows).sort_values("timestamp").set_index("timestamp")

        # Merge with existing data if file exists
        if self.out_path.exists() and self.out_path.stat().st_size > 0:
            old = pd.read_parquet(self.out_path)
            df = pd.concat([old, df]).sort_index()
            # Remove duplicates, keeping latest
            df = df[~df.index.duplicated(keep="last")]

        # Write to Parquet
        df.to_parquet(self.out_path)
        print(f"Wrote {len(self.rows)} rows -> {self.out_path}")
        self.rows.clear()

    async def run(self):
        """
        Main recording loop.

        Orchestrates the entire recording process:
            1. Setup signal handlers for graceful shutdown
            2. Connect to WebSocket and start event consumer
            3. Get initial snapshot and align with stream
            4. Emit snapshots every second
            5. Flush to disk periodically (every 30 snapshots)
            6. Handle gaps by re-syncing

        Recording Strategy:
            - Snapshots emitted at whole-second boundaries (t=1000ms, 2000ms, ...)
            - Batch writes every 30 seconds to minimize I/O overhead
            - Automatic re-sync on detected gaps in update stream
            - Graceful shutdown saves all pending data

        Raises:
            RuntimeError: On WebSocket connection failures or region blocking
        """
        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.stop)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        async with aiohttp.ClientSession() as session:
            try:
                # Connect to WebSocket
                async with websockets.connect(
                    self.ws_url, max_queue=2000, ping_interval=20, ping_timeout=20
                ) as ws:
                    # Start WebSocket consumer task
                    consumer_task = asyncio.create_task(self._ws_consumer(ws))

                    # Get initial snapshot and wait for alignment
                    await self._get_snapshot(session)
                    while not self._align_and_apply_buffered():
                        await asyncio.sleep(0.05)

                    print(
                        f"[{self.symbol}] Aligned @ u={self.last_u} "
                        f"(lastUpdateId={self.last_update_id}). Recording..."
                    )

                    # Setup timing for snapshot emission
                    start_ms = utcnow_ms()
                    next_emit_ms = (start_ms // 1000 + 1) * 1000  # Next whole second
                    end_ms = start_ms + self.seconds * 1000 if self.seconds else None

                    try:
                        # Main recording loop
                        while not self._stop:
                            # Process buffered events
                            aligned = self._align_and_apply_buffered()

                            # If not aligned and buffer has events, gap detected
                            if not aligned and self.event_buffer:
                                print("Gap detected; re-syncing snapshot...")
                                self.event_buffer.clear()
                                await self._get_snapshot(session)
                                continue

                            # Emit snapshot at whole-second boundaries
                            now_ms = utcnow_ms()
                            if now_ms >= next_emit_ms:
                                self._emit_snapshot_row(next_emit_ms)
                                next_emit_ms += 1000

                                # Flush to disk every 30 snapshots (30 seconds)
                                if len(self.rows) >= 30:
                                    await self._flush_rows()

                            # Check if recording duration exceeded
                            if end_ms and now_ms >= end_ms:
                                break

                            # Small sleep to avoid busy-waiting
                            await asyncio.sleep(0.01)
                    finally:
                        # Ensure all data is written before exit
                        await self._flush_rows()
                        consumer_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await consumer_task

            except websockets.exceptions.InvalidStatus as e:
                # Common error: HTTP 451 on binance.com from US
                raise RuntimeError(
                    f"WebSocket rejected ({e}). If you are in a restricted region, "
                    f"use --venue binanceus or choose another compliant venue."
                ) from e


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with fields:
            - venue: Exchange name
            - symbol: Trading pair
            - depth: Number of levels
            - seconds: Recording duration
            - out: Output file path
    """
    p = argparse.ArgumentParser(
        description="Order Book Recorder (Binance / Binance.US)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record BTC for 1 hour with 20 levels
  python orderbook_recorder.py --symbol BTCUSDT --depth 20 --seconds 3600
  
  # Record ETH indefinitely (until Ctrl+C)
  python orderbook_recorder.py --symbol ETHUSDT --seconds 0
  
  # Use global Binance (if not in US)
  python orderbook_recorder.py --venue binance --symbol BTCUSDT
        """,
    )
    p.add_argument(
        "--venue",
        default="binanceus",
        choices=list(VENUES),
        help="Exchange venue (default: binanceus)",
    )
    p.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Trading pair symbol, e.g., BTCUSDT, ETHUSDT (default: BTCUSDT)",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=100,
        help="Top-N levels to persist, max 1000 (default: 100)",
    )
    p.add_argument(
        "--seconds",
        type=int,
        default=300,
        help="Duration in seconds, 0=indefinite (default: 300)",
    )
    p.add_argument(
        "--out",
        default="orderbook.parquet",
        help="Output Parquet file path (default: orderbook.parquet)",
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
    rec = OrderBookRecorder(
        venue=args.venue,
        symbol=args.symbol,
        depth=args.depth,
        out_path=args.out,
        seconds=seconds,
    )
    await rec.run()


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
