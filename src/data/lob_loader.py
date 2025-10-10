"""
Load and parse saved LOB snapshot files.
"""

from pathlib import Path
from typing import List, Optional
import pandas as pd
from ..config import RAW_DATA_DIR


class LOBLoader:
    """Load LOB snapshots from disk."""

    def __init__(self, data_dir: Path = None):
        # Use config default if not specified
        if data_dir is None:
            self.data_dir = RAW_DATA_DIR / "binance_snapshots"
        else:
            self.data_dir = Path(data_dir)

        print(f"LOBLoader initialized with data_dir: {self.data_dir}")
        print(f"Directory exists: {self.data_dir.exists()}")

    def load_files(self, pattern: str = "*.parquet") -> pd.DataFrame:
        """Load all matching LOB files, preserving timestamp."""
        files = sorted(self.data_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(
                f"No files found matching '{pattern}' in {self.data_dir}\n"
                f"Directory contents: {list(self.data_dir.iterdir()) if self.data_dir.exists() else 'Directory does not exist'}"
            )

        print(f"Loading {len(files)} LOB snapshot files...")

        dfs = []
        for file in files:
            df = pd.read_parquet(file)

            # If timestamp is the index, reset it to a column
            if df.index.name == "timestamp" or isinstance(df.index, pd.DatetimeIndex):
                print(f"  Resetting timestamp from index to column in {file.name}")
                df = df.reset_index()

            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)

        # Process timestamp if it exists as a column
        if "timestamp" in combined.columns:
            combined["timestamp"] = pd.to_datetime(combined["timestamp"])
            combined = combined.sort_values("timestamp").reset_index(drop=True)
            print(f"Timestamp column preserved and sorted")
        else:
            print(f"Warning: No timestamp column found")

        print(
            f"Loaded {len(combined):,} snapshots with {len(combined.columns)} columns"
        )

        return combined

    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to consistent format.
        Converts bid_px_N/bid_qty_N to bid_price_N/bid_volume_N (0-indexed)
        Converts ask_px_N/ask_qty_N to ask_price_N/ask_volume_N (0-indexed)
        """
        df = df.copy()

        # Rename columns to standard format
        rename_map = {}
        for col in df.columns:
            if col.startswith("bid_px_"):
                level = col.split("_")[-1]
                # Level is 1-indexed in your data, convert to 0-indexed
                new_level = int(level) - 1
                rename_map[col] = f"bid_price_{new_level}"
            elif col.startswith("bid_qty_"):
                level = col.split("_")[-1]
                new_level = int(level) - 1
                rename_map[col] = f"bid_volume_{new_level}"
            elif col.startswith("ask_px_"):
                level = col.split("_")[-1]
                new_level = int(level) - 1
                rename_map[col] = f"ask_price_{new_level}"
            elif col.startswith("ask_qty_"):
                level = col.split("_")[-1]
                new_level = int(level) - 1
                rename_map[col] = f"ask_volume_{new_level}"

        if rename_map:
            df = df.rename(columns=rename_map)
            print(f"Standardized {len(rename_map)} column names")

        return df

    def parse_lob_levels(self, df: pd.DataFrame, depth: int = 20) -> pd.DataFrame:
        """
        Parse bids/asks into separate columns.

        Handles two formats:
        1. Already parsed (bid_px_N/bid_qty_N) - standardize names
        2. Nested lists (bids/asks columns) - parse into columns
        """
        df = df.copy()

        # Check if already parsed (your format)
        if "bid_px_1" in df.columns or "bid_price_0" in df.columns:
            print("LOB levels already parsed, standardizing column names...")
            df = self.standardize_column_names(df)
            return df

        # Check if nested format
        if "bids" in df.columns and "asks" in df.columns:
            print("Parsing nested bids/asks format...")

            for i in range(depth):
                df[f"bid_price_{i}"] = df["bids"].apply(
                    lambda x: (
                        float(x[i][0]) if isinstance(x, list) and len(x) > i else None
                    )
                )
                df[f"bid_volume_{i}"] = df["bids"].apply(
                    lambda x: (
                        float(x[i][1]) if isinstance(x, list) and len(x) > i else None
                    )
                )
                df[f"ask_price_{i}"] = df["asks"].apply(
                    lambda x: (
                        float(x[i][0]) if isinstance(x, list) and len(x) > i else None
                    )
                )
                df[f"ask_volume_{i}"] = df["asks"].apply(
                    lambda x: (
                        float(x[i][1]) if isinstance(x, list) and len(x) > i else None
                    )
                )

            df = df.drop(columns=["bids", "asks"])

        return df

    def load_and_parse(
        self, pattern: str = "*.parquet", depth: int = 20
    ) -> pd.DataFrame:
        """
        Load files and parse LOB structure.
        Returns DataFrame with timestamp and standardized LOB levels.
        """
        df = self.load_files(pattern)
        df = self.parse_lob_levels(df, depth=depth)

        # Verify timestamp is present
        if "timestamp" not in df.columns:
            print("⚠️  Warning: No timestamp column found in final DataFrame")
        else:
            print(
                f"✅ Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}"
            )

        # Print final structure
        bid_cols = [c for c in df.columns if c.startswith("bid_")]
        ask_cols = [c for c in df.columns if c.startswith("ask_")]
        print(
            f"✅ Final structure: {len(bid_cols)} bid columns, {len(ask_cols)} ask columns"
        )

        return df
