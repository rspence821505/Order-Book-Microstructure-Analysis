"""
Load and parse saved trade files.
"""

from pathlib import Path
import pandas as pd


class TradeLoader:
    """Load trade data from disk."""

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data/raw/binance_trades")

    def load_files(self, pattern: str = "*.parquet") -> pd.DataFrame:
        """Load all matching trade files."""
        files = sorted(self.data_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(f"No files found in {self.data_dir}")

        print(f"Loading {len(files)} trade files...")

        dfs = [pd.read_parquet(f) for f in files]
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)

        print(f"Loaded {len(combined):,} trades")
        return combined

    def compute_aggressiveness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trade aggressiveness features."""
        df["side"] = df["is_buyer_maker"].map({True: "sell", False: "buy"})
        # More features can be added here
        return df
