"""
Unit tests for Polygon.io data loaders.

Tests the functionality of:
    - PolygonTradeLoader
    - PolygonQuoteLoader
    - PolygonLOBLoader
    - DataSynchronizer

Author: Rylan Spence
Date: 2025
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil

# Import loaders
from src.data.polygon_trade_loader import PolygonTradeLoader, load_polygon_trades
from src.data.polygon_quote_loader import PolygonQuoteLoader, load_polygon_quotes
from src.data.polygon_lob_loader import PolygonLOBLoader, load_polygon_lob
from src.data.synchronizer import DataSynchronizer, align_trades_quotes


# Fixtures for test data
@pytest.fixture
def sample_trades():
    """Create sample trade data."""
    timestamps = pd.date_range('2024-01-01 09:30:00', periods=100, freq='1S')
    return pd.DataFrame({
        'timestamp': timestamps,
        'price': np.random.uniform(150.0, 151.0, 100),
        'size': np.random.randint(100, 1000, 100),
        'exchange': np.random.choice([4, 11, 19, 20], 100),  # NASDAQ, NYSE, ARCA, BATS
        'conditions': [[]] * 50 + [[37]] * 30 + [[12]] * 20,  # Mix of regular and odd-lot
        'trade_id': [f'trade_{i}' for i in range(100)]
    })


@pytest.fixture
def sample_quotes():
    """Create sample NBBO quote data."""
    timestamps = pd.date_range('2024-01-01 09:30:00', periods=100, freq='500ms')
    bid_prices = np.random.uniform(149.90, 150.00, 100)
    ask_prices = bid_prices + np.random.uniform(0.01, 0.05, 100)

    return pd.DataFrame({
        'timestamp': timestamps,
        'bid_price': bid_prices,
        'bid_size': np.random.randint(100, 500, 100),
        'ask_price': ask_prices,
        'ask_size': np.random.randint(100, 500, 100),
        'bid_exchange': np.random.choice([4, 11], 100),
        'ask_exchange': np.random.choice([4, 11], 100),
    })


@pytest.fixture
def sample_lob():
    """Create sample L2 order book snapshots."""
    timestamps = pd.date_range('2024-01-01 09:30:00', periods=50, freq='1S')
    n_levels = 10

    data = {'timestamp': timestamps}

    # Generate bid side (decreasing prices)
    for i in range(n_levels):
        data[f'bid_price_{i}'] = 150.0 - i * 0.01 + np.random.uniform(-0.001, 0.001, 50)
        data[f'bid_size_{i}'] = np.random.randint(100, 1000, 50)

    # Generate ask side (increasing prices)
    for i in range(n_levels):
        data[f'ask_price_{i}'] = 150.05 + i * 0.01 + np.random.uniform(-0.001, 0.001, 50)
        data[f'ask_size_{i}'] = np.random.randint(100, 1000, 50)

    return pd.DataFrame(data)


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


# Test PolygonTradeLoader
class TestPolygonTradeLoader:
    """Test suite for PolygonTradeLoader."""

    def test_init(self, temp_data_dir):
        """Test loader initialization."""
        loader = PolygonTradeLoader(
            data_dir=temp_data_dir,
            ticker='AAPL'
        )
        assert loader.ticker == 'AAPL'
        assert loader.data_dir.exists()

    def test_filter_regular_trades(self, sample_trades):
        """Test filtering for regular trades."""
        loader = PolygonTradeLoader(ticker='AAPL')
        filtered = loader.filter_regular_trades(sample_trades)

        # Should exclude odd-lot trades (condition 37)
        assert len(filtered) < len(sample_trades)
        assert all(37 not in conds for conds in filtered['conditions'])

    def test_save_and_load_parquet(self, sample_trades, temp_data_dir):
        """Test saving and loading trade data."""
        loader = PolygonTradeLoader(data_dir=temp_data_dir, ticker='AAPL')

        # Save
        saved_path = loader.save_to_parquet(sample_trades, filename='test_trades.parquet')
        assert saved_path.exists()

        # Load
        loaded = loader.load_files(pattern='test_trades.parquet')
        assert len(loaded) == len(sample_trades)
        assert 'timestamp' in loaded.columns

    def test_classify_aggressiveness(self, sample_trades, sample_quotes):
        """Test Lee-Ready trade classification."""
        loader = PolygonTradeLoader(ticker='AAPL')

        # Align trades with quotes first
        classified = loader.classify_aggressiveness(sample_trades, sample_quotes)

        assert 'trade_side' in classified.columns
        assert 'is_aggressive_buy' in classified.columns
        assert 'is_aggressive_sell' in classified.columns

        # Check that sides are valid
        assert classified['trade_side'].isin([-1, 0, 1]).all()


# Test PolygonQuoteLoader
class TestPolygonQuoteLoader:
    """Test suite for PolygonQuoteLoader."""

    def test_init(self, temp_data_dir):
        """Test loader initialization."""
        loader = PolygonQuoteLoader(
            data_dir=temp_data_dir,
            ticker='AAPL'
        )
        assert loader.ticker == 'AAPL'
        assert loader.data_dir.exists()

    def test_filter_valid_quotes(self, sample_quotes):
        """Test filtering for valid quotes."""
        loader = PolygonQuoteLoader(ticker='AAPL')

        # Add some invalid quotes
        invalid_quotes = sample_quotes.copy()
        invalid_quotes.loc[0, 'bid_price'] = invalid_quotes.loc[0, 'ask_price']  # Locked market
        invalid_quotes.loc[1, 'bid_price'] = invalid_quotes.loc[1, 'ask_price'] + 0.01  # Crossed

        filtered = loader.filter_valid_quotes(invalid_quotes)

        # Should remove locked and crossed markets
        assert len(filtered) < len(invalid_quotes)
        assert (filtered['ask_price'] > filtered['bid_price']).all()

    def test_compute_nbbo_features(self, sample_quotes):
        """Test NBBO feature computation."""
        loader = PolygonQuoteLoader(ticker='AAPL')
        quotes_with_features = loader.compute_nbbo_features(sample_quotes)

        # Check all expected features exist
        assert 'mid_price' in quotes_with_features.columns
        assert 'quoted_spread' in quotes_with_features.columns
        assert 'relative_spread' in quotes_with_features.columns
        assert 'spread_bps' in quotes_with_features.columns
        assert 'weighted_mid_price' in quotes_with_features.columns
        assert 'order_flow_imbalance' in quotes_with_features.columns

        # Check calculations are correct
        expected_mid = (sample_quotes['bid_price'] + sample_quotes['ask_price']) / 2
        pd.testing.assert_series_equal(
            quotes_with_features['mid_price'],
            expected_mid,
            check_names=False
        )

    def test_resample_quotes(self, sample_quotes):
        """Test quote resampling."""
        loader = PolygonQuoteLoader(ticker='AAPL')
        resampled = loader.resample_quotes(sample_quotes, freq='5S', method='last')

        # Should have fewer rows after resampling
        assert len(resampled) < len(sample_quotes)
        assert 'timestamp' in resampled.columns

    def test_detect_stale_quotes(self, sample_quotes):
        """Test stale quote detection."""
        loader = PolygonQuoteLoader(ticker='AAPL')

        # Add a large gap
        quotes_with_gap = sample_quotes.copy()
        quotes_with_gap.loc[50, 'timestamp'] = quotes_with_gap.loc[49, 'timestamp'] + pd.Timedelta('10S')

        flagged = loader.detect_stale_quotes(quotes_with_gap, max_quote_age=1.0)

        assert 'is_stale' in flagged.columns
        assert flagged['is_stale'].any()  # Should flag the gap


# Test PolygonLOBLoader
class TestPolygonLOBLoader:
    """Test suite for PolygonLOBLoader."""

    def test_init(self, temp_data_dir):
        """Test loader initialization."""
        loader = PolygonLOBLoader(
            data_dir=temp_data_dir,
            ticker='AAPL',
            depth=10
        )
        assert loader.ticker == 'AAPL'
        assert loader.depth == 10

    def test_standardize_column_names(self, sample_lob):
        """Test column name standardization from 1-indexed to 0-indexed."""
        loader = PolygonLOBLoader(ticker='AAPL')

        # Create 1-indexed DataFrame
        lob_1indexed = sample_lob.copy()
        lob_1indexed = lob_1indexed.rename(columns={
            'bid_price_0': 'bid_price_1',
            'bid_size_0': 'bid_size_1',
            'ask_price_0': 'ask_price_1',
            'ask_size_0': 'ask_size_1',
        })

        standardized = loader.standardize_column_names(lob_1indexed, from_1_indexed=True)

        assert 'bid_price_0' in standardized.columns
        assert 'bid_price_1' not in standardized.columns

    def test_parse_lob_levels(self, sample_lob):
        """Test LOB level validation."""
        loader = PolygonLOBLoader(ticker='AAPL', depth=10)
        validated = loader.parse_lob_levels(sample_lob, max_depth=10)

        # Should validate that bid prices are decreasing
        for i in range(9):
            bid_col = f'bid_price_{i}'
            next_bid_col = f'bid_price_{i+1}'
            if bid_col in validated.columns and next_bid_col in validated.columns:
                valid_rows = validated[[bid_col, next_bid_col]].dropna()
                assert (valid_rows[bid_col] > valid_rows[next_bid_col]).all()

    def test_compute_lob_features(self, sample_lob):
        """Test LOB feature computation."""
        loader = PolygonLOBLoader(ticker='AAPL', depth=10)
        lob_with_features = loader.compute_lob_features(sample_lob)

        # Check expected features
        assert 'mid_price' in lob_with_features.columns
        assert 'quoted_spread' in lob_with_features.columns
        assert 'relative_spread' in lob_with_features.columns
        assert 'depth_at_best' in lob_with_features.columns
        assert 'depth_imbalance' in lob_with_features.columns

    def test_compute_book_slope(self, sample_lob):
        """Test order book slope calculation."""
        loader = PolygonLOBLoader(ticker='AAPL', depth=10)
        bid_slope = loader.compute_book_slope(sample_lob, side='bid', max_levels=5)

        assert len(bid_slope) == len(sample_lob)
        assert bid_slope.name == 'bid_slope'


# Test DataSynchronizer
class TestDataSynchronizer:
    """Test suite for DataSynchronizer."""

    def test_init(self, sample_trades, sample_quotes):
        """Test synchronizer initialization."""
        sync = DataSynchronizer(
            trades=sample_trades,
            quotes=sample_quotes,
            tolerance='10ms'
        )
        assert sync.trades is not None
        assert sync.quotes is not None
        assert sync.tolerance == pd.Timedelta('10ms')

    def test_align_trades_with_quotes(self, sample_trades, sample_quotes):
        """Test trade-quote alignment."""
        sync = DataSynchronizer(trades=sample_trades, quotes=sample_quotes)
        aligned = sync.align_trades_with_quotes(method='asof', direction='backward')

        # Should have same number of rows as trades
        assert len(aligned) == len(sample_trades)

        # Should have columns from both trades and quotes
        assert 'price' in aligned.columns  # From trades
        assert 'bid_price' in aligned.columns or 'bid_price_quote' in aligned.columns  # From quotes

    def test_align_trades_with_lob(self, sample_trades, sample_lob):
        """Test trade-LOB alignment."""
        sync = DataSynchronizer(trades=sample_trades, lob=sample_lob)
        aligned = sync.align_trades_with_lob(method='asof', direction='backward')

        # Should have same number of rows as trades
        assert len(aligned) == len(sample_trades)

        # Should have LOB columns
        lob_cols = [col for col in aligned.columns if 'bid_price_' in col or 'ask_price_' in col]
        assert len(lob_cols) > 0

    def test_create_unified_timeline(self, sample_trades, sample_quotes):
        """Test unified timeline creation."""
        sync = DataSynchronizer(trades=sample_trades, quotes=sample_quotes)
        unified = sync.create_unified_timeline(freq='1S', method='ffill')

        # Should have timestamp column
        assert 'timestamp' in unified.columns

        # Should have regular frequency
        time_diffs = unified['timestamp'].diff().dropna()
        assert (time_diffs == pd.Timedelta('1S')).all()

    def test_compute_trade_impact(self, sample_trades, sample_quotes):
        """Test trade impact calculation."""
        sync = DataSynchronizer(trades=sample_trades, quotes=sample_quotes)

        # First align
        aligned = sync.align_trades_with_quotes()

        # Compute NBBO features to get mid_price
        from src.data.polygon_quote_loader import PolygonQuoteLoader
        loader = PolygonQuoteLoader(ticker='AAPL')

        if 'bid_price' in aligned.columns and 'ask_price' in aligned.columns:
            aligned['mid_price'] = (aligned['bid_price'] + aligned['ask_price']) / 2
            aligned['trade_side'] = 1  # Assume all buys for simplicity

            # Compute impact
            with_impact = sync.compute_trade_impact(aligned, horizons=[1, 5, 10])

            assert 'impact_1tick' in with_impact.columns
            assert 'impact_5tick' in with_impact.columns
            assert 'impact_10tick' in with_impact.columns


# Test convenience functions
def test_align_trades_quotes_convenience(sample_trades, sample_quotes):
    """Test convenience function for trade-quote alignment."""
    aligned = align_trades_quotes(sample_trades, sample_quotes, method='asof')
    assert len(aligned) == len(sample_trades)


def test_load_polygon_trades_convenience(sample_trades, temp_data_dir):
    """Test convenience function for loading trades."""
    # Save sample data
    sample_trades.to_parquet(temp_data_dir / 'test.parquet', index=False)

    # Load using convenience function
    loaded = load_polygon_trades(
        ticker='AAPL',
        data_dir=temp_data_dir,
        filter_regular=False
    )

    assert len(loaded) > 0


def test_load_polygon_quotes_convenience(sample_quotes, temp_data_dir):
    """Test convenience function for loading quotes."""
    # Save sample data
    sample_quotes.to_parquet(temp_data_dir / 'test.parquet', index=False)

    # Load using convenience function
    loaded = load_polygon_quotes(
        ticker='AAPL',
        data_dir=temp_data_dir,
        filter_valid=False,
        compute_features=True
    )

    assert len(loaded) > 0
    assert 'mid_price' in loaded.columns  # Features should be computed


# Integration tests
class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline(self, sample_trades, sample_quotes, temp_data_dir):
        """Test full data processing pipeline."""
        # Step 1: Load and process trades
        trade_loader = PolygonTradeLoader(data_dir=temp_data_dir, ticker='AAPL')
        trades = trade_loader.filter_regular_trades(sample_trades)

        # Step 2: Load and process quotes
        quote_loader = PolygonQuoteLoader(data_dir=temp_data_dir, ticker='AAPL')
        quotes = quote_loader.filter_valid_quotes(sample_quotes)
        quotes = quote_loader.compute_nbbo_features(quotes)

        # Step 3: Synchronize
        sync = DataSynchronizer(trades=trades, quotes=quotes)
        aligned = sync.align_trades_with_quotes(method='asof')

        # Step 4: Classify trades
        aligned_with_classification = trade_loader.classify_aggressiveness(
            aligned,
            quotes
        )

        # Verify pipeline output
        assert len(aligned_with_classification) > 0
        assert 'mid_price' in aligned_with_classification.columns
        assert 'trade_side' in aligned_with_classification.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
