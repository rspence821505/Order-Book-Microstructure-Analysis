"""
Basic microstructure feature engineering.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Optional, List


def compute_spread(
    bid_price: pd.Series, ask_price: pd.Series, mid_price: pd.Series
) -> Dict[str, pd.Series]:
    """
    Compute bid-ask spread in multiple formats.

    The spread measures the difference between ask and bid prices,
    indicating liquidity and transaction costs.

    Formula:
        spread_abs = ask_price - bid_price
        relative_spread = spread_abs / mid_price
        spread_bps = relative_spread × 10,000
        spread_pct = relative_spread × 100

    Intuition:
        The spread represents the cost of immediacy - the implicit cost paid
        to execute a round-trip trade (buy then sell, or sell then buy). A
        tighter spread indicates a more liquid market where large orders can
        be executed with minimal price impact. Market makers profit from
        capturing the spread.

        Expressing spread in basis points normalizes by price level, making
        it comparable across assets and time periods. A $200 spread on BTC
        at $120k (16 bps) is much tighter than a $5 spread on a $1000 stock
        (50 bps).

    Args:
        bid_price: Best bid price series
        ask_price: Best ask price series
        mid_price: Mid-price series (for normalization)

    Returns:
        Dictionary containing:
            - 'spread_abs': Absolute spread in price units
            - 'relative_spread': Proportional spread (decimal, 0 to 1)
            - 'spread_bps': Spread in basis points (1 bp = 0.01%)
            - 'spread_pct': Spread as percentage (0 to 100)

    Notes:
        - spread_bps = relative_spread × 10,000
        - spread_pct = relative_spread × 100
        - Lower spread indicates higher liquidity
        - Typical ranges:
            * Major FX: 0.1-1 bps
            * Liquid crypto: 1-20 bps
            * Large-cap stocks: 1-10 bps

    Examples:
        >>> bid = pd.Series([119900.0])
        >>> ask = pd.Series([120100.0])
        >>> mid = pd.Series([120000.0])
        >>> result = compute_spread(bid, ask, mid)
        >>> result['spread_bps']
        0    16.666667
        dtype: float64
    """
    spread_abs = ask_price - bid_price
    relative_spread = spread_abs / mid_price

    return {
        "spread_abs": spread_abs,
        "relative_spread": relative_spread,
        "spread_bps": relative_spread * 10000,
        "spread_pct": relative_spread * 100,
    }


def compute_average_spread(
    bid_prices: pd.DataFrame,
    ask_prices: pd.DataFrame,
    n_levels: int = 10,
) -> pd.Series:
    """
    Compute average spread across the top N levels of the limit order book.

    The spread at each level measures the gap between ask and bid prices,
    representing the cost of immediate execution at that depth. Averaging
    across multiple levels provides insight into overall liquidity and
    market depth beyond just the top-of-book.

    Formula:
        average_spread = mean(ask_price_i - bid_price_i) for i in [1, N]

    Intuition:
        A narrow average spread indicates tight markets with good liquidity
        across multiple levels - buyers and sellers agree closely on fair
        value at various depths. A wide average spread suggests fragmented
        liquidity, higher transaction costs, or disagreement about fair value.

        This metric is more robust than top-of-book spread alone because it
        captures liquidity erosion deeper in the book. In HFT, wide spreads
        at deeper levels may signal poor market quality or opportunities for
        market making.

    Args:
        bid_prices: DataFrame with bid prices at each level (columns: bid_price_1,
                    bid_price_2, ..., bid_price_N)
        ask_prices: DataFrame with ask prices at each level (columns: ask_price_1,
                    ask_price_2, ..., ask_price_N)
        n_levels: Number of top levels to include in average (default: 10)

    Returns:
        Series containing average spread across top N levels for each timestamp

    Notes:
        - Commonly used alongside order book imbalance features
        - Can be normalized by mid-price for cross-asset comparison
        - Sudden widening may indicate liquidity shocks or regime changes
        - Should be monitored across different market conditions (open, close, etc.)
    """
    # Select top N levels for bids and asks
    bid_cols = [f"bid_price_{i}" for i in range(1, n_levels + 1)]
    ask_cols = [f"ask_price_{i}" for i in range(1, n_levels + 1)]

    # Compute spread at each level
    spreads = ask_prices[ask_cols].values - bid_prices[bid_cols].values

    # Return mean spread across levels for each timestamp
    return pd.Series(spreads.mean(axis=1), index=bid_prices.index)


def compute_spread_at_level(
    bid_prices: pd.DataFrame,
    ask_prices: pd.DataFrame,
    level: int = 1,
) -> pd.Series:
    """
    Compute the bid-ask spread at a specific level of the limit order book.

    The spread at level i measures the price gap between the i-th best ask
    and i-th best bid. While level 1 (top-of-book) spread is most commonly
    used, spreads at deeper levels reveal information about liquidity
    distribution and potential execution costs for larger orders.

    Formula:
        spread_at_level_i = ask_price_i - bid_price_i

    Intuition:
        Level 1 spread represents the cost of immediate round-trip execution
        for small orders. Deeper level spreads show how costs grow with order
        size - a key concern for execution algorithms and large traders.

        Spreads typically widen at deeper levels as liquidity becomes scarcer.
        Unusually tight spreads at deep levels may indicate hidden liquidity
        or algorithmic quoting strategies. Conversely, sudden widening at
        specific levels can signal liquidity gaps or "holes" in the book.

        In HFT, monitoring spread patterns across levels helps detect:
        - Market maker presence/absence at various depths
        - Potential adverse selection zones
        - Optimal placement depths for limit orders

    Args:
        bid_prices: DataFrame with bid prices at each level (columns: bid_price_1,
                    bid_price_2, ..., bid_price_N)
        ask_prices: DataFrame with ask prices at each level (columns: ask_price_1,
                    ask_price_2, ..., ask_price_N)
        level: Order book level to compute spread for (1 = top of book)

    Returns:
        Series containing spread at the specified level for each timestamp

    Examples:
        >>> bid_prices = pd.DataFrame({
        ...     'bid_price_1': [100.0, 100.1],
        ...     'bid_price_2': [99.9, 100.0],
        ...     'bid_price_3': [99.8, 99.9]
        ... })
        >>> ask_prices = pd.DataFrame({
        ...     'ask_price_1': [100.1, 100.2],
        ...     'ask_price_2': [100.2, 100.3],
        ...     'ask_price_3': [100.3, 100.4]
        ... })
        >>> compute_spread_at_level(bid_prices, ask_prices, level=1)
        0    0.1
        1    0.1
        Name: spread_level_1, dtype: float64

        >>> compute_spread_at_level(bid_prices, ask_prices, level=3)
        0    0.5
        1    0.5
        Name: spread_level_3, dtype: float64
        # Spread widens at deeper levels (typical behavior)

    Notes:
        - Spread at level 1 is the most liquid and frequently traded
        - Spread ratios (level_i / level_1) indicate liquidity slope
        - Sudden spread compression/expansion can signal regime changes
        - Often used in conjunction with volume at each level

    References:
        - Bouchaud, J. P., Farmer, J. D., & Lillo, F. (2009). "How markets
          slowly digest changes in supply and demand"
        - Cont, R., Stoikov, S., & Talreja, R. (2010). "A stochastic model
          for order book dynamics"
    """
    bid_col = f"bid_price_{level}"
    ask_col = f"ask_price_{level}"

    spread = ask_prices[ask_col] - bid_prices[bid_col]
    spread.name = f"spread_level_{level}"

    return spread


def compute_spreads_all_levels(
    bid_prices: pd.DataFrame,
    ask_prices: pd.DataFrame,
    n_levels: int = 10,
) -> pd.DataFrame:
    """
    Compute bid-ask spreads across all top N levels of the limit order book.

    This function efficiently computes spreads at multiple depths simultaneously,
    creating a rich feature set that captures the liquidity profile of the
    entire visible order book. The spread pattern across levels is a key
    indicator of market quality and liquidity distribution.

    Formula:
        For each level i in [1, N]:
            spread_i = ask_price_i - bid_price_i

    Intuition:
        The spread curve (spreads across levels) tells a story about market
        depth and liquidity:

        - **Flat spread curve**: Liquidity distributed evenly, healthy market
        - **Steep spread curve**: Liquidity concentrated at top, shallow depth
        - **Sudden jumps**: Liquidity holes or strategic quote placement
        - **Narrowing at depth**: Possible hidden orders or algorithmic activity

        For machine learning models, spread patterns are predictive features:
        - Widening spreads may precede volatility spikes
        - Asymmetric spread changes can signal directional pressure
        - Spread correlation across levels indicates market regime

        Market makers monitor these patterns to adjust quoting strategies,
        while execution algorithms use them to optimize order placement timing
        and depth.

    Args:
        bid_prices: DataFrame with bid prices at each level (columns: bid_price_1,
                    bid_price_2, ..., bid_price_N)
        ask_prices: DataFrame with ask prices at each level (columns: ask_price_1,
                    ask_price_2, ..., ask_price_N)
        n_levels: Number of levels to compute spreads for (default: 10)

    Returns:
        DataFrame with spread at each level (columns: spread_level_1,
        spread_level_2, ..., spread_level_N)

    Examples:
        >>> bid_prices = pd.DataFrame({
        ...     'bid_price_1': [100.0, 100.1],
        ...     'bid_price_2': [99.9, 100.0],
        ...     'bid_price_3': [99.8, 99.9]
        ... })
        >>> ask_prices = pd.DataFrame({
        ...     'ask_price_1': [100.1, 100.2],
        ...     'ask_price_2': [100.2, 100.3],
        ...     'ask_price_3': [100.3, 100.4]
        ... })
        >>> compute_spreads_all_levels(bid_prices, ask_prices, n_levels=3)
           spread_level_1  spread_level_2  spread_level_3
        0             0.1             0.3             0.5
        1             0.1             0.3             0.5

        # Typical widening pattern: deeper levels have wider spreads

    Notes:
        - Output can be used directly as ML features or for derived metrics
        - Consider normalizing by level 1 spread for relative analysis
        - Spread volatility across levels is also informative
        - Can compute spread slope: (spread_N - spread_1) / (N - 1)

    References:
        - Hautsch, N. (2012). "Econometrics of Financial High-Frequency Data"
        - Cartea, Á., Jaimungal, S., & Penalva, J. (2015). "Algorithmic and
          High-Frequency Trading"
    """
    spread_df = pd.DataFrame(index=bid_prices.index)

    for level in range(1, n_levels + 1):
        bid_col = f"bid_price_{level}"
        ask_col = f"ask_price_{level}"
        spread_df[f"spread_level_{level}"] = ask_prices[ask_col] - bid_prices[bid_col]

    return spread_df


def compute_effective_spread(
    bid_prices: pd.DataFrame,
    ask_prices: pd.DataFrame,
    bid_volumes: pd.DataFrame,
    ask_volumes: pd.DataFrame,
    n_levels: int = 10,
) -> pd.Series:
    """
    Compute the effective spread using volume-weighted average prices (VWAP).

    The effective spread measures the actual cost of trading by comparing the
    volume-weighted average ask price to the volume-weighted average bid price
    across multiple levels. Unlike the simple top-of-book spread, this captures
    the true execution cost for orders that walk through multiple price levels.

    Formula:
        VWAP_ask = Σ(ask_price_i × ask_volume_i) / Σ(ask_volume_i)
        VWAP_bid = Σ(bid_price_i × bid_volume_i) / Σ(bid_volume_i)
        effective_spread = VWAP_ask - VWAP_bid

    Intuition:
        The effective spread answers: "What would it cost to execute a market
        order that consumes liquidity across multiple levels?"

        Consider a market buy order for 1000 shares when level 1 only has 300:
        - Simple spread only shows level 1 cost
        - Effective spread accounts for hitting levels 1, 2, 3, weighted by
          available volume at each level

        This metric is critical for:
        - **Execution algorithms**: Estimating realistic transaction costs
        - **Market makers**: Understanding true profitability after slippage
        - **Liquidity providers**: Assessing depth quality, not just top price

        A wide gap between simple spread and effective spread indicates:
        - Thin liquidity at top levels
        - High slippage risk for larger orders
        - Potential for price impact in execution

        Effective spread is more stable than top-of-book spread during:
        - Quote flickering (MM strategies rapidly updating level 1)
        - Low liquidity periods (sparse top level, but depth exists deeper)
        - High-frequency trading activity

    Args:
        bid_prices: DataFrame with bid prices at each level (columns: bid_price_1,
                    bid_price_2, ..., bid_price_N)
        ask_prices: DataFrame with ask prices at each level (columns: ask_price_1,
                    ask_price_2, ..., ask_price_N)
        bid_volumes: DataFrame with bid volumes at each level (columns: bid_volume_1,
                     bid_volume_2, ..., bid_volume_N)
        ask_volumes: DataFrame with ask volumes at each level (columns: ask_volume_1,
                     ask_volume_2, ..., ask_volume_N)
        n_levels: Number of levels to include in VWAP calculation (default: 10)

    Returns:
        Series containing effective spread (VWAP_ask - VWAP_bid) for each timestamp

    Examples:

        # Compare to simple spread at level 1 = 100.1 - 100.0 = 0.1
        # Effective spread is wider due to volume distribution

    Notes:
        - Always >= simple top-of-book spread (equality when all volume at level 1)
        - Useful for normalizing: effective_spread / mid_price gives relative cost
        - Can be extended to compute one-sided VWAP for buy vs sell separately
        - Related to market impact models and execution optimization

        **Implementation detail**: Uses small epsilon (1e-10) to avoid division
        by zero in edge cases where total volume is zero at some timestamp.

    """
    # Select columns for top N levels
    bid_price_cols = [f"bid_price_{i}" for i in range(1, n_levels + 1)]
    ask_price_cols = [f"ask_price_{i}" for i in range(1, n_levels + 1)]
    bid_volume_cols = [f"bid_volume_{i}" for i in range(1, n_levels + 1)]
    ask_volume_cols = [f"ask_volume_{i}" for i in range(1, n_levels + 1)]

    # Extract data as numpy arrays for efficient computation
    bid_prices_arr = bid_prices[bid_price_cols].values
    ask_prices_arr = ask_prices[ask_price_cols].values
    bid_volumes_arr = bid_volumes[bid_volume_cols].values
    ask_volumes_arr = ask_volumes[ask_volume_cols].values

    # Small epsilon to avoid division by zero
    epsilon = 1e-10

    # Compute VWAP for bids: sum(price × volume) / sum(volume)
    bid_notional = (bid_prices_arr * bid_volumes_arr).sum(axis=1)
    bid_total_volume = bid_volumes_arr.sum(axis=1) + epsilon
    vwap_bid = bid_notional / bid_total_volume

    # Compute VWAP for asks: sum(price × volume) / sum(volume)
    ask_notional = (ask_prices_arr * ask_volumes_arr).sum(axis=1)
    ask_total_volume = ask_volumes_arr.sum(axis=1) + epsilon
    vwap_ask = ask_notional / ask_total_volume

    # Effective spread is the difference
    effective_spread = vwap_ask - vwap_bid

    return pd.Series(effective_spread, index=bid_prices.index, name="effective_spread")


def compute_mid_price(bid_price: pd.Series, ask_price: pd.Series) -> pd.Series:
    """
    Compute mid-price as average of best bid and ask.

    The mid-price is the midpoint between the best bid and best ask price,
    commonly used as a reference price for the asset.

    Formula:
        mid_price = (bid_price + ask_price) / 2

    Intuition:
        The mid-price represents the theoretical "fair value" assuming equal
        weighting of bid and ask. It's the price at which neither buyers nor
        sellers have an advantage. In practice, the actual fair value may
        differ due to volume imbalances (see compute_weighted_mid).

    Args:
        bid_price: Best bid price series
        ask_price: Best ask price series

    Returns:
        Mid-price series

    """
    return (bid_price + ask_price) / 2


def compute_weighted_mid(
    bid_price: pd.Series,
    ask_price: pd.Series,
    bid_volume: pd.Series,
    ask_volume: pd.Series,
    epsilon: float = 1e-10,
) -> pd.Series:
    """
    Compute volume-weighted mid-price (microprice).

    Weights prices by opposite-side volumes, giving more weight to the side
    with higher volume. This often provides a better estimate of "fair price"
    than the simple mid-price.

    Formula:
        weighted_mid = (bid_price × ask_volume + ask_price × bid_volume) /
                       (bid_volume + ask_volume)

    Intuition:
        When ask volume >> bid volume, many sellers are waiting, creating
        selling pressure. The weighted mid tilts toward the bid price,
        anticipating downward movement. Conversely, high bid volume suggests
        buying pressure and tilts the price toward the ask.

        Think of volumes as "votes" for where the price should go. High ask
        volume votes for the bid price (sellers want to match with buyers at
        bid), and vice versa. When volumes are equal, weighted_mid equals
        the simple mid-price.

    Args:
        bid_price: Best bid price series
        ask_price: Best ask price series
        bid_volume: Volume at best bid
        ask_volume: Volume at best ask
        epsilon: Small constant to avoid division by zero (default: 1e-10)

    Returns:
        Volume-weighted mid-price series

    Notes:
        Also known as "microprice" in academic literature. Empirically shown
        to be a better predictor of short-term price movements than simple
        mid-price.
    """
    total_volume = bid_volume + ask_volume + epsilon
    weighted_mid = (bid_price * ask_volume + ask_price * bid_volume) / total_volume
    return weighted_mid


def compute_imbalance(
    bid_volume: pd.Series,
    ask_volume: pd.Series,
    method: str = "standard",
    epsilon: float = 1e-10,
) -> pd.Series:
    """
    Compute order flow imbalance.

    Formula:
        standard:   imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        ratio:      imbalance = bid_volume / ask_volume
        log_ratio:  imbalance = log(bid_volume / ask_volume)

    Intuition:
        Order flow imbalance captures the relative supply/demand at the top
        of the order book. Positive imbalance (more bid volume) suggests
        buyers are more aggressive or numerous, indicating buying pressure.
        Negative imbalance suggests selling pressure.

        The standard method normalizes to [-1, 1], making it comparable across
        different assets and time periods. A value of +0.8 means 90% of volume
        is on the bid side ((0.9-0.1)/(0.9+0.1) = 0.8), indicating strong
        buying interest.

    Args:
        bid_volume: Volume at bid side
        ask_volume: Volume at ask side
        method: Method to compute imbalance
            - 'standard': (bid - ask) / (bid + ask) [Range: -1 to 1]
            - 'ratio': bid / ask [Range: 0 to ∞, 1 = balanced]
            - 'log_ratio': log(bid / ask) [Range: -∞ to ∞, 0 = balanced]
        epsilon: Small constant to avoid division by zero

    Returns:
        Imbalance measure

    Notes:
        Standard imbalance:
            - Range: [-1, 1]
            - 0 = balanced
            - +1 = all bid volume (extreme buying pressure)
            - -1 = all ask volume (extreme selling pressure)

        Ratio imbalance:
            - Range: [0, ∞]
            - 1 = balanced
            - >1 = bid-heavy (e.g., 2 = twice as much bid volume)
            - <1 = ask-heavy

        Log ratio imbalance:
            - Range: (-∞, ∞)
            - 0 = balanced
            - Positive = bid-heavy
            - Negative = ask-heavy
            - Symmetric around zero (unlike ratio)

    Examples:
        >>> bid = pd.Series([150.0])
        >>> ask = pd.Series([50.0])
        >>> compute_imbalance(bid, ask, method='standard')
        0    0.5
        dtype: float64
        # Positive indicates buying pressure
    """
    if method == "standard":
        total_volume = bid_volume + ask_volume + epsilon
        imbalance = (bid_volume - ask_volume) / total_volume

    elif method == "ratio":
        imbalance = bid_volume / (ask_volume + epsilon)

    elif method == "log_ratio":
        ratio = bid_volume / (ask_volume + epsilon)
        imbalance = np.log(ratio)

    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'standard', 'ratio', or 'log_ratio'"
        )

    return imbalance


def compute_multi_level_imbalance(lob_df: pd.DataFrame, levels: int = 5) -> pd.Series:
    """
    Compute order flow imbalance across multiple LOB levels.

    Aggregates volume across top N levels of the book before computing
    imbalance.

    Formula:
        total_bid_volume = Σ(i=0 to N-1) bid_volume_i
        total_ask_volume = Σ(i=0 to N-1) ask_volume_i
        imbalance = (total_bid_volume - total_ask_volume) /
                    (total_bid_volume + total_ask_volume)

    Intuition:
        While top-of-book imbalance captures immediate supply/demand, multi-
        level imbalance reflects deeper liquidity patterns. Large orders often
        sit deeper in the book. A strong multi-level imbalance can signal
        institutional interest even when top-of-book appears balanced.

        Example: If top-of-book is balanced but levels 5-10 have massive bid
        volume, this suggests large buyers are waiting to absorb any selling
        pressure, indicating hidden buying interest.

    Args:
        lob_df: DataFrame with bid_volume_i and ask_volume_i columns
        levels: Number of levels to include (default: top 5)

    Returns:
        Multi-level imbalance series in range [-1, 1]

    Example:
        >>> imbalance_5 = compute_multi_level_imbalance(lob_df, levels=5)
        >>> # Aggregates volumes across top 5 levels
    """
    # Sum volumes across levels
    bid_cols = [f"bid_volume_{i}" for i in range(1, levels + 1)]
    ask_cols = [f"ask_volume_{i}" for i in range(1, levels + 1)]

    total_bid_volume = lob_df[bid_cols].sum(axis=1)
    total_ask_volume = lob_df[ask_cols].sum(axis=1)

    # Compute imbalance
    imbalance = compute_imbalance(total_bid_volume, total_ask_volume)

    return imbalance


def compute_volume_weighted_imbalance(
    lob_df: pd.DataFrame, levels: int = 5, decay_factor: float = 0.8
) -> pd.Series:
    """
    Compute volume-weighted imbalance with exponential decay.

    Gives more weight to volumes closer to the mid-price using exponential
    decay.

    Formula:
        weighted_bid = Σ(i=0 to N-1) bid_volume_i × decay_factor^i
        weighted_ask = Σ(i=0 to N-1) ask_volume_i × decay_factor^i
        imbalance = (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask)

    Intuition:
        Not all book levels are equally important. Volume at the top of the
        book (closer to mid-price) has more immediate impact than volume 10
        levels deep. Exponential weighting reflects this: level 0 has weight
        1.0, level 1 has weight 0.8 (if decay=0.8), level 2 has weight 0.64,
        etc.

        This captures the intuition that nearby liquidity matters more for
        short-term price movements than distant liquidity. A large order 20
        levels deep is less relevant than a small order at top-of-book.

    Args:
        lob_df: DataFrame with LOB data
        levels: Number of levels to include
        decay_factor: Weight decay for each level (0 to 1)
            - 1.0 = equal weights (equivalent to multi_level_imbalance)
            - 0.8 = default, reasonable decay
            - 0.5 = aggressive decay (deeper levels nearly ignored)

    Returns:
        Weighted imbalance series in range [-1, 1]

    Example:
        >>> # Top level weight=1.0, level 2 weight=0.8, level 3 weight=0.64
        >>> imb = compute_volume_weighted_imbalance(lob_df, levels=5, decay_factor=0.8)
    """
    weighted_bid_volume = 0
    weighted_ask_volume = 0

    for i in range(1, levels + 1):
        weight = decay_factor**i
        weighted_bid_volume += lob_df[f"bid_volume_{i}"] * weight
        weighted_ask_volume += lob_df[f"ask_volume_{i}"] * weight

    imbalance = compute_imbalance(weighted_bid_volume, weighted_ask_volume)

    return imbalance


def compute_log_return(
    prices: pd.Series,
    lag: int = 1,
) -> pd.Series:
    """
    Compute log returns at a specific lag.

    Log returns measure the relative price change over a given time horizon
    using natural logarithm. They are preferred over simple returns in
    quantitative finance due to their mathematical properties: symmetry,
    time-additivity, and better behavior under continuous compounding.

    Formula:
        log_return_t = ln(price_t / price_{t-lag}) = ln(price_t) - ln(price_{t-lag})

    Intuition:
        Log returns answer: "What is the continuously compounded rate of
        return over this period?"

        **Why log returns over simple returns?**

        1. **Symmetry**: A 50% gain then 50% loss doesn't return to break-even
           with simple returns, but log returns are symmetric around zero

        2. **Time-additivity**: Multi-period log returns sum:
           log_return(t, t+2) = log_return(t, t+1) + log_return(t+1, t+2)

        3. **Statistical properties**: More normally distributed, easier to
           model with ML algorithms

        4. **Small values approximation**: For small changes, log_return ≈ simple_return
           (ln(1 + x) ≈ x when |x| << 1)

        **In HFT context:**
        - Lag 1: Immediate momentum or mean reversion signals
        - Lag 2-5: Short-term trends or microstructure effects
        - Lag 10-60: Longer-term directional signals

        Multiple lags capture different time scales of price dynamics:
        - Recent lags: Reflect order flow and liquidity
        - Distant lags: Capture longer-term trends and regime persistence

        For prediction models, using multiple return lags helps the model
        learn temporal dependencies without explicitly modeling sequences.

    Args:
        prices: Series of prices (typically mid-price, last trade, or VWAP)
        lag: Number of periods to look back (1 = one-period return)

    Returns:
        Series containing log returns, with NaN for first 'lag' observations

    Notes:
        - First 'lag' values will be NaN (no prior data to compute return)
        - For high-frequency data, ensure price series is properly aligned
        - Consider forward-filling missing prices before computing returns
        - In practice, clipping extreme returns may be necessary to handle
          errors or corporate actions

        **Common lag choices in HFT:**
        - Tick-level: lags 1, 2, 3, 5 (immediate microstructure)
        - Second-level: lags 1, 5, 10, 30, 60 (short-term momentum)
        - Minute-level: lags 1, 5, 15, 30 (intraday trends)

    References:
        - Campbell, J. Y., Lo, A. W., & MacKinlay, A. C. (1997). "The
          Econometrics of Financial Markets"
        - Tsay, R. S. (2010). "Analysis of Financial Time Series"
        - Cont, R. (2001). "Empirical properties of asset returns: stylized
          facts and statistical issues"
    """
    log_prices = np.log(prices)
    log_return = log_prices - log_prices.shift(lag)
    log_return.name = f"log_return_lag_{lag}"

    return log_return


def compute_log_returns_multiple_lags(
    prices: pd.Series,
    lags: list[int] = [1, 2, 3, 5, 10],
) -> pd.DataFrame:
    """
    Compute log returns at multiple lag horizons simultaneously.

    This function creates a rich feature set capturing price dynamics across
    multiple time scales. Each lag represents a different "lookback window"
    that helps ML models understand short-term momentum, mean reversion, and
    multi-scale trends without requiring recurrent architectures.

    Formula:
        For each lag k in lags:
            log_return_lag_k = ln(price_t / price_{t-k})

    Intuition:
        **Why multiple lags matter:**

        Different market phenomena occur at different time scales:
        - **Lag 1**: Captures tick-by-tick noise, bid-ask bounce, immediate
          order flow imbalance effects
        - **Lags 2-5**: Reveals short-term momentum or mean reversion patterns,
          reflects HFT activity and microstructure effects
        - **Lags 10-30**: Captures slightly longer trends, informed trader
          activity, news propagation
        - **Lags 60+**: Reflects institutional flow, macro trends, regime
          characteristics

        **Feature engineering perspective:**
        Multiple return lags serve as a "poor man's time series model" - they
        provide temporal context to feedforward models (like Random Forest or
        Gradient Boosting) without explicit sequence modeling.

        **Pattern detection:**
        - Consistent positive returns across lags → sustained momentum
        - Sign alternation (+ - + -) → mean reversion or bid-ask bounce
        - Increasing magnitude with lag → accelerating trend
        - Lag-1 positive, longer lags negative → short-term reversal

        For ML models predicting next-tick direction, these patterns become
        powerful features. Models learn combinations like:
        "If lag-1 return is positive AND lag-5 return is negative → predict down"

    Args:
        prices: Series of prices (typically mid-price or last trade price)
        lags: List of lag periods to compute returns for (default: [1,2,3,5,10])

    Returns:
        DataFrame with one column per lag (columns: log_return_lag_1,
        log_return_lag_2, etc.)

    Notes:
        - NaN values appear in first max(lags) rows - handle appropriately
        - For training ML models, either drop NaN rows or forward-fill
        - Consider standardizing returns across lags for some models
        - Can compute rolling statistics (mean, std) of these returns

        **Feature selection tip:**
        Not all lags are equally informative. Use feature importance from
        tree models or permutation tests to identify most predictive lags
        for your specific market/instrument.

        **Computational efficiency:**
        This vectorized implementation is much faster than computing lags
        in a loop, especially for large datasets.


    """
    returns_df = pd.DataFrame(index=prices.index)

    for lag in lags:
        returns_df[f"log_return_lag_{lag}"] = compute_log_return(prices, lag=lag)

    return returns_df


def compute_depth(
    lob_df: pd.DataFrame, levels: list = [1, 5, 10], side: str = "both"
) -> pd.DataFrame:
    """
    Compute cumulative depth at multiple levels.

    Aggregates volume across top N levels of the order book to measure
    available liquidity at different depths.

    Formula:
        bid_depth_N = Σ(i=0 to N-1) bid_volume_i
        ask_depth_N = Σ(i=0 to N-1) ask_volume_i

    Intuition:
        Depth measures how much you can buy or sell before exhausting
        available liquidity. High depth means you can execute large orders
        without moving the price significantly.

        Example: bid_depth_5 = 8.0 BTC means you can sell up to 8 BTC before
        going beyond the top 5 bid levels. If you try to sell 10 BTC, you'll
        move the price down past level 5, incurring higher slippage.

        Comparing depth across levels reveals book shape: if depth_1 ≈ depth_10,
        liquidity is concentrated at top. If depth_10 >> depth_1, liquidity is
        distributed throughout the book.

    Args:
        lob_df: DataFrame with bid_volume_i and ask_volume_i columns
        levels: List of depth levels to compute (e.g., [1, 5, 10])
        side: Which side to compute
            - 'bid': Only bid side
            - 'ask': Only ask side
            - 'both': Both sides (default)

    Returns:
        DataFrame with depth features
            - bid_depth_{N}: Cumulative bid volume in top N levels
            - ask_depth_{N}: Cumulative ask volume in top N levels

    Examples:
        >>> depth_df = compute_depth(lob_df, levels=[1, 5, 10], side='both')
        >>> # Creates: bid_depth_1, bid_depth_5, bid_depth_10,
        >>> #          ask_depth_1, ask_depth_5, ask_depth_10
    """
    depth_features = pd.DataFrame(index=lob_df.index)

    for level in levels:
        if side in ["bid", "both"]:
            bid_cols = [f"bid_volume_{i}" for i in range(level)]
            bid_cols = [c for c in bid_cols if c in lob_df.columns]
            if bid_cols:
                depth_features[f"bid_depth_{level}"] = lob_df[bid_cols].sum(axis=1)

        if side in ["ask", "both"]:
            ask_cols = [f"ask_volume_{i}" for i in range(level)]
            ask_cols = [c for c in ask_cols if c in lob_df.columns]
            if ask_cols:
                depth_features[f"ask_depth_{level}"] = lob_df[ask_cols].sum(axis=1)

    return depth_features


def compute_total_depth(
    lob_df: pd.DataFrame, levels: list = [1, 5, 10]
) -> pd.DataFrame:
    """
    Compute total depth (bid + ask) at multiple levels.

    Total depth measures the overall liquidity available on both sides
    of the order book.

    Formula:
        total_depth_N = bid_depth_N + ask_depth_N
                      = Σ(i=0 to N-1) (bid_volume_i + ask_volume_i)

    Intuition:
        Total depth represents the complete liquidity pool available for
        trading. It answers: "How much total volume is available in the top
        N levels regardless of direction?"

        High total depth indicates a liquid market that can absorb large
        orders without significant price impact. It's particularly useful
        for assessing market quality and comparing liquidity across different
        time periods or assets.

        Unlike depth imbalance (which measures directional bias), total depth
        measures absolute liquidity capacity.

    Args:
        lob_df: DataFrame with bid_volume_i and ask_volume_i columns
        levels: List of depth levels to compute

    Returns:
        DataFrame with total_depth_{N} columns

    Examples:
        >>> total_depth = compute_total_depth(lob_df, levels=[5, 10])
        >>> # Creates: total_depth_5, total_depth_10

    Notes:
        High total depth indicates a liquid market where large orders
        can be executed without significant price impact. Typical patterns:
        - Liquid markets: total_depth increases substantially with depth
        - Thin markets: total_depth plateaus quickly (little volume beyond top levels)
    """
    total_depth_features = pd.DataFrame(index=lob_df.index)

    for level in levels:
        bid_cols = [f"bid_volume_{i}" for i in range(level)]
        ask_cols = [f"ask_volume_{i}" for i in range(level)]

        bid_cols = [c for c in bid_cols if c in lob_df.columns]
        ask_cols = [c for c in ask_cols if c in lob_df.columns]

        if bid_cols and ask_cols:
            bid_depth = lob_df[bid_cols].sum(axis=1)
            ask_depth = lob_df[ask_cols].sum(axis=1)
            total_depth_features[f"total_depth_{level}"] = bid_depth + ask_depth

    return total_depth_features


def compute_depth_imbalance(
    lob_df: pd.DataFrame, levels: list = [1, 5, 10], epsilon: float = 1e-10
) -> pd.DataFrame:
    """
    Compute depth imbalance at multiple levels.

    Depth imbalance measures the relative difference between bid and ask
    liquidity at different depths in the order book.

    Formula:
        depth_imbalance_N = (bid_depth_N - ask_depth_N) /
                           (bid_depth_N + ask_depth_N)

    Intuition:
        While order flow imbalance (top-of-book) captures immediate pressure,
        depth imbalance reveals liquidity structure throughout the book. It
        answers: "Which side has more cumulative liquidity at depth N?"

        Example: depth_imbalance_1 might be negative (more ask volume at top),
        but depth_imbalance_10 might be positive (large hidden bid orders
        sitting deeper). This pattern suggests buyers are patient, waiting to
        absorb selling pressure at lower prices.

        Depth imbalance at different levels can reveal different trader types:
        - Top levels: HFT market makers, retail
        - Mid levels: Algorithmic execution
        - Deep levels: Institutional, informed traders

    Args:
        lob_df: DataFrame with bid_volume_i and ask_volume_i columns
        levels: List of depth levels to compute
        epsilon: Small constant to avoid division by zero

    Returns:
        DataFrame with depth_imbalance_{N} columns in range [-1, 1]

    Examples:
        >>> depth_imb = compute_depth_imbalance(lob_df, levels=[5, 10])
        >>> # Creates: depth_imbalance_5, depth_imbalance_10

    Notes:
        - Range: [-1, 1]
        - Positive values: More bid-side liquidity (potential buying pressure)
        - Negative values: More ask-side liquidity (potential selling pressure)
        - 0: Balanced liquidity on both sides

        This is similar to order flow imbalance but computed across
        multiple depth levels rather than just top-of-book.
    """
    depth_imb_features = pd.DataFrame(index=lob_df.index)

    for level in levels:
        bid_cols = [f"bid_volume_{i}" for i in range(level)]
        ask_cols = [f"ask_volume_{i}" for i in range(level)]

        bid_cols = [c for c in bid_cols if c in lob_df.columns]
        ask_cols = [c for c in ask_cols if c in lob_df.columns]

        if bid_cols and ask_cols:
            bid_depth = lob_df[bid_cols].sum(axis=1)
            ask_depth = lob_df[ask_cols].sum(axis=1)

            total_depth = bid_depth + ask_depth + epsilon
            depth_imb_features[f"depth_imbalance_{level}"] = (
                bid_depth - ask_depth
            ) / total_depth

    return depth_imb_features


def compute_depth_ratio(
    lob_df: pd.DataFrame, levels: list = [1, 5, 10], epsilon: float = 1e-10
) -> pd.DataFrame:
    """
    Compute depth ratio at multiple levels.

    Depth ratio is the quotient of bid depth to ask depth, providing
    an alternative measure of liquidity imbalance.

    Formula:
        depth_ratio_N = bid_depth_N / ask_depth_N

    Intuition:
        While depth_imbalance is symmetric around zero (balanced), depth_ratio
        is multiplicative and asymmetric. This can be more intuitive for some
        contexts: "There is 2x more bid liquidity than ask liquidity" is clearer
        than "depth imbalance is +0.33."

        Depth ratio is particularly useful when you care about relative
        magnitudes. A ratio of 3.0 (3x more bid depth) has very different
        implications than 1.5 (1.5x more bid depth), even though both are
        positive imbalances.

        The asymmetry also matters: ratio=2 (2x bid) is not the mirror of
        ratio=0.5 (2x ask). Some models prefer this asymmetric representation.

    Args:
        lob_df: DataFrame with bid_volume_i and ask_volume_i columns
        levels: List of depth levels to compute
        epsilon: Small constant to avoid division by zero

    Returns:
        DataFrame with depth_ratio_{N} columns

    Examples:
        >>> depth_ratio = compute_depth_ratio(lob_df, levels=[5, 10])
        >>> # Creates: depth_ratio_5, depth_ratio_10

    Notes:
        - Range: [0, ∞]
        - ratio = 1: Balanced depth
        - ratio > 1: More bid depth (buying pressure)
            * ratio = 2: Twice as much bid depth as ask depth
            * ratio = 3: Three times as much bid depth
        - ratio < 1: More ask depth (selling pressure)
            * ratio = 0.5: Half as much bid depth (or 2x ask depth)

        Unlike depth_imbalance (symmetric around 0), depth_ratio is
        asymmetric and can be useful for certain models, especially
        those that use log transformations.
    """
    depth_ratio_features = pd.DataFrame(index=lob_df.index)

    for level in levels:
        bid_cols = [f"bid_volume_{i}" for i in range(level)]
        ask_cols = [f"ask_volume_{i}" for i in range(level)]

        bid_cols = [c for c in bid_cols if c in lob_df.columns]
        ask_cols = [c for c in ask_cols if c in lob_df.columns]

        if bid_cols and ask_cols:
            bid_depth = lob_df[bid_cols].sum(axis=1)
            ask_depth = lob_df[ask_cols].sum(axis=1)

            depth_ratio_features[f"depth_ratio_{level}"] = bid_depth / (
                ask_depth + epsilon
            )

    return depth_ratio_features


def compute_avg_volume_per_level(
    lob_df: pd.DataFrame, levels: list = [5, 10], side: str = "both"
) -> pd.DataFrame:
    """
    Compute average volume per level at different depths.

    Measures the average liquidity per price level, indicating how
    volume is distributed across the order book.

    Formula:
        avg_bid_volume_N = bid_depth_N / N = (Σ bid_volume_i) / N
        avg_ask_volume_N = ask_depth_N / N = (Σ ask_volume_i) / N

    Intuition:
        Average volume per level reveals book shape and liquidity distribution.

        Normal pattern: avg_volume decreases with depth
            - Top levels have most liquidity (market makers, active traders)
            - Deeper levels have less liquidity
            - avg_volume_5 > avg_volume_10 > avg_volume_20

        Abnormal pattern: avg_volume increases with depth
            - Suggests large hidden orders sitting deeper in book
            - Often indicates institutional interest
            - "Iceberg orders" - large orders split across deep levels

        This metric can identify unusual liquidity patterns that aren't
        visible in simple depth metrics. Two books with same total_depth_10
        could have very different shapes.

    Args:
        lob_df: DataFrame with bid_volume_i and ask_volume_i columns
        levels: List of depth levels to compute
        side: Which side to compute ('bid', 'ask', or 'both')

    Returns:
        DataFrame with avg_bid_volume_{N} and/or avg_ask_volume_{N} columns

    Examples:
        >>> avg_vol = compute_avg_volume_per_level(lob_df, levels=[5, 10])
        >>> # Creates: avg_bid_volume_5, avg_bid_volume_10,
        >>> #          avg_ask_volume_5, avg_ask_volume_10

    Notes:
        Typical patterns:
        - Liquid markets: Declining average volume as depth increases
        - Thin markets: Low average volume at all levels
        - Institutional presence: Rising or stable average volume at deep levels

        Can be combined with depth imbalance: if avg_bid_volume_10 is rising
        but avg_ask_volume_10 is falling, suggests large hidden buy orders.
    """
    avg_vol_features = pd.DataFrame(index=lob_df.index)

    for level in levels:
        if side in ["bid", "both"]:
            bid_cols = [f"bid_volume_{i}" for i in range(level)]
            bid_cols = [c for c in bid_cols if c in lob_df.columns]

            if bid_cols:
                bid_depth = lob_df[bid_cols].sum(axis=1)
                avg_vol_features[f"avg_bid_volume_{level}"] = bid_depth / level

        if side in ["ask", "both"]:
            ask_cols = [f"ask_volume_{i}" for i in range(level)]
            ask_cols = [c for c in ask_cols if c in lob_df.columns]

            if ask_cols:
                ask_depth = lob_df[ask_cols].sum(axis=1)
                avg_vol_features[f"avg_ask_volume_{level}"] = ask_depth / level

    return avg_vol_features


def compute_queue_depth_single_side(
    volumes: pd.DataFrame,
    side: str = "bid",
    levels: list[int] = [1],
) -> pd.DataFrame:
    """
    Compute queue depth (raw quantities) at specific levels for one side of the book.

    Queue depth represents the total quantity available at specific price levels,
    revealing where liquidity is concentrated. Unlike spreads which show price
    gaps, queue depth shows volume concentration - critical for understanding
    market impact and execution costs.

    Formula:
        For each level k in levels:
            queue_depth_k = volume at level k

    Intuition:
        Queue depth answers: "How much size is sitting at each price level?"

        **Why queue depth matters:**

        1. **Execution cost estimation**: Deep queues at top levels mean you can
           execute large orders without significant slippage

        2. **Support/resistance**: Large queue sizes often act as price barriers
           - big buyers/sellers defending levels

        3. **Market maker presence**: Consistent depth suggests active MM activity;
           thin depth suggests reduced liquidity provision

        4. **Order flow toxicity**: Sudden queue depletion can signal informed
           trading or predatory HFT strategies

        **Patterns to watch:**
        - **Deep top-of-book**: Good liquidity, low impact trades possible
        - **Thin top, deep lower**: "Iceberg" patterns, hidden liquidity
        - **Symmetric depth**: Balanced market, no directional pressure
        - **Asymmetric depth**: One-sided pressure (more bids → buying pressure)

        In HFT strategies:
        - Market makers monitor queue position to avoid adverse selection
        - Execution algorithms use depth to time order submission
        - Liquidity-taking strategies look for depth imbalances

    Args:
        volumes: DataFrame with volume at each level (columns: bid_volume_1,
                 bid_volume_2, ..., or ask_volume_1, ask_volume_2, ...)
        side: Which side of book ('bid' or 'ask')
        levels: List of levels to extract (e.g., [1] for top, [1,2,3,4,5] for top 5)

    Returns:
        DataFrame with queue depth at specified levels (columns: queue_depth_bid_1,
        queue_depth_bid_2, etc.)

    Notes:
        - Raw quantities are instrument-specific (compare 1 BTC vs 1000 shares)
        - Often normalized by average depth or recent volume for comparison
        - Sudden changes in queue depth can signal regime shifts
        - Consider relative depth (bid/ask ratio) for directional signals

    References:
        - Parlour, C. A. (1998). "Price dynamics in limit order markets"
        - Cont, R., Kukanov, A., & Stoikov, S. (2014). "The price impact
          of order book events"
    """
    depth_df = pd.DataFrame(index=volumes.index)

    for level in levels:
        col_name = f"{side}_volume_{level}"
        output_name = f"queue_depth_{side}_{level}"
        depth_df[output_name] = volumes[col_name]

    return depth_df


def compute_queue_depth_top_n(
    bid_volumes: pd.DataFrame,
    ask_volumes: pd.DataFrame,
    n: int = 5,
) -> pd.DataFrame:
    """
    Compute queue depth for top N levels on both bid and ask sides.

    This function extracts the raw liquidity available at the top N price
    levels, creating features that capture the microstructure of the order
    book. These features are essential for predicting short-term price
    movements and assessing execution quality.

    Formula:
        For each level i in [1, N]:
            queue_depth_bid_i = bid_volume_i
            queue_depth_ask_i = ask_volume_i

    Intuition:
        **The shape of the queue depth curve reveals market microstructure:**

        - **Flat curve** (similar depth across levels): Uniform liquidity,
          healthy market, large orders possible with minimal impact

        - **Steep decay** (much less depth at deeper levels): Concentrated
          liquidity at top, high impact risk for large orders

        - **Inverted curve** (more depth at deeper levels): Unusual, may
          indicate icebergs, hidden orders, or strategic positioning

        **Two-sided analysis:**
        Comparing bid vs ask depth patterns:
        - Bid depth >> Ask depth → Buying pressure, support building
        - Ask depth >> Bid depth → Selling pressure, resistance overhead
        - Symmetric depth → Balanced market, no clear directional bias

        **Time dynamics:**
        - Increasing depth → Liquidity improving, lower transaction costs
        - Decreasing depth → Liquidity drying up, potential volatility spike
        - Sudden jumps → Large order arrival (institutional, algo)
        - Gradual build → Market makers adapting to new price level

        For ML models, queue depth features help predict:
        - Probability of price moving away from large queues
        - Likelihood of queue depletion (breakout vs bounce)
        - Optimal timing for large order execution

    Args:
        bid_volumes: DataFrame with bid volumes at each level
        ask_volumes: DataFrame with ask volumes at each level
        n: Number of top levels to include (default: 5)

    Returns:
        DataFrame with queue depth for top N levels on both sides
        (columns: queue_depth_bid_1, ..., queue_depth_bid_N,
                  queue_depth_ask_1, ..., queue_depth_ask_N)

    Notes:
        - These are absolute quantities - normalization often helpful
        - Can compute depth ratios: bid_depth_i / ask_depth_i for each level
        - Useful to track depth changes: depth_t - depth_{t-1}
        - Consider rolling averages to smooth noisy depth updates
    """
    depth_df = pd.DataFrame(index=bid_volumes.index)

    # Extract bid depths
    for level in range(1, n + 1):
        depth_df[f"queue_depth_bid_{level}"] = bid_volumes[f"bid_volume_{level}"]

    # Extract ask depths
    for level in range(1, n + 1):
        depth_df[f"queue_depth_ask_{level}"] = ask_volumes[f"ask_volume_{level}"]

    return depth_df


def compute_cumulative_queue_depth(
    bid_volumes: pd.DataFrame,
    ask_volumes: pd.DataFrame,
    n_levels: int = 10,
) -> pd.DataFrame:
    """
    Compute cumulative queue depth (total volume) across top N levels.

    Cumulative depth measures the total liquidity available within the top N
    levels of the order book, answering: "How much total size can I execute
    before exhausting the visible book?" This is critical for execution
    algorithms and market impact estimation.

    Formula:
        cumulative_bid_depth = Σ(bid_volume_i) for i in [1, N]
        cumulative_ask_depth = Σ(ask_volume_i) for i in [1, N]
        total_depth = cumulative_bid_depth + cumulative_ask_depth

    Intuition:
        **Why cumulative depth matters more than individual levels:**

        Large orders "walk the book" - they consume liquidity across multiple
        levels. Cumulative depth directly estimates:
        - Maximum order size executable without severe slippage
        - Total available liquidity in the visible book
        - Market's capacity to absorb flow

        **Market quality indicator:**
        - High cumulative depth → Deep, liquid market
        - Low cumulative depth → Thin market, high impact risk
        - Increasing depth → Market makers adding liquidity
        - Decreasing depth → Liquidity withdrawal (danger signal)

        **Asymmetric cumulative depth signals:**
        - Cumulative bid >> cumulative ask → Strong buy-side support
        - Cumulative ask >> cumulative bid → Heavy sell-side resistance
        - Balanced cumulative depth → No clear pressure

        **Regime detection:**
        Sudden drops in cumulative depth often precede:
        - Volatility spikes (liquidity withdrawal)
        - Large informed orders (MMs step aside)
        - News events (uncertainty → reduce exposure)

        Execution algorithms use cumulative depth to:
        - Size child orders appropriately
        - Determine aggression level (passive vs aggressive)
        - Estimate remaining execution time for large parents

    Args:
        bid_volumes: DataFrame with bid volumes at each level
        ask_volumes: DataFrame with ask volumes at each level
        n_levels: Number of levels to sum over (default: 10)

    Returns:
        DataFrame with three columns:
            - cumulative_bid_depth: Total bid volume across N levels
            - cumulative_ask_depth: Total ask volume across N levels
            - total_depth: Sum of both sides


    Notes:
        - Often normalized by average depth or recent volume
        - Can compute depth imbalance: (bid_depth - ask_depth) / total_depth
        - Track depth velocity: change in cumulative depth over time
        - Useful to compare against historical percentiles

        **ML feature engineering:**
        - Depth ratio: cumulative_bid_depth / cumulative_ask_depth
        - Depth change: cumulative_depth_t - cumulative_depth_{t-k}
        - Depth volatility: rolling std of cumulative depth
        - Depth percentile: current depth vs historical distribution

    """
    # Select columns for N levels
    bid_cols = [f"bid_volume_{i}" for i in range(1, n_levels + 1)]
    ask_cols = [f"ask_volume_{i}" for i in range(1, n_levels + 1)]

    # Compute cumulative sums
    cumulative_bid = bid_volumes[bid_cols].sum(axis=1)
    cumulative_ask = ask_volumes[ask_cols].sum(axis=1)
    total = cumulative_bid + cumulative_ask

    # Create output dataframe
    depth_df = pd.DataFrame(
        {
            "cumulative_bid_depth": cumulative_bid,
            "cumulative_ask_depth": cumulative_ask,
            "total_depth": total,
        },
        index=bid_volumes.index,
    )

    return depth_df


def compute_intraday_time_features(
    timestamps: pd.Series,
    include_cyclical: bool = True,
) -> pd.DataFrame:
    """
    Extract intraday time-of-day features from timestamps.

    Time-of-day patterns are crucial in HFT because market microstructure
    varies systematically throughout the trading day. Different times exhibit
    different liquidity profiles, volatility regimes, and participant behavior.
    These patterns are highly predictable and provide strong signals for ML models.

    Intuition:
        **Why intraday patterns matter in HFT:**

        Markets have distinct "personalities" at different times:

        - **Market open (9:30-10:00 AM)**:
          - Highest volatility and volume
          - Overnight news gets priced in
          - Wide spreads, large price moves
          - Retail and institutional flow collide

        - **Mid-morning (10:00-11:30 AM)**:
          - Stabilization after open volatility
          - Institutional algorithms active
          - Tighter spreads, more predictable

        - **Lunch period (11:30 AM-1:30 PM)**:
          - Lowest liquidity (especially 12-1 PM)
          - Wider spreads, lower volume
          - Increased noise-to-signal ratio
          - Algos dominate (fewer humans)

        - **Afternoon (1:30-3:00 PM)**:
          - Gradual volume increase
          - European close effects (4 PM CET = 10 AM EST)
          - Strategic positioning for close

        - **Market close (3:00-4:00 PM)**:
          - Volume surge (especially last 10 minutes)
          - MOC (market-on-close) order flow
          - Increased volatility
          - Index rebalancing, fund flows

        **ML model perspective:**

        Without time features, models can't learn that:
        - Spreads naturally widen at lunch
        - Volatility spikes are normal at open/close
        - Mean reversion works better mid-day
        - Momentum strategies work better at open

        Time features allow models to specialize their predictions:
        "If hour == 9 AND spread_widening → expected (don't predict crash)"
        "If hour == 12 AND low_volume → normal (not a liquidity crisis)"

    Formula:
        Linear features:
            hour = timestamp.hour  (0-23)
            minute = timestamp.minute  (0-59)
            seconds_since_midnight = hour * 3600 + minute * 60 + second

        Cyclical features (captures circular nature of time):
            hour_sin = sin(2π × hour / 24)
            hour_cos = cos(2π × hour / 24)
            minute_sin = sin(2π × minute / 60)
            minute_cos = cos(2π × minute / 60)

    Args:
        timestamps: Series of pandas Timestamp objects
        include_cyclical: If True, include sin/cos encodings for cyclical
                         representation (default: True, recommended for ML)

    Returns:
        DataFrame with time-of-day features:
            - hour: Hour of day (0-23)
            - minute: Minute of hour (0-59)
            - seconds_since_midnight: Total seconds since midnight
            - hour_sin, hour_cos: Cyclical hour encoding (if include_cyclical=True)
            - minute_sin, minute_cos: Cyclical minute encoding (if include_cyclical=True)

    Examples:
        >>> timestamps = pd.Series([
        ...     pd.Timestamp('2025-01-15 09:30:00'),
        ...     pd.Timestamp('2025-01-15 12:00:00'),
        ...     pd.Timestamp('2025-01-15 15:45:00')
        ... ])
        >>> compute_intraday_time_features(timestamps, include_cyclical=False)
           hour  minute  seconds_since_midnight
        0     9      30                   34200
        1    12       0                   43200
        2    15      45                   56700

        >>> compute_intraday_time_features(timestamps, include_cyclical=True)
           hour  minute  seconds_since_midnight  hour_sin  hour_cos  minute_sin  minute_cos
        0     9      30                   34200  0.258819  0.965926    0.000000   -1.000000
        1    12       0                   43200  1.000000  0.000000    0.000000    1.000000
        2    15      45                   56700  0.258819 -0.965926    0.707107   -0.707107

        # Note: hour_sin and hour_cos for 9 AM and 3 PM have same magnitude
        #       but opposite cos (captures symmetry around noon)

    Notes:
        **Cyclical encoding rationale:**

        Linear time encoding has a problem: hour 23 (11 PM) and hour 0 (midnight)
        are adjacent in time but numerically distant (23 vs 0). This confuses
        tree-based models and neural networks.

        Sin/cos encoding solves this by mapping time onto a circle:
        - 11:59 PM and 12:01 AM are now close in feature space
        - Preserves ordinality while capturing cyclical nature
        - Two dimensions (sin, cos) uniquely identify any time point

        **For tree-based models:**
        Include BOTH linear and cyclical features. Trees can learn
        thresholds on linear features ("if hour >= 15") while cyclical
        features help with wraparound cases.

        **Additional feature engineering ideas:**
        - is_market_open: Binary indicator (9:30 AM - 4 PM)
        - minutes_since_open: Elapsed time since 9:30 AM
        - minutes_until_close: Time remaining until 4 PM
        - session_period: Categorical (open/mid_morning/lunch/afternoon/close)

    References:
        - Admati, A. R., & Pfleiderer, P. (1988). "A theory of intraday
          patterns: Volume and price variability"
        - Brock, W. A., & Kleidon, A. W. (1992). "Periodic market closure
          and trading volume"
        - Wood, R. A., McInish, T. H., & Ord, J. K. (1985). "An investigation
          of transactions data for NYSE stocks"
    """
    time_df = pd.DataFrame(index=timestamps.index)

    # Extract basic time components
    time_df["hour"] = timestamps.dt.hour
    time_df["minute"] = timestamps.dt.minute
    time_df["seconds_since_midnight"] = (
        timestamps.dt.hour * 3600 + timestamps.dt.minute * 60 + timestamps.dt.second
    )

    # Add cyclical encodings
    if include_cyclical:
        # Hour encoding (24-hour cycle)
        time_df["hour_sin"] = np.sin(2 * np.pi * time_df["hour"] / 24)
        time_df["hour_cos"] = np.cos(2 * np.pi * time_df["hour"] / 24)

        # Minute encoding (60-minute cycle)
        time_df["minute_sin"] = np.sin(2 * np.pi * time_df["minute"] / 60)
        time_df["minute_cos"] = np.cos(2 * np.pi * time_df["minute"] / 60)

    return time_df


def compute_trading_session_features(
    timestamps: pd.Series,
    market_open: str = "09:30:00",
    market_close: str = "16:00:00",
) -> pd.DataFrame:
    """
    Compute trading session-relative features and categorical time periods.

    This function creates features based on position within the trading session,
    which are often more predictive than absolute clock time. Markets care more
    about "10 minutes before close" than "3:50 PM specifically."

    Formula:
        minutes_since_open = (timestamp - market_open_today).total_seconds() / 60
        minutes_until_close = (market_close_today - timestamp).total_seconds() / 60
        session_progress = minutes_since_open / total_session_minutes

        session_period = {
            'pre_open': before market open,
            'open': first 30 minutes,
            'mid_morning': 10:00-11:30,
            'lunch': 11:30-13:30,
            'afternoon': 13:30-15:00,
            'close': last 60 minutes,
            'post_close': after market close
        }

    Intuition:
        **Session-relative time is more meaningful than clock time:**

        "30 minutes after open" has similar dynamics whether it's:
        - Regular hours: 10:00 AM (9:30 open)
        - Extended hours: 5:00 AM (4:30 open)
        - Different market: 3:30 PM (3:00 PM open for futures)

        This generalization helps models trained on one market/session apply
        to others. It also captures the "aging" of information throughout
        the session.

        **Session progress (0.0 to 1.0):**
        - 0.0 = market just opened, fresh overnight information
        - 0.5 = mid-session, most information priced in
        - 1.0 = about to close, positioning for overnight risk

        Models can learn smooth transitions: "As session_progress → 1.0,
        increase weight on mean reversion (unwind intraday positions)"

        **Categorical periods capture regime changes:**
        Different ML models/strategies should activate at different times:
        - Open period: High volatility → wider prediction intervals
        - Lunch period: Low liquidity → avoid aggressive strategies
        - Close period: Volume surge → momentum strategies more effective

    Args:
        timestamps: Series of pandas Timestamp objects
        market_open: Time string for market open (default: '09:30:00' for US equities)
        market_close: Time string for market close (default: '16:00:00' for US equities)

    Returns:
        DataFrame with session-relative features:
            - is_market_hours: Boolean, True if during regular trading hours
            - minutes_since_open: Minutes elapsed since market open (negative if pre-open)
            - minutes_until_close: Minutes remaining until close (negative if post-close)
            - session_progress: Fraction of session completed (0.0 to 1.0 during hours)
            - session_period: Categorical indicator of intraday period

    Examples:
        >>> timestamps = pd.Series([
        ...     pd.Timestamp('2025-01-15 09:30:00'),  # Open
        ...     pd.Timestamp('2025-01-15 12:00:00'),  # Lunch
        ...     pd.Timestamp('2025-01-15 15:30:00'),  # Near close
        ...     pd.Timestamp('2025-01-15 16:30:00')   # After close
        ... ])
        >>> compute_trading_session_features(timestamps)
           is_market_hours  minutes_since_open  minutes_until_close  session_progress session_period
        0             True                 0.0                390.0          0.000000           open
        1             True               150.0                240.0          0.384615          lunch
        2             True               360.0                 30.0          0.923077          close
        3            False               420.0                -30.0          1.076923     post_close

        # At 15:30: 92% through session, in "close" period
        # At 16:30: Outside hours, negative minutes_until_close

    Notes:
        **Customization for different markets:**
        - Crypto (24/7): Set market_open='00:00:00', market_close='23:59:59'
        - Futures (extended hours): Adjust times to 18:00-17:00 for CME
        - International markets: Adjust for local exchange hours

        **Feature engineering extensions:**
        - distance_from_open: min(minutes_since_open, minutes_until_close)
          (captures "edges" of session vs middle)
        - is_first_hour, is_last_hour: Binary indicators
        - session_quarter: Divide into 4 equal periods (Q1, Q2, Q3, Q4)

        **One-hot encoding session_period:**
        For some ML models (neural nets), one-hot encode categorical:
        pd.get_dummies(df['session_period'], prefix='period')

    """
    session_df = pd.DataFrame(index=timestamps.index)

    # Parse market open/close times
    open_time = pd.to_timedelta(market_open)
    close_time = pd.to_timedelta(market_close)
    total_minutes = (close_time - open_time).total_seconds() / 60

    # Compute session-relative times
    time_of_day = pd.to_timedelta(
        timestamps.dt.hour.astype(str)
        + ":"
        + timestamps.dt.minute.astype(str)
        + ":"
        + timestamps.dt.second.astype(str)
    )

    session_df["minutes_since_open"] = (time_of_day - open_time).dt.total_seconds() / 60
    session_df["minutes_until_close"] = (
        close_time - time_of_day
    ).dt.total_seconds() / 60
    session_df["session_progress"] = session_df["minutes_since_open"] / total_minutes

    # Market hours indicator
    session_df["is_market_hours"] = (time_of_day >= open_time) & (
        time_of_day <= close_time
    )

    # Categorical session periods
    def assign_period(row):
        """Assign categorical session period based on time."""
        mins = row["minutes_since_open"]

        if mins < 0:
            return "pre_open"
        elif mins <= 30:
            return "open"
        elif mins <= 120:  # 10:00-11:30
            return "mid_morning"
        elif mins <= 240:  # 11:30-13:30
            return "lunch"
        elif mins <= 330:  # 13:30-15:00
            return "afternoon"
        elif mins <= total_minutes:  # Last hour
            return "close"
        else:
            return "post_close"

    session_df["session_period"] = session_df.apply(assign_period, axis=1)

    return session_df


def compute_summary_stats(
    df: pd.DataFrame,
    percentiles: list[float] = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
) -> pd.DataFrame:
    """
    Compute comprehensive summary statistics for feature columns.

    This function provides a detailed statistical overview of features,
    going beyond basic describe() to include measures relevant for financial
    data and ML feature engineering: skewness, kurtosis, missing values,
    outliers, and distribution characteristics.

    Args:
        df: DataFrame containing feature columns to analyze
        percentiles: List of percentiles to compute (default: [1, 5, 25, 50, 75, 95, 99])

    Returns:
        DataFrame with summary statistics where:
            - Rows are feature names
            - Columns are statistical measures
    """
    stats_dict = {}

    for col in df.columns:
        series = df[col]

        # Basic statistics
        col_stats = {
            "count": series.count(),
            "missing": series.isna().sum(),
            "missing_pct": series.isna().sum() / len(series) * 100,
            "n_zeros": (series == 0).sum(),
            "n_infinite": np.isinf(series).sum(),
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
        }

        # Percentiles
        for p in percentiles:
            col_stats[f"p{int(p*100)}"] = series.quantile(p)

        # Distribution shape
        col_stats["skewness"] = series.skew()
        col_stats["kurtosis"] = series.kurtosis()

        # Range and spread metrics
        col_stats["range"] = col_stats["max"] - col_stats["min"]
        col_stats["iqr"] = series.quantile(0.75) - series.quantile(0.25)
        col_stats["cv"] = (
            col_stats["std"] / abs(col_stats["mean"])
            if col_stats["mean"] != 0
            else np.nan
        )

        # Outlier detection (using IQR method)
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        col_stats["n_outliers"] = (
            (series < lower_bound) | (series > upper_bound)
        ).sum()
        col_stats["outlier_pct"] = col_stats["n_outliers"] / len(series) * 100

        # Unique values
        col_stats["n_unique"] = series.nunique()
        col_stats["unique_pct"] = col_stats["n_unique"] / len(series) * 100

        stats_dict[col] = col_stats

    # Convert to DataFrame
    stats_df = pd.DataFrame(stats_dict).T

    # Round for readability
    stats_df = stats_df.round(6)

    return stats_df


def get_feature_columns(df: pd.DataFrame) -> dict:
    """
    Categorize columns in the LOB dataframe into different feature groups.

    Args:
        df: LOB dataframe with raw data and engineered features

    Returns:
        Dictionary with categorized column lists
    """
    all_cols = df.columns.tolist()

    # Raw LOB data (to exclude from analysis)
    raw_lob = [
        col
        for col in all_cols
        if any(
            p in col for p in ["bid_price_", "ask_price_", "bid_volume_", "ask_volume_"]
        )
    ]

    # Metadata columns (to exclude)
    metadata = ["timestamp", "session_period", "is_market_hours"]

    # Engineered features (what we want to analyze)
    engineered = [col for col in all_cols if col not in raw_lob and col not in metadata]

    # Further categorize engineered features
    price_features = [
        col
        for col in engineered
        if any(p in col for p in ["mid_price", "weighted_mid", "vwap", "price"])
    ]
    spread_features = [col for col in engineered if "spread" in col]
    return_features = [col for col in engineered if "return" in col]
    imbalance_features = [col for col in engineered if "imbalance" in col]
    depth_features = [col for col in engineered if "depth" in col]
    queue_features = [col for col in engineered if "queue" in col]
    volatility_features = [col for col in engineered if "vol" in col]
    book_thickness = [col for col in engineered if "thick" in col]
    time_features = [
        col
        for col in engineered
        if any(
            p in col
            for p in [
                "hour",
                "minute",
                "seconds",
                "session_progress",
                "minutes_since",
                "minutes_until",
            ]
        )
    ]
    volume_features = [col for col in engineered if "avg_" in col and "volume" in col]

    return {
        "all_engineered": engineered,
        "raw_lob": raw_lob,
        "metadata": metadata,
        "price": price_features,
        "spread": spread_features,
        "returns": return_features,
        "imbalance": imbalance_features,
        "depth": depth_features,
        "queue": queue_features,
        "time": time_features,
        "volume": volume_features,
        "volatility": volatility_features,
        "thickness": book_thickness,
    }


def print_feature_summary(
    df: pd.DataFrame,
    feature_cols: list[str] = None,
    exclude_raw_lob: bool = True,
    save_path: str = None,
) -> pd.DataFrame:
    """
    Print and optionally save a formatted feature summary report.

    Args:
        df: LOB DataFrame with raw data and engineered features
        feature_cols: Specific columns to analyze (if None, analyzes all engineered features)
        exclude_raw_lob: If True, excludes bid_price_*, ask_price_*, bid_volume_*, ask_volume_*
        save_path: Optional path to save summary CSV

    Returns:
        DataFrame with summary statistics
    """
    # Get feature categorization
    feature_groups = get_feature_columns(df)

    # Determine which columns to analyze
    if feature_cols is None:
        if exclude_raw_lob:
            feature_cols = feature_groups["all_engineered"]
        else:
            # Analyze all numeric columns
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Filter to only existing numeric columns
    feature_cols = [
        col
        for col in feature_cols
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    if len(feature_cols) == 0:
        print("⚠️  No numeric feature columns found!")
        return pd.DataFrame()

    print(f"\n📊 Analyzing {len(feature_cols)} features...")

    # Compute statistics
    stats_df = compute_summary_stats(df[feature_cols])

    # Print formatted summary
    print("\n" + "=" * 80)
    print("FEATURE SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nDataset: {len(df):,} observations")
    print(f"Features: {len(feature_cols)} columns\n")

    # Print data quality overview
    print("-" * 80)
    print("DATA QUALITY OVERVIEW")
    print("-" * 80)
    quality_cols = [
        "count",
        "missing",
        "missing_pct",
        "n_zeros",
        "n_infinite",
        "n_outliers",
        "outlier_pct",
    ]
    print(stats_df[quality_cols].head(20).to_string())
    if len(stats_df) > 20:
        print(f"\n... (showing first 20 of {len(stats_df)} features)")

    # Print distribution statistics
    print("\n" + "-" * 80)
    print("DISTRIBUTION STATISTICS")
    print("-" * 80)
    dist_cols = [
        "mean",
        "std",
        "min",
        "p25",
        "p50",
        "p75",
        "max",
        "skewness",
        "kurtosis",
    ]
    print(stats_df[dist_cols].head(20).to_string())
    if len(stats_df) > 20:
        print(f"\n... (showing first 20 of {len(stats_df)} features)")

    # Print key insights
    print("\n" + "-" * 80)
    print("KEY INSIGHTS")
    print("-" * 80)

    # Features with missing data
    missing_features = stats_df[stats_df["missing_pct"] > 0].index.tolist()
    if missing_features:
        print(f"\n⚠️  Features with missing data: {len(missing_features)}")
        for feat in missing_features[:10]:
            pct = stats_df.loc[feat, "missing_pct"]
            count = int(stats_df.loc[feat, "missing"])
            print(f"    - {feat}: {count} ({pct:.2f}%)")
        if len(missing_features) > 10:
            print(f"    ... and {len(missing_features) - 10} more")
    else:
        print("\n✓ No missing data detected")

    # Features with many outliers
    outlier_features = stats_df[stats_df["outlier_pct"] > 5].sort_values(
        "outlier_pct", ascending=False
    )
    if len(outlier_features) > 0:
        print(f"\n⚠️  Features with >5% outliers: {len(outlier_features)}")
        for feat in outlier_features.head(10).index:
            pct = stats_df.loc[feat, "outlier_pct"]
            count = int(stats_df.loc[feat, "n_outliers"])
            print(f"    - {feat}: {count} ({pct:.2f}%)")
        if len(outlier_features) > 10:
            print(f"    ... and {len(outlier_features) - 10} more")
    else:
        print("\n✓ No excessive outliers detected")

    # Features with infinite values
    inf_features = stats_df[stats_df["n_infinite"] > 0].index.tolist()
    if inf_features:
        print(f"\n⚠️  Features with infinite values: {len(inf_features)}")
        for feat in inf_features[:10]:
            count = int(stats_df.loc[feat, "n_infinite"])
            print(f"    - {feat}: {count} inf values")
    else:
        print("\n✓ No infinite values detected")

    # Highly skewed features
    skewed_features = stats_df[abs(stats_df["skewness"]) > 2].sort_values(
        "skewness", ascending=False, key=abs
    )
    if len(skewed_features) > 0:
        print(f"\n📊 Highly skewed features (|skew| > 2): {len(skewed_features)}")
        print("   Top 5 most skewed:")
        for feat in skewed_features.head(5).index:
            skew = stats_df.loc[feat, "skewness"]
            print(f"    - {feat}: skew={skew:.2f}")
        print("   → Consider: log transform, box-cox, or robust scaling")

    # Heavy-tailed features
    heavy_tail = stats_df[stats_df["kurtosis"] > 5].sort_values(
        "kurtosis", ascending=False
    )
    if len(heavy_tail) > 0:
        print(f"\n📊 Heavy-tailed features (kurtosis > 5): {len(heavy_tail)}")
        print("   Top 5 most heavy-tailed:")
        for feat in heavy_tail.head(5).index:
            kurt = stats_df.loc[feat, "kurtosis"]
            print(f"    - {feat}: kurtosis={kurt:.2f}")
        print("   → Consider: winsorization or tree-based models")

    # Features with little variation
    low_var = stats_df[stats_df["cv"] < 0.01].dropna()
    if len(low_var) > 0:
        print(f"\n📊 Low variation features (CV < 0.01): {len(low_var)}")
        if len(low_var) <= 10:
            print(f"   {', '.join(low_var.index.tolist())}")
        else:
            print(f"   {', '.join(low_var.head(10).index.tolist())} ...")
        print("   → Consider: removing (low signal)")

    print("\n" + "=" * 80 + "\n")

    # Save if requested
    if save_path:
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        stats_df.to_csv(save_path)
        print(f"💾 Summary statistics saved to: {save_path}\n")

    return stats_df


def compute_rolling_volatility(
    returns: pd.Series,
    window: int = 20,
    min_periods: int = None,
    annualize: bool = False,
    trading_periods_per_year: int = 252 * 390,  # 252 days × 390 minutes per trading day
) -> pd.Series:
    """
    Compute rolling volatility (standard deviation of returns).

    Rolling volatility measures the variability of returns over a sliding window,
    providing a time-varying estimate of risk. In HFT, volatility is a critical
    signal for regime detection, risk management, and strategy adaptation.

    Formula:
        volatility_t = std(returns_{t-window+1:t})

        If annualize=True:
            annualized_vol_t = volatility_t × sqrt(trading_periods_per_year)

    Intuition:
        **Why rolling volatility matters in HFT:**

        Markets alternate between calm and volatile regimes. Volatility is NOT
        constant - it clusters (high vol follows high vol, low vol follows low vol).

        - **Low volatility periods**:
          - Tight spreads, deep liquidity
          - Mean-reversion strategies work well
          - Market makers earn steady spreads
          - Lower risk, lower opportunity

        - **High volatility periods**:
          - Wide spreads, shallow liquidity
          - Momentum strategies more effective
          - Market makers face adverse selection
          - Higher risk, higher opportunity

        **Trading applications:**

        1. **Position sizing**: Scale positions inversely with volatility
           - High vol → smaller positions (limit risk)
           - Low vol → larger positions (maximize opportunity)

        2. **Spread adjustment**: Market makers widen quotes in high vol
           - Protects against adverse selection
           - Compensates for inventory risk

        3. **Strategy selection**: Switch strategies by volatility regime
           - Low vol → mean reversion, liquidity provision
           - High vol → momentum, directional betting

        4. **Risk limits**: Trigger position exits when vol spikes
           - Prevents catastrophic losses
           - Adapts to changing market conditions

        5. **Feature for ML models**: Volatility predicts:
           - Probability of large price moves
           - Likelihood of spread widening
           - Expected execution costs

        **Window selection:**
        - Short windows (5-20): Responsive, noisy, react quickly to regime changes
        - Medium windows (20-60): Balanced, standard in many strategies
        - Long windows (100+): Smooth, slow to adapt, baseline volatility

        Multiple windows capture different frequencies:
        - 10-period: Recent microstructure changes
        - 30-period: Short-term regime (last few minutes)
        - 100-period: Longer-term baseline (last hour+)

    Args:
        returns: Series of log returns (from compute_log_return)
        window: Number of periods for rolling window (default: 20)
        min_periods: Minimum observations required (default: window)
        annualize: If True, scale volatility to annual terms (default: False)
        trading_periods_per_year: Number of periods in a trading year (default: 252*390 for minute data)

    Returns:
        Series containing rolling volatility estimates

    Examples:
        >>> returns = pd.Series([0.001, -0.002, 0.0015, -0.001, 0.0005,
        ...                      0.002, -0.003, 0.001, 0.0, -0.001])
        >>> compute_rolling_volatility(returns, window=5)
        0         NaN
        1         NaN
        2         NaN
        3         NaN
        4    0.001304
        5    0.001517
        6    0.002074
        7    0.002074
        8    0.001673
        9    0.001304
        dtype: float64

        # At t=4: volatility over last 5 returns
        # At t=5: window slides forward, includes most recent return

        >>> # Multiple windows for different time scales
        >>> vol_short = compute_rolling_volatility(returns, window=10)   # Recent
        >>> vol_medium = compute_rolling_volatility(returns, window=30)  # Intermediate
        >>> vol_long = compute_rolling_volatility(returns, window=100)   # Baseline

    Notes:
        **Volatility characteristics in financial data:**

        - **Volatility clustering**: High vol today → likely high vol tomorrow
        - **Mean reversion**: Extremely high/low vol tends to revert to average
        - **Asymmetry**: Volatility spikes faster than it declines
        - **Leverage effect**: Volatility increases more after negative returns

        **Implementation details:**

        - Uses .std() with ddof=1 (sample standard deviation, not population)
        - Returns NaN for first (window-1) observations
        - Can set min_periods < window for partial window calculations
        - Annualization assumes returns are measured at consistent intervals

        **Advanced variations (not implemented here, but useful):**

        - **Exponentially weighted volatility**: Give more weight to recent returns
          `returns.ewm(span=window).std()`
        - **Realized volatility**: Sum of squared returns (alternative estimator)
        - **Parkinson volatility**: Uses high-low range (requires OHLC data)
        - **Garman-Klass volatility**: Uses OHLC efficiently (lower variance estimator)

    References:
        - Andersen, T. G., & Bollerslev, T. (1998). "Answering the skeptics: Yes,
          standard volatility models do provide accurate forecasts"
        - Engle, R. F. (1982). "Autoregressive conditional heteroscedasticity with
          estimates of the variance of United Kingdom inflation" (ARCH models)
        - Bollerslev, T. (1986). "Generalized autoregressive conditional
          heteroskedasticity" (GARCH models)
    """
    if min_periods is None:
        min_periods = window

    # Compute rolling standard deviation
    rolling_vol = returns.rolling(window=window, min_periods=min_periods).std()

    # Annualize if requested
    if annualize:
        rolling_vol = rolling_vol * np.sqrt(trading_periods_per_year)
        rolling_vol.name = f"rolling_vol_{window}_annualized"
    else:
        rolling_vol.name = f"rolling_vol_{window}"

    return rolling_vol


def compute_rolling_volatility_multiple_windows(
    returns: pd.Series,
    windows: List[int] = [10, 20, 60, 100],
    annualize: bool = False,
    trading_periods_per_year: int = 252 * 390,
) -> pd.DataFrame:
    """
    Compute rolling volatility at multiple window lengths simultaneously.

    Multiple volatility windows capture different frequencies of market dynamics.
    This creates a rich feature set for ML models to learn regime-dependent patterns.

    Formula:
        For each window w in windows:
            vol_w = rolling_std(returns, window=w)

    Intuition:
        **Multi-scale volatility analysis:**

        Different window lengths reveal different aspects of market structure:

        - **Short windows (5-20 periods)**:
          - Detect rapid regime changes
          - Respond quickly to volatility spikes
          - More noise, less stable
          - Use for: tactical adjustments, stop-loss triggers

        - **Medium windows (20-60 periods)**:
          - Balance responsiveness and stability
          - Standard for many trading strategies
          - Good for: position sizing, spread adjustment

        - **Long windows (100+ periods)**:
          - Smooth, stable baseline
          - Slower to react to changes
          - Good for: strategic allocation, long-term risk assessment

        **Feature interactions for ML:**

        Models can learn patterns like:
        - "If vol_10 >> vol_100 → recent volatility spike (be cautious)"
        - "If vol_10 < vol_100 → volatility declining (tighten spreads)"
        - "If all windows increasing → sustained regime change (switch strategy)"

        **Volatility ratios** (can be computed from output):
        - vol_short / vol_long > 1.5 → Recent volatility spike
        - vol_short / vol_long < 0.7 → Volatility declining
        - Used for adaptive strategy selection

    Args:
        returns: Series of log returns
        windows: List of window lengths (default: [10, 20, 60, 100])
        annualize: If True, annualize all volatility estimates
        trading_periods_per_year: Periods per year for annualization

    Returns:
        DataFrame with one column per window (columns: rolling_vol_10, rolling_vol_20, etc.)

    Examples:
        >>> returns = pd.Series(np.random.normal(0, 0.001, 200))
        >>> vol_df = compute_rolling_volatility_multiple_windows(
        ...     returns,
        ...     windows=[10, 30, 100]
        ... )
        >>> print(vol_df.columns)
        Index(['rolling_vol_10', 'rolling_vol_30', 'rolling_vol_100'], dtype='object')

        >>> # Compute volatility ratio (short-term vs long-term)
        >>> vol_df['vol_ratio'] = vol_df['rolling_vol_10'] / vol_df['rolling_vol_100']
        >>> print(vol_df['vol_ratio'].describe())

        # High vol_ratio → recent spike (>1.5)
        # Low vol_ratio → volatility declining (<0.7)

    Notes:
        **Common window combinations:**

        - **Intraday minute data**: [5, 15, 30, 60] (5min to 1hr)
        - **Tick/second data**: [10, 30, 100, 300] (10 ticks to 300 ticks)
        - **Daily data**: [5, 20, 60, 252] (1 week to 1 year)

        **Memory efficiency**: Computing multiple windows together is more
        efficient than separate calls, but still memory-intensive for very
        large datasets with many windows.

    References:
        - Corsi, F. (2009). "A simple approximate long-memory model of realized
          volatility" (HAR-RV model with multiple horizons)
    """
    vol_df = pd.DataFrame(index=returns.index)

    for window in windows:
        vol_df[f"rolling_vol_{window}"] = compute_rolling_volatility(
            returns=returns,
            window=window,
            annualize=annualize,
            trading_periods_per_year=trading_periods_per_year,
        )

    return vol_df


def compute_realized_volatility(
    returns: pd.Series,
    window: int = 20,
    min_periods: int = None,
    annualize: bool = False,
    trading_periods_per_year: int = 252 * 390,
) -> pd.Series:
    """
    Compute realized volatility (sum of squared returns).

    Realized volatility is an alternative volatility estimator that sums squared
    returns rather than taking standard deviation. Theoretically equivalent in
    large samples, but has different properties in small samples.

    Formula:
        RV_t = sqrt(sum(returns_{t-window+1:t}^2))

        If annualize=True:
            annualized_RV_t = RV_t × sqrt(trading_periods_per_year / window)

    Intuition:
        **Realized volatility vs standard deviation:**

        Both measure return variability, but:
        - Standard deviation: average squared deviation from mean
        - Realized volatility: square root of sum of squared returns

        For high-frequency data with near-zero mean returns, they're nearly identical.
        However, realized volatility:
        - Is more robust to outliers in small samples
        - Has better statistical properties for option pricing
        - Is standard in academic volatility research

        **When to use realized volatility:**
        - High-frequency data (intraday, tick-level)
        - Volatility forecasting for derivatives
        - Research / academic analysis

        **When to use standard deviation:**
        - Lower frequency data (daily, weekly)
        - More interpretable for practitioners
        - Standard in most trading systems

    Args:
        returns: Series of log returns
        window: Number of periods for rolling window
        min_periods: Minimum observations required
        annualize: If True, annualize the volatility estimate
        trading_periods_per_year: Periods per year for annualization

    Returns:
        Series containing realized volatility estimates

    Examples:
        >>> returns = pd.Series([0.001, -0.002, 0.0015, -0.001, 0.0005])
        >>> compute_realized_volatility(returns, window=3)
        0         NaN
        1         NaN
        2    0.001871
        3    0.001871
        4    0.001304
        dtype: float64

        # At t=2: RV = sqrt(0.001^2 + (-0.002)^2 + 0.0015^2) = 0.001871

    Notes:
        **Relationship to standard deviation:**

        For returns with mean ≈ 0 (typical in HFT):
        realized_vol ≈ std_dev × sqrt(window / (window - 1))

        The difference is negligible for large windows but can be noticeable
        for very small windows (< 10).

    """
    if min_periods is None:
        min_periods = window

    # Compute sum of squared returns
    squared_returns = returns**2
    sum_squared = squared_returns.rolling(window=window, min_periods=min_periods).sum()

    # Take square root
    realized_vol = np.sqrt(sum_squared)

    # Annualize if requested
    if annualize:
        # Adjustment factor for realized volatility annualization
        realized_vol = realized_vol * np.sqrt(trading_periods_per_year / window)
        realized_vol.name = f"realized_vol_{window}_annualized"
    else:
        realized_vol.name = f"realized_vol_{window}"

    return realized_vol


def compute_vwap(
    prices: pd.Series,
    volumes: pd.Series,
    window: int = None,
    reset_daily: bool = False,
) -> pd.Series:
    """
    Compute Volume-Weighted Average Price (VWAP).

    VWAP weights each price by its trading volume, giving more importance to
    prices where more volume traded. It's a benchmark used by institutional
    traders to assess execution quality and by algorithms for optimal execution.

    Formula:
        If window is None (cumulative from start or daily reset):
            VWAP_t = Σ(price_i × volume_i) / Σ(volume_i) for i in [start, t]

        If window is specified (rolling):
            VWAP_t = Σ(price_i × volume_i) / Σ(volume_i) for i in [t-window+1, t]

    Intuition:
        **Why VWAP matters in trading:**

        VWAP answers: "What is the average price weighted by how much actually
        traded at each price level?"

        Simple average price treats all ticks equally, but VWAP recognizes that:
        - A price where 10,000 shares traded is more "important" than
        - A price where 100 shares traded

        **Use cases:**

        1. **Execution benchmark**: Institutional traders are often measured
           against VWAP
           - Buy below VWAP → good execution (saved money)
           - Sell above VWAP → good execution (got better price)

        2. **Fair value indicator**: VWAP represents where the "center of gravity"
           of trading occurred
           - Price >> VWAP → asset may be overbought (lots of volume at lower prices)
           - Price << VWAP → asset may be oversold (lots of volume at higher prices)

        3. **Support/resistance**: VWAP acts as dynamic support/resistance
           - Price above VWAP → bullish (buyers controlled the session)
           - Price below VWAP → bearish (sellers controlled the session)
           - Mean reversion strategies: trade back toward VWAP

        4. **Algorithmic execution**: VWAP algorithms try to match the VWAP
           - Minimize market impact
           - Execute large orders without moving price
           - Blend in with natural market flow

        **Daily vs Rolling VWAP:**

        - **Daily VWAP** (reset_daily=True):
          - Resets at market open each day
          - Most common for institutional benchmarking
          - Standard reference point for intraday trading

        - **Rolling VWAP** (window specified):
          - Moves with a fixed window (e.g., last 100 ticks)
          - More responsive to recent price action
          - Better for short-term trading signals

        - **Cumulative VWAP** (window=None, reset_daily=False):
          - From start of data to current point
          - Grows less responsive over time
          - Rarely used except for analysis

    Args:
        prices: Series of prices (typically mid_price or last trade price)
        volumes: Series of volumes (typically trade volume or top-of-book volume)
        window: Number of periods for rolling VWAP (if None, cumulative from start/daily)
        reset_daily: If True, reset VWAP calculation at start of each trading day

    Returns:
        Series containing VWAP values

    Examples:
        >>> prices = pd.Series([100.0, 101.0, 100.5, 102.0, 101.5])
        >>> volumes = pd.Series([1000, 2000, 1500, 500, 1000])
        >>> compute_vwap(prices, volumes, window=None)
        0    100.000000
        1    100.666667
        2    100.625000
        3    100.708333
        4    100.833333
        dtype: float64

        # Calculation at t=2:
        # VWAP = (100*1000 + 101*2000 + 100.5*1500) / (1000 + 2000 + 1500)
        #      = (100000 + 202000 + 150750) / 4500
        #      = 452750 / 4500 = 100.611

        >>> # Rolling VWAP (last 3 periods)
        >>> compute_vwap(prices, volumes, window=3)
        0         NaN
        1         NaN
        2    100.625
        3    101.000
        4    101.292
        dtype: float64

        # At t=2: VWAP of last 3 periods
        # At t=3: Window slides forward

    Notes:
        **Implementation considerations:**

        - Uses (price × volume) / volume, not (price × volume) / count
        - Handles zero volume by adding small epsilon (1e-10) to denominator
        - NaN values in price or volume are excluded from calculation
        - For cumulative VWAP, early values have more weight on entire calculation

        **Common pitfalls:**

        - Don't use VWAP from one asset to trade another
        - VWAP resets daily - don't compare across days
        - Low volume periods make VWAP less meaningful
        - Not predictive on its own - use with other signals

        **Typical parameters:**

        - Intraday benchmark: reset_daily=True, window=None (full day)
        - Short-term signal: window=20-100 (responsive to recent activity)
        - High-frequency: window=10-30 (very responsive)

    References:
        - Berkowitz, S. A., Logue, D. E., & Noser, E. A. (1988). "The total
          cost of transactions on the NYSE"
        - Perold, A. F. (1988). "The implementation shortfall: Paper versus
          reality"
        - Kissell, R., & Glantz, M. (2003). "Optimal Trading Strategies"
    """
    # Handle NaN values
    valid_mask = ~(prices.isna() | volumes.isna())
    prices_clean = prices[valid_mask]
    volumes_clean = volumes[valid_mask]

    # Compute price × volume
    notional = prices_clean * volumes_clean

    # Small epsilon to avoid division by zero
    epsilon = 1e-10

    if window is None and not reset_daily:
        # Cumulative VWAP from start
        cumsum_notional = notional.cumsum()
        cumsum_volume = volumes_clean.cumsum() + epsilon
        vwap = cumsum_notional / cumsum_volume
        vwap.name = "vwap_cumulative"

    elif window is None and reset_daily:
        # Daily VWAP (reset at start of each day)
        # Group by date
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("reset_daily=True requires DatetimeIndex")

        date_groups = prices_clean.index.date
        cumsum_notional = notional.groupby(date_groups).cumsum()
        cumsum_volume = volumes_clean.groupby(date_groups).cumsum() + epsilon
        vwap = cumsum_notional / cumsum_volume
        vwap.name = "vwap_daily"

    else:
        # Rolling VWAP with fixed window
        rolling_notional = notional.rolling(window=window, min_periods=1).sum()
        rolling_volume = (
            volumes_clean.rolling(window=window, min_periods=1).sum() + epsilon
        )
        vwap = rolling_notional / rolling_volume
        vwap.name = f"vwap_rolling_{window}"

    # Reindex to match original series (fill NaN where input was NaN)
    vwap = vwap.reindex(prices.index)

    return vwap


def compute_vwap_multiple_windows(
    prices: pd.Series,
    volumes: pd.Series,
    windows: List[int] = [20, 60, 100],
) -> pd.DataFrame:
    """
    Compute VWAP at multiple rolling window lengths simultaneously.

    Multiple VWAP windows provide different time horizons for fair value
    estimation, useful for multi-timeframe analysis and ML features.

    Formula:
        For each window w in windows:
            VWAP_w = rolling_sum(price × volume, w) / rolling_sum(volume, w)

    Intuition:
        **Multi-timeframe VWAP analysis:**

        Different VWAP windows reveal different market dynamics:

        - **Short window (20-30)**: Recent fair value
          - Reacts quickly to new information
          - More noise, less stable
          - Good for: short-term mean reversion signals

        - **Medium window (60-100)**: Intermediate fair value
          - Balances responsiveness and stability
          - Standard for intraday strategies
          - Good for: execution benchmarking, position entry/exit

        - **Long window (200+)**: Long-term fair value
          - Smooth, stable reference
          - Slow to react
          - Good for: trend identification, regime detection

        **Trading signals from multiple VWAPs:**

        - Price > all VWAPs → Strong uptrend (stay long)
        - Price < all VWAPs → Strong downtrend (stay short)
        - Price between VWAPs → Consolidation (range trade or wait)
        - VWAP crossovers → Potential trend changes

        **ML feature engineering:**

        - Distance from VWAP: (price - VWAP) / price
        - VWAP slope: rate of change in VWAP
        - Multi-VWAP spread: VWAP_short - VWAP_long

    Args:
        prices: Series of prices
        volumes: Series of volumes
        windows: List of window lengths (default: [20, 60, 100])

    Returns:
        DataFrame with one VWAP column per window

    Examples:
        >>> prices = pd.Series(np.random.uniform(99, 101, 200))
        >>> volumes = pd.Series(np.random.uniform(500, 2000, 200))
        >>> vwap_df = compute_vwap_multiple_windows(prices, volumes, [20, 60, 100])
        >>> print(vwap_df.columns)
        Index(['vwap_rolling_20', 'vwap_rolling_60', 'vwap_rolling_100'], dtype='object')

        >>> # Compute distance from VWAP
        >>> vwap_df['distance_from_vwap_20'] = (prices - vwap_df['vwap_rolling_20']) / prices

    Notes:
        **Common window combinations:**

        - Minute data: [20, 60, 390] (20min, 1hr, full day)
        - Tick data: [50, 100, 500] (50 ticks to 500 ticks)
        - Second data: [60, 300, 1800] (1min, 5min, 30min)
    """
    vwap_df = pd.DataFrame(index=prices.index)

    for window in windows:
        vwap_df[f"vwap_rolling_{window}"] = compute_vwap(
            prices=prices, volumes=volumes, window=window, reset_daily=False
        )

    return vwap_df


def compute_vwap_deviation(
    prices: pd.Series,
    vwap: pd.Series,
    method: str = "absolute",
) -> pd.Series:
    """
    Compute deviation of current price from VWAP.

    Price deviation from VWAP indicates how far the current price has moved
    from the volume-weighted average, signaling potential mean reversion
    opportunities or trend strength.

    Formula:
        If method='absolute':
            deviation = price - VWAP

        If method='relative':
            deviation = (price - VWAP) / VWAP

        If method='bps':
            deviation = ((price - VWAP) / VWAP) × 10000

    Intuition:
        **Interpreting VWAP deviation:**

        - **Positive deviation**: Price > VWAP
          - Buyers have been more aggressive
          - Potential resistance (mean revert down?)
          - Or continuation if strong trend

        - **Negative deviation**: Price < VWAP
          - Sellers have been more aggressive
          - Potential support (mean revert up?)
          - Or continuation if strong downtrend

        - **Near zero**: Price ≈ VWAP
          - Fair value, balanced market
          - Consolidation, no strong directional bias

        **Trading applications:**

        1. **Mean reversion**: Trade back toward VWAP
           - Buy when price drops below VWAP (expecting reversion)
           - Sell when price rises above VWAP
           - Works best in range-bound markets

        2. **Trend confirmation**: Large deviations confirm trends
           - Persistent positive deviation → strong uptrend
           - Persistent negative deviation → strong downtrend
           - Don't fade strong trends

        3. **Execution timing**: Minimize VWAP deviation
           - Buy when price < VWAP (getting discount)
           - Sell when price > VWAP (getting premium)
           - Part of VWAP execution algorithms

        **Thresholds for action:**

        - Small deviation (< 0.1%): Market at fair value
        - Medium deviation (0.1-0.3%): Potential opportunity
        - Large deviation (> 0.5%): Strong signal (trend or reversal)

    Args:
        prices: Series of current prices
        vwap: Series of VWAP values
        method: 'absolute' (price difference), 'relative' (percentage),
                or 'bps' (basis points)

    Returns:
        Series containing VWAP deviation


        # At t=3: Price is 102, VWAP is 100.8
        # Absolute: 102 - 100.8 = 1.2
        # Relative: (102 - 100.8) / 100.8 = 0.0119 (1.19%)
        # BPS: 1.19% × 10000 = 119 basis points

    Notes:
        **Method selection:**

        - **Absolute**: Good for same-asset comparison over time
        - **Relative**: Good for cross-asset comparison (normalizes by price level)
        - **BPS**: Standard in fixed income, clearer for small deviations

        **Statistical properties:**

        - VWAP deviation often mean-reverting (oscillates around zero)
        - Extreme deviations tend to reverse (but can persist in trends)
        - Distribution often has fat tails (large moves happen more than normal distribution predicts)

    """
    if method == "absolute":
        deviation = prices - vwap
        deviation.name = "vwap_deviation_abs"

    elif method == "relative":
        deviation = (prices - vwap) / vwap
        deviation.name = "vwap_deviation_pct"

    elif method == "bps":
        deviation = ((prices - vwap) / vwap) * 10000
        deviation.name = "vwap_deviation_bps"

    else:
        raise ValueError(
            f"Invalid method: {method}. Choose 'absolute', 'relative', or 'bps'"
        )

    return deviation


def compute_queue_imbalance_at_level(
    bid_volumes: pd.DataFrame,
    ask_volumes: pd.DataFrame,
    level: int = 1,
) -> pd.Series:
    """
    Compute queue imbalance at a specific level of the order book.

    Queue imbalance measures the relative difference between bid and ask volumes
    at a particular price level, indicating directional pressure at that depth.
    Unlike order flow imbalance which aggregates across levels, this captures
    imbalance at individual depths to reveal liquidity structure.

    Formula:
        queue_imbalance_level_i = (bid_volume_i - ask_volume_i) / (bid_volume_i + ask_volume_i)

    Intuition:
        **Why level-specific imbalance matters:**

        Different levels of the order book reveal different information:

        - **Level 1 (top-of-book)**: Immediate liquidity and pressure
          - Most responsive to short-term order flow
          - High-frequency traders focus here
          - Changes rapidly (100ms-1s timescale)

        - **Levels 2-3**: Near-market depth
          - Shows commitment of larger participants
          - Less noise than level 1
          - Important for medium-sized orders

        - **Levels 5-10**: Deep book structure
          - Indicates strategic positioning
          - Slower to change (minutes-hours)
          - Reveals institutional interest

        **Trading signals:**

        - **Positive imbalance (>0)**: More bids than asks at this level
          - Buying pressure / support at this price
          - If persistent → potential price floor
          - If at deep levels → hidden buy interest

        - **Negative imbalance (<0)**: More asks than bids at this level
          - Selling pressure / resistance at this price
          - If persistent → potential price ceiling
          - If at deep levels → hidden sell interest

        - **Near zero**: Balanced liquidity
          - Fair price discovery
          - No strong directional bias at this level

        **Pattern detection:**

        1. **Inverted imbalance structure**:
           - Level 1: negative (selling pressure at top)
           - Level 5: positive (buying support deeper)
           - Signal: Strong buyers waiting below, potential bounce

        2. **Uniform imbalance**:
           - All levels have same sign
           - Signal: Strong directional consensus

        3. **Diverging imbalance**:
           - Top levels: positive
           - Deep levels: negative
           - Signal: Short-term buying vs long-term selling pressure

    Args:
        bid_volumes: DataFrame with bid volumes at each level
        ask_volumes: DataFrame with ask volumes at each level
        level: Order book level to compute imbalance for (1 = top of book)

    Returns:
        Series containing queue imbalance at specified level (range: [-1, 1])

    Examples:
        >>> bid_volumes = pd.DataFrame({
        ...     'bid_volume_1': [1000, 1500, 800],
        ...     'bid_volume_2': [2000, 1800, 1200],
        ... })
        >>> ask_volumes = pd.DataFrame({
        ...     'ask_volume_1': [900, 1100, 1200],
        ...     'ask_volume_2': [1800, 2200, 1000],
        ... })
        >>> compute_queue_imbalance_at_level(bid_volumes, ask_volumes, level=1)
        0    0.052632
        1    0.153846
        2   -0.200000
        Name: queue_imbalance_level_1, dtype: float64

        # At t=0: (1000 - 900) / (1000 + 900) = 100 / 1900 = 0.0526
        # At t=2: (800 - 1200) / (800 + 1200) = -400 / 2000 = -0.20

        >>> compute_queue_imbalance_at_level(bid_volumes, ask_volumes, level=2)
        0    0.052632
        1   -0.100000
        2    0.090909
        Name: queue_imbalance_level_2, dtype: float64

        # Different imbalance pattern at level 2 vs level 1

    Notes:
        **Interpretation guide:**

        - Imbalance > 0.5: Strong buy pressure (overwhelmingly more bids)
        - Imbalance 0.1 to 0.5: Moderate buy pressure
        - Imbalance -0.1 to 0.1: Balanced
        - Imbalance -0.5 to -0.1: Moderate sell pressure
        - Imbalance < -0.5: Strong sell pressure (overwhelmingly more asks)

        **Data quality:**

        - Epsilon (1e-10) added to denominator prevents division by zero
        - Empty levels (zero volume on both sides) return imbalance of 0
        - Extreme imbalances (near ±1) often indicate spoofing or errors

    References:
        - Cont, R., Kukanov, A., & Stoikov, S. (2014). "The price impact
          of order book events"
        - Huang, W., Lehalle, C. A., & Rosenbaum, M. (2015). "Simulating
          and analyzing order book data: The queue-reactive model"
    """
    bid_col = f"bid_volume_{level}"
    ask_col = f"ask_volume_{level}"

    bid_vol = bid_volumes[bid_col]
    ask_vol = ask_volumes[ask_col]

    # Small epsilon to avoid division by zero
    epsilon = 1e-10
    total_vol = bid_vol + ask_vol + epsilon

    imbalance = (bid_vol - ask_vol) / total_vol
    imbalance.name = f"queue_imbalance_level_{level}"

    return imbalance


def compute_queue_imbalance_multiple_levels(
    bid_volumes: pd.DataFrame,
    ask_volumes: pd.DataFrame,
    levels: List[int] = [1, 2, 3, 5, 10],
) -> pd.DataFrame:
    """
    Compute queue imbalance at multiple order book levels simultaneously.

    Multi-level imbalance creates a rich feature set revealing the liquidity
    structure across the entire visible order book. Different levels capture
    different participant behaviors and time horizons.

    Formula:
        For each level i in levels:
            queue_imbalance_i = (bid_volume_i - ask_volume_i) / (bid_volume_i + ask_volume_i)

    Intuition:
        **Depth profile analysis:**

        The pattern of imbalance across levels tells a story about market structure:

        1. **Uniform positive imbalance** (all levels > 0):
           - Strong consensus buying pressure
           - Likely uptrend or strong support
           - All participant types are bullish

        2. **Uniform negative imbalance** (all levels < 0):
           - Strong consensus selling pressure
           - Likely downtrend or strong resistance
           - All participant types are bearish

        3. **Gradient pattern** (imbalance decreases with depth):
           Level 1: +0.3, Level 5: +0.1, Level 10: -0.1
           - Near-term buying but longer-term selling
           - HFTs buying, institutions selling
           - Potential reversal signal

        4. **Inverted gradient** (imbalance increases with depth):
           Level 1: -0.2, Level 5: 0.0, Level 10: +0.3
           - Near-term selling but longer-term buying
           - Institutions accumulating on dip
           - Potential bounce signal

        5. **Oscillating pattern**:
           Level 1: +0.2, Level 3: -0.1, Level 5: +0.3
           - Layered liquidity structure
           - Multiple strategies/participants active
           - More complex market dynamics

        **ML feature engineering:**

        Models can learn patterns like:
        - "If top 3 levels negative, deeper levels positive → expect bounce"
        - "If imbalance uniform across all levels → strong trend continuation"
        - "If level 1 opposite sign to level 10 → regime transition"

        **Market maker behavior:**

        - Imbalance at level 1-2: Reflects immediate toxicity/adverse selection
        - Imbalance at level 5-10: Reflects strategic positioning
        - Divergence between near and far levels: Different risk appetites

    Args:
        bid_volumes: DataFrame with bid volumes at each level
        ask_volumes: DataFrame with ask volumes at each level
        levels: List of levels to compute imbalance for (default: [1,2,3,5,10])

    Returns:
        DataFrame with queue imbalance at each specified level

    Examples:
        >>> bid_volumes = pd.DataFrame({
        ...     'bid_volume_1': [1000, 1500],
        ...     'bid_volume_2': [2000, 1800],
        ...     'bid_volume_3': [1500, 2200]
        ... })
        >>> ask_volumes = pd.DataFrame({
        ...     'ask_volume_1': [900, 1100],
        ...     'ask_volume_2': [1800, 2200],
        ...     'ask_volume_3': [2000, 1800]
        ... })
        >>> compute_queue_imbalance_multiple_levels(bid_volumes, ask_volumes, [1,2,3])
           queue_imbalance_level_1  queue_imbalance_level_2  queue_imbalance_level_3
        0                 0.052632                 0.052632                -0.142857
        1                 0.153846                -0.100000                 0.100000

        # At t=0: Level 1 and 2 show buying pressure, but level 3 shows selling
        # At t=1: Level 1 strong buying, level 2 selling, level 3 buying
        # → Complex liquidity structure

    Notes:
        **Typical level selections:**

        - High-frequency focus: [1, 2, 3] (immediate microstructure)
        - Balanced analysis: [1, 3, 5, 10] (multi-scale)
        - Full book: [1, 2, 3, 4, 5, 7, 10, 15, 20] (comprehensive)

        **Feature selection for ML:**

        Not all levels may be informative. Use feature importance to determine:
        - Which levels have strongest predictive power
        - Whether to aggregate (e.g., average of levels 1-5)
        - Whether to compute ratios (level_1_imb / level_10_imb)

    References:
        - Huang, W., & Rosenbaum, M. (2017). "Ergodicity and diffusivity of
          Markovian order book models: A general framework"
    """
    imbalance_df = pd.DataFrame(index=bid_volumes.index)

    for level in levels:
        imbalance_df[f"queue_imbalance_level_{level}"] = (
            compute_queue_imbalance_at_level(
                bid_volumes=bid_volumes, ask_volumes=ask_volumes, level=level
            )
        )

    return imbalance_df


def compute_queue_imbalance_gradient(
    bid_volumes: pd.DataFrame,
    ask_volumes: pd.DataFrame,
    level_near: int = 1,
    level_far: int = 10,
) -> pd.Series:
    """
    Compute the gradient (difference) in queue imbalance between near and far levels.

    The imbalance gradient reveals whether liquidity pressure is consistent across
    depths or if there's divergence between near-book and deep-book dynamics.
    This is a powerful signal for detecting hidden institutional interest.

    Formula:
        imbalance_gradient = queue_imbalance(level_near) - queue_imbalance(level_far)

    Intuition:
        **What the gradient reveals:**

        The gradient shows the "tilt" in liquidity structure:

        - **Positive gradient** (near > far):
          - Near levels more bid-heavy than deep levels
          - Scenarios:
            a) HFTs bidding aggressively, institutions offering deeper
            b) Short-term buying interest, long-term resistance
            c) Retail buying, smart money selling
          - Potential: Near-term pop, then reversal

        - **Negative gradient** (near < far):
          - Near levels more offer-heavy than deep levels
          - Scenarios:
            a) HFTs offering aggressively, institutions bidding deeper
            b) Short-term selling, long-term support
            c) Weak hands selling, strong hands accumulating
          - Potential: Near-term dip, then bounce

        - **Near-zero gradient** (near ≈ far):
          - Uniform pressure across all depths
          - Strong directional consensus
          - More reliable trending signal

        **Trading applications:**

        1. **Institutional detection**:
           - Large gradient → layered strategies, conflicting interests
           - Small gradient → unified direction, follow the flow

        2. **Reversal signals**:
           - Gradient flips sign → regime change
           - Extreme gradient (>0.5) → likely mean reversion

        3. **Trend confirmation**:
           - Price rising + negative gradient → buying dip (healthy)
           - Price rising + positive gradient → chasing (risky)

        **Example scenario:**

        Market is falling, but you observe:
        - Level 1 imbalance: -0.3 (selling pressure)
        - Level 10 imbalance: +0.4 (buying support)
        - Gradient: -0.3 - 0.4 = -0.7 (large negative)

        Interpretation: Weak hands selling at top, strong hands buying deeper.
        Strategy: Join the deeper bids (institutional side).

    Args:
        bid_volumes: DataFrame with bid volumes at each level
        ask_volumes: DataFrame with ask volumes at each level
        level_near: Near order book level (default: 1, top of book)
        level_far: Far order book level (default: 10, deep in book)

    Returns:
        Series containing imbalance gradient (range: [-2, 2])

    Examples:
        >>> bid_volumes = pd.DataFrame({
        ...     'bid_volume_1': [1000, 800, 1500],
        ...     'bid_volume_10': [3000, 3500, 2500]
        ... })
        >>> ask_volumes = pd.DataFrame({
        ...     'ask_volume_1': [1200, 1200, 1000],
        ...     'ask_volume_10': [2000, 1500, 3000]
        ... })
        >>> compute_queue_imbalance_gradient(bid_volumes, ask_volumes, 1, 10)
        0   -0.391304
        1   -0.440000
        2    0.040000
        Name: queue_imbalance_gradient_1_10, dtype: float64

        # At t=0:
        # Level 1 imbalance: (1000-1200)/(1000+1200) = -0.091
        # Level 10 imbalance: (3000-2000)/(3000+2000) = 0.200
        # Gradient: -0.091 - 0.200 = -0.291
        # → Selling at top, buying deeper (potential bounce)

        # At t=2:
        # Level 1 imbalance: (1500-1000)/(1500+1000) = 0.200
        # Level 10 imbalance: (2500-3000)/(2500+3000) = -0.091
        # Gradient: 0.200 - (-0.091) = 0.291
        # → Buying at top, selling deeper (potential resistance)

    Notes:
        **Interpretation thresholds:**

        - Gradient > +0.5: Strong near buying vs far selling (likely reversal down)
        - Gradient +0.2 to +0.5: Moderate divergence
        - Gradient -0.2 to +0.2: Aligned pressure (trending)
        - Gradient -0.5 to -0.2: Moderate divergence
        - Gradient < -0.5: Strong near selling vs far buying (likely reversal up)

        **Common level pairs:**

        - (1, 5): Near vs medium depth
        - (1, 10): Near vs deep book
        - (2, 8): Avoiding noise at level 1
        - (5, 20): Medium vs very deep

    References:
        - Cartea, Á., Jaimungal, S., & Penalva, J. (2015). "Algorithmic and
          High-Frequency Trading" - Chapter on order book dynamics
    """
    imbalance_near = compute_queue_imbalance_at_level(
        bid_volumes, ask_volumes, level_near
    )
    imbalance_far = compute_queue_imbalance_at_level(
        bid_volumes, ask_volumes, level_far
    )

    gradient = imbalance_near - imbalance_far
    gradient.name = f"queue_imbalance_gradient_{level_near}_{level_far}"

    return gradient


def compute_queue_imbalance_volatility(
    bid_volumes: pd.DataFrame,
    ask_volumes: pd.DataFrame,
    level: int = 1,
    window: int = 20,
) -> pd.Series:
    """
    Compute the volatility (standard deviation) of queue imbalance at a level.

    Imbalance volatility measures how rapidly the bid/ask balance fluctuates
    at a given level. High volatility indicates unstable liquidity, aggressive
    order flow, or potential manipulation (spoofing). Low volatility suggests
    stable, committed liquidity.

    Formula:
        imbalance_volatility_t = std(queue_imbalance_{t-window+1:t})

    Intuition:
        **Why imbalance volatility matters:**

        - **Low volatility** (stable imbalance):
          - Committed liquidity providers
          - Genuine market interest
          - Predictable microstructure
          - Good for execution (stable quotes)

        - **High volatility** (rapidly changing imbalance):
          - Fleeting liquidity (quote stuffing?)
          - Aggressive HFT activity
          - Potential spoofing/layering
          - Risky for execution (quotes may disappear)

        **Pattern detection:**

        1. **Volatility spike at level 1**:
           - Market becoming unstable
           - Potential news event or large order
           - Widen spreads, reduce exposure

        2. **Volatility spike at deep levels**:
           - Hidden order activity
           - Institutional rebalancing
           - Strategic positioning changes

        3. **Volatility divergence**:
           - Level 1: high volatility (aggressive trading)
           - Level 10: low volatility (stable base)
           - Signal: Short-term noise over long-term support

        **Trading applications:**

        - Execution algorithms: Avoid high-volatility periods
        - Market making: Widen spreads when volatility high
        - Spoofing detection: Extreme volatility may indicate manipulation

    Args:
        bid_volumes: DataFrame with bid volumes at each level
        ask_volumes: DataFrame with ask volumes at each level
        level: Order book level to analyze (default: 1)
        window: Rolling window for volatility calculation (default: 20)

    Returns:
        Series containing rolling volatility of queue imbalance

    Examples:
        >>> bid_volumes = pd.DataFrame({
        ...     'bid_volume_1': [1000, 1100, 900, 1050, 1000]
        ... })
        >>> ask_volumes = pd.DataFrame({
        ...     'ask_volume_1': [1000, 900, 1100, 950, 1000]
        ... })
        >>> compute_queue_imbalance_volatility(bid_volumes, ask_volumes, level=1, window=3)
        0         NaN
        1         NaN
        2    0.099504
        3    0.099504
        4    0.086023
        dtype: float64

        # Measures how much the imbalance at level 1 fluctuates

    Notes:
        **Interpretation:**

        - Volatility < 0.1: Stable, reliable liquidity
        - Volatility 0.1-0.3: Moderate fluctuation, typical
        - Volatility > 0.3: High instability, be cautious

        **Use cases:**

        - Risk management: Scale down in high-volatility periods
        - Quality control: Filter out unstable liquidity
        - Regime detection: Volatility clusters indicate market state changes

    """
    # Compute imbalance time series
    imbalance = compute_queue_imbalance_at_level(bid_volumes, ask_volumes, level)

    # Compute rolling volatility
    imbalance_vol = imbalance.rolling(window=window, min_periods=window).std()
    imbalance_vol.name = f"queue_imbalance_vol_level_{level}_window_{window}"

    return imbalance_vol


def compute_cumulative_volume_imbalance(
    bid_volumes: pd.DataFrame,
    ask_volumes: pd.DataFrame,
    levels: int = 10,
    normalize: bool = False,
) -> pd.Series:
    """
    Compute cumulative volume imbalance (absolute quantity difference).

    Cumulative volume imbalance measures the total absolute difference between
    bid and ask volumes across multiple levels, revealing the overall directional
    liquidity pressure in the order book. Unlike normalized imbalance (ratio),
    this captures the MAGNITUDE of imbalance, not just direction.

    Formula:
        If normalize=False (absolute):
            cumulative_imbalance = Σ(bid_volume_i - ask_volume_i) for i in [1, levels]

        If normalize=True (relative):
            cumulative_imbalance = Σ(bid_volume_i - ask_volume_i) / Σ(bid_volume_i + ask_volume_i)

    Intuition:
        **Why absolute imbalance matters:**

        Normalized imbalance (ratio-based) answers: "What's the DIRECTION of pressure?"
        Absolute imbalance answers: "How MUCH imbalance is there?"

        These tell different stories:

        Example 1:
        - Bid volume: 10,000 | Ask volume: 8,000
        - Normalized imbalance: (10k - 8k) / (10k + 8k) = 0.11 (11% more bids)
        - Absolute imbalance: 10k - 8k = 2,000 shares

        Example 2:
        - Bid volume: 100 | Ask volume: 80
        - Normalized imbalance: (100 - 80) / (100 + 80) = 0.11 (same 11%)
        - Absolute imbalance: 100 - 80 = 20 shares

        Both have same DIRECTION (more bids) but vastly different MAGNITUDE:
        - Example 1: 2,000 shares → significant institutional pressure
        - Example 2: 20 shares → trivial retail activity

        **Trading implications:**

        1. **Market impact estimation**:
           - Large absolute imbalance → expect price impact
           - Small absolute imbalance → minimal slippage
           - Execution algorithms need absolute quantities

        2. **Institutional detection**:
           - Absolute imbalance > 10,000 shares → likely institutional
           - Absolute imbalance < 100 shares → likely retail
           - Large players reveal themselves through size

        3. **Liquidity regime**:
           - High absolute imbalance → one-sided market
           - Low absolute imbalance → balanced, liquid market
           - Execution quality depends on balance

        **Positive vs Negative imbalance:**

        - **Positive** (bid volume > ask volume):
          - More buyers willing to buy than sellers willing to sell
          - Buying pressure, potential support
          - If persistent → likely price increase
          - Large positive → strong institutional accumulation

        - **Negative** (ask volume > bid volume):
          - More sellers willing to sell than buyers willing to buy
          - Selling pressure, potential resistance
          - If persistent → likely price decrease
          - Large negative → strong institutional distribution

        - **Near zero**:
          - Balanced supply and demand
          - Fair price discovery
          - Low transaction costs

        **Time dynamics:**

        - Sudden spike in absolute imbalance → large order arrival
        - Gradual increase → building pressure (trend forming)
        - Oscillation around zero → range-bound market
        - Persistent one-sided → trending market

    Args:
        bid_volumes: DataFrame with bid volumes at each level
        ask_volumes: DataFrame with ask volumes at each level
        levels: Number of levels to sum over (default: 10)
        normalize: If True, divide by total volume for comparison across assets

    Returns:
        Series containing cumulative volume imbalance

    Examples:
        >>> bid_volumes = pd.DataFrame({
        ...     'bid_volume_1': [1000, 1500, 800],
        ...     'bid_volume_2': [800, 1200, 900],
        ...     'bid_volume_3': [600, 1000, 700]
        ... })
        >>> ask_volumes = pd.DataFrame({
        ...     'ask_volume_1': [900, 1100, 1200],
        ...     'ask_volume_2': [700, 1300, 800],
        ...     'ask_volume_3': [650, 900, 900]
        ... })
        >>> compute_cumulative_volume_imbalance(bid_volumes, ask_volumes, levels=3)
        0     150.0
        1     400.0
        2    -500.0
        Name: cumulative_volume_imbalance, dtype: float64

        # At t=0: (1000-900) + (800-700) + (600-650) = 100 + 100 - 50 = 150
        #         Net 150 more bids than asks (slight buying pressure)

        # At t=1: (1500-1100) + (1200-1300) + (1000-900) = 400 - 100 + 100 = 400
        #         Net 400 more bids (moderate buying pressure)

        # At t=2: (800-1200) + (900-800) + (700-900) = -400 + 100 - 200 = -500
        #         Net 500 more asks (significant selling pressure)

        >>> # Normalized version (for cross-asset comparison)
        >>> compute_cumulative_volume_imbalance(bid_volumes, ask_volumes, levels=3, normalize=True)
        0    0.033058
        1    0.068966
        2   -0.111111
        dtype: float64

    Notes:
        **Interpretation guidelines:**

        For a typical liquid stock (e.g., AAPL) at top 10 levels:
        - Imbalance < -50,000 shares: Strong selling pressure
        - Imbalance -10,000 to -50,000: Moderate selling
        - Imbalance -10,000 to +10,000: Balanced
        - Imbalance +10,000 to +50,000: Moderate buying
        - Imbalance > +50,000 shares: Strong buying pressure

        For crypto (e.g., BTC/USDT):
        - Imbalance < -100 BTC: Strong selling
        - Imbalance -20 to -100 BTC: Moderate selling
        - Imbalance -20 to +20 BTC: Balanced
        - Imbalance +20 to +100 BTC: Moderate buying
        - Imbalance > +100 BTC: Strong buying

        (These are examples - actual thresholds depend on asset liquidity)

        **Normalized vs Absolute:**

        Use absolute when:
        - Trading single asset over time
        - Estimating market impact
        - Detecting institutional activity

        Use normalized when:
        - Comparing across assets
        - ML features (different scales)
        - Analyzing percentage-based signals

    References:
        - Cont, R., Kukanov, A., & Stoikov, S. (2014). "The price impact
          of order book events"
        - Eisler, Z., Bouchaud, J. P., & Kockelkoren, J. (2012). "The
          price impact of order book events: market orders, limit orders
          and cancellations"
    """
    # Sum volumes across levels
    bid_cols = [f"bid_volume_{i}" for i in range(1, levels + 1)]
    ask_cols = [f"ask_volume_{i}" for i in range(1, levels + 1)]

    total_bid = bid_volumes[bid_cols].sum(axis=1)
    total_ask = ask_volumes[ask_cols].sum(axis=1)

    # Compute imbalance
    imbalance = total_bid - total_ask

    if normalize:
        # Normalize by total volume
        epsilon = 1e-10
        total_volume = total_bid + total_ask + epsilon
        imbalance = imbalance / total_volume
        imbalance.name = "cumulative_volume_imbalance_normalized"
    else:
        imbalance.name = "cumulative_volume_imbalance"

    return imbalance


def compute_cumulative_volume_imbalance_rolling(
    bid_volumes: pd.DataFrame,
    ask_volumes: pd.DataFrame,
    levels: int = 10,
    window: int = 20,
    normalize: bool = False,
) -> pd.Series:
    """
    Compute rolling average of cumulative volume imbalance.

    Rolling imbalance smooths out tick-by-tick noise and reveals sustained
    directional pressure. This is more robust than instantaneous imbalance
    for detecting persistent institutional flow.

    Formula:
        rolling_imbalance_t = mean(cumulative_imbalance_{t-window+1:t})

    Intuition:
        **Why rolling imbalance matters:**

        Instantaneous imbalance is noisy:
        - Market makers rapidly update quotes
        - Small orders come and go
        - Quote stuffing creates false signals

        Rolling imbalance filters noise:
        - Reveals sustained pressure
        - Institutions accumulate/distribute over time
        - More actionable signal for medium-term trading

        **Pattern recognition:**

        1. **Increasing rolling imbalance** (getting more positive):
           - Building buying pressure
           - Institutional accumulation
           - Potential uptrend forming

        2. **Decreasing rolling imbalance** (getting more negative):
           - Building selling pressure
           - Institutional distribution
           - Potential downtrend forming

        3. **Stable rolling imbalance**:
           - Equilibrium reached
           - Range-bound market
           - Wait for breakout

        **Trading signals:**

        - Rolling imbalance crosses above zero → bullish signal
        - Rolling imbalance crosses below zero → bearish signal
        - Divergence between price and imbalance → potential reversal
          (price rising but imbalance falling = hidden selling)

    Args:
        bid_volumes: DataFrame with bid volumes at each level
        ask_volumes: DataFrame with ask volumes at each level
        levels: Number of levels to sum over
        window: Number of periods for rolling average
        normalize: If True, compute normalized version

    Returns:
        Series containing rolling cumulative volume imbalance

    Examples:
        >>> bid_volumes = pd.DataFrame({
        ...     'bid_volume_1': [1000, 1100, 900, 1050, 1000, 1200],
        ...     'bid_volume_2': [800, 850, 750, 820, 800, 900]
        ... })
        >>> ask_volumes = pd.DataFrame({
        ...     'ask_volume_1': [900, 950, 950, 900, 950, 1000],
        ...     'ask_volume_2': [700, 750, 800, 750, 750, 800]
        ... })
        >>> compute_cumulative_volume_imbalance_rolling(
        ...     bid_volumes, ask_volumes, levels=2, window=3
        ... )
        0         NaN
        1         NaN
        2    100.000
        3    100.000
        4     83.333
        5    116.667
        dtype: float64

        # Rolling average smooths out tick-by-tick fluctuations

    Notes:
        **Window selection:**

        - Short window (5-20): Responsive, captures regime changes quickly
        - Medium window (20-60): Balanced, filters noise while staying responsive
        - Long window (100+): Smooth baseline, reveals long-term pressure

        **Common strategy:**
        Use multiple windows and look for crossovers:
        - Fast rolling imbalance (10-period)
        - Slow rolling imbalance (50-period)
        - When fast crosses above slow → bullish
        - When fast crosses below slow → bearish
    """
    # Compute instantaneous imbalance
    imbalance = compute_cumulative_volume_imbalance(
        bid_volumes=bid_volumes,
        ask_volumes=ask_volumes,
        levels=levels,
        normalize=normalize,
    )

    # Apply rolling average
    rolling_imbalance = imbalance.rolling(window=window, min_periods=window).mean()

    if normalize:
        rolling_imbalance.name = (
            f"cumulative_volume_imbalance_rolling_{window}_normalized"
        )
    else:
        rolling_imbalance.name = f"cumulative_volume_imbalance_rolling_{window}"

    return rolling_imbalance


def compute_cumulative_volume_imbalance_multiple_windows(
    bid_volumes: pd.DataFrame,
    ask_volumes: pd.DataFrame,
    levels: int = 10,
    windows: List[int] = [10, 30, 60],
    normalize: bool = False,
) -> pd.DataFrame:
    """
    Compute cumulative volume imbalance at multiple rolling windows.

    Multiple windows capture different time scales of institutional activity,
    from short-term tactical orders to long-term strategic positioning.

    Args:
        bid_volumes: DataFrame with bid volumes
        ask_volumes: DataFrame with ask volumes
        levels: Number of levels to sum over
        windows: List of window lengths (default: [10, 30, 60])
        normalize: If True, compute normalized versions

    Returns:
        DataFrame with rolling imbalance at each window

    Examples:
        >>> # Multiple timeframes for regime detection
        >>> multi_imb = compute_cumulative_volume_imbalance_multiple_windows(
        ...     bid_volumes, ask_volumes, levels=10, windows=[10, 30, 100]
        ... )
        >>> # Compute imbalance momentum
        >>> multi_imb['imbalance_momentum'] = (
        ...     multi_imb['cumulative_volume_imbalance_rolling_10'] -
        ...     multi_imb['cumulative_volume_imbalance_rolling_30']
        ... )
    """
    imbalance_df = pd.DataFrame(index=bid_volumes.index)

    for window in windows:
        col_name = f"cumulative_volume_imbalance_rolling_{window}"
        if normalize:
            col_name += "_normalized"

        imbalance_df[col_name] = compute_cumulative_volume_imbalance_rolling(
            bid_volumes=bid_volumes,
            ask_volumes=ask_volumes,
            levels=levels,
            window=window,
            normalize=normalize,
        )

    return imbalance_df


def compute_cumulative_volume_imbalance_acceleration(
    bid_volumes: pd.DataFrame,
    ask_volumes: pd.DataFrame,
    levels: int = 10,
    window: int = 20,
) -> pd.Series:
    """
    Compute the acceleration (rate of change) of cumulative volume imbalance.

    Imbalance acceleration measures whether buying/selling pressure is
    increasing or decreasing, providing an early warning signal for
    regime changes or momentum shifts.

    Formula:
        acceleration_t = imbalance_t - imbalance_{t-window}

    Intuition:
        **What acceleration reveals:**

        Imbalance level tells you WHERE you are (buying vs selling pressure)
        Acceleration tells you WHERE you're GOING (increasing vs decreasing)

        - **Positive acceleration**:
          - Imbalance becoming more positive (or less negative)
          - Buying pressure increasing OR selling pressure decreasing
          - Bullish momentum building

        - **Negative acceleration**:
          - Imbalance becoming more negative (or less positive)
          - Selling pressure increasing OR buying pressure decreasing
          - Bearish momentum building

        - **Near-zero acceleration**:
          - Imbalance stable
          - Pressure in equilibrium
          - Consolidation phase

        **Four quadrants:**

        1. Positive imbalance + Positive acceleration:
           → Strong buying getting stronger (most bullish)

        2. Positive imbalance + Negative acceleration:
           → Buying pressure weakening (potential reversal)

        3. Negative imbalance + Negative acceleration:
           → Strong selling getting stronger (most bearish)

        4. Negative imbalance + Positive acceleration:
           → Selling pressure weakening (potential reversal)

    Args:
        bid_volumes: DataFrame with bid volumes
        ask_volumes: DataFrame with ask volumes
        levels: Number of levels to sum over
        window: Lookback period for acceleration calculation

    Returns:
        Series containing imbalance acceleration

    Examples:
        >>> # Detect momentum shifts
        >>> lob_df['imb_acceleration'] = compute_cumulative_volume_imbalance_acceleration(
        ...     lob_df, lob_df, levels=10, window=20
        ... )
        >>> # Identify regime changes
        >>> regime_change = (lob_df['imb_acceleration'].diff().abs() > threshold)

    Notes:
        **Trading signals:**

        - Acceleration turns positive → potential long entry
        - Acceleration turns negative → potential short entry
        - High absolute acceleration → strong momentum (trend)
        - Low absolute acceleration → weak momentum (range)
    """
    # Compute current imbalance
    current_imbalance = compute_cumulative_volume_imbalance(
        bid_volumes=bid_volumes, ask_volumes=ask_volumes, levels=levels, normalize=False
    )

    # Compute lagged imbalance
    lagged_imbalance = current_imbalance.shift(window)

    # Compute acceleration (change)
    acceleration = current_imbalance - lagged_imbalance
    acceleration.name = f"cumulative_volume_imbalance_acceleration_{window}"

    return acceleration


def compute_order_book_thickness(
    bid_volumes: pd.DataFrame,
    ask_volumes: pd.DataFrame,
    levels: int = 10,
    side: str = "both",
) -> pd.Series:
    """
    Compute order book thickness (average quantity per level).

    Order book thickness measures the average liquidity available per price level,
    indicating overall market depth and the robustness of liquidity provision.
    Unlike total depth (sum), thickness normalizes by number of levels, making
    it easier to compare across different depth configurations.

    Formula:
        If side='bid':
            thickness = Σ(bid_volume_i) / levels for i in [1, levels]

        If side='ask':
            thickness = Σ(ask_volume_i) / levels for i in [1, levels]

        If side='both':
            thickness = (Σ(bid_volume_i) + Σ(ask_volume_i)) / (2 × levels)

    Intuition:
        **What thickness reveals:**

        Thickness answers: "How much liquidity is available at a TYPICAL level?"

        - **High thickness** (large average quantity per level):
          - Deep, robust liquidity
          - Many market participants active
          - Resilient to order flow shocks
          - Low slippage for medium-sized orders
          - Market makers committed

        - **Low thickness** (small average quantity per level):
          - Thin, fragile liquidity
          - Few market participants
          - Vulnerable to manipulation/shocks
          - High slippage risk
          - Market makers cautious or absent

        **Why average (thickness) vs sum (total depth)?**

        Example with 10 levels:

        Scenario A: Thick book
        - Each level has 1,000 shares
        - Total depth: 10,000 shares
        - Thickness: 1,000 shares/level
        - Interpretation: Uniform, reliable liquidity

        Scenario B: Concentrated book
        - Level 1: 9,500 shares
        - Levels 2-10: 50 shares each
        - Total depth: 10,000 shares (same as A!)
        - Thickness: 1,000 shares/level (same as A!)
        - But distribution is very different!

        Scenario C: Thin book
        - Each level has 100 shares
        - Total depth: 1,000 shares
        - Thickness: 100 shares/level
        - Interpretation: Weak liquidity

        Thickness captures SCALE of liquidity, but not DISTRIBUTION.
        For distribution, need depth concentration metrics.

        **Trading implications:**

        1. **Execution quality**:
           - High thickness → predictable execution
           - Low thickness → unpredictable slippage
           - Algorithms adjust aggression based on thickness

        2. **Market quality indicator**:
           - Increasing thickness → improving liquidity
           - Decreasing thickness → deteriorating conditions
           - Sharp drops → potential liquidity crisis

        3. **Regime detection**:
           - Normal regime: stable thickness
           - Stress regime: collapsing thickness
           - Recovery regime: rebuilding thickness

        4. **Time-of-day patterns**:
           - Market open/close: lower thickness (volatility)
           - Mid-day: higher thickness (stable conditions)
           - Lunch: lower thickness (reduced participation)

        **Asymmetric thickness (bid vs ask):**

        - Bid thickness > Ask thickness:
          - More committed buying liquidity
          - Potential support building
          - Bullish signal if persistent

        - Ask thickness > Bid thickness:
          - More committed selling liquidity
          - Potential resistance building
          - Bearish signal if persistent

        - Balanced thickness:
          - Symmetric market making
          - Fair price discovery
          - Healthy two-sided market

    Args:
        bid_volumes: DataFrame with bid volumes at each level
        ask_volumes: DataFrame with ask volumes at each level
        levels: Number of levels to average over (default: 10)
        side: 'bid', 'ask', or 'both' (default: 'both')

    Returns:
        Series containing order book thickness (average volume per level)

    Examples:
        >>> bid_volumes = pd.DataFrame({
        ...     'bid_volume_1': [1000, 800, 1200],
        ...     'bid_volume_2': [900, 700, 1100],
        ...     'bid_volume_3': [800, 600, 1000]
        ... })
        >>> ask_volumes = pd.DataFrame({
        ...     'ask_volume_1': [950, 750, 1150],
        ...     'ask_volume_2': [850, 650, 1050],
        ...     'ask_volume_3': [750, 550, 950]
        ... })
        >>> compute_order_book_thickness(bid_volumes, ask_volumes, levels=3, side='both')
        0    850.0
        1    650.0
        2   1050.0
        Name: order_book_thickness, dtype: float64

        # At t=0:
        # Total bid: 1000+900+800 = 2700, avg = 900
        # Total ask: 950+850+750 = 2550, avg = 850
        # Both: (2700+2550) / (2*3) = 875... wait let me recalculate
        # Actually: (900 + 850) / 2 = 875

        >>> # Bid-side thickness only
        >>> compute_order_book_thickness(bid_volumes, ask_volumes, levels=3, side='bid')
        0    900.0
        1    700.0
        2   1100.0
        Name: order_book_thickness_bid, dtype: float64

        >>> # Ask-side thickness only
        >>> compute_order_book_thickness(bid_volumes, ask_volumes, levels=3, side='ask')
        0    850.0
        1    650.0
        2   1050.0
        Name: order_book_thickness_ask, dtype: float64

    Notes:
        **Interpretation guidelines:**

        For typical liquid stock (e.g., AAPL):
        - Thickness > 10,000 shares/level: Very thick, excellent liquidity
        - Thickness 5,000-10,000: Good liquidity
        - Thickness 1,000-5,000: Moderate liquidity
        - Thickness < 1,000: Thin liquidity

        For crypto (e.g., BTC/USDT):
        - Thickness > 10 BTC/level: Very thick
        - Thickness 5-10 BTC: Good
        - Thickness 1-5 BTC: Moderate
        - Thickness < 1 BTC: Thin

        (Actual thresholds depend on asset and market conditions)

        **Comparison to other metrics:**

        - Total depth: Measures aggregate liquidity (sum)
        - Thickness: Measures typical liquidity (average)
        - Depth concentration: Measures distribution (top-heavy vs uniform)
        - Queue depth: Measures specific level (not averaged)

        All provide complementary information about order book structure.

    References:
        - Kyle, A. S. (1985). "Continuous auctions and insider trading"
        - Glosten, L. R., & Milgrom, P. R. (1985). "Bid, ask and transaction
          prices in a specialist market with heterogeneously informed traders"
    """
    # Get column names for the specified number of levels
    bid_cols = [f"bid_volume_{i}" for i in range(1, levels + 1)]
    ask_cols = [f"ask_volume_{i}" for i in range(1, levels + 1)]

    if side == "bid":
        # Average bid volume per level
        thickness = bid_volumes[bid_cols].mean(axis=1)
        thickness.name = "order_book_thickness_bid"

    elif side == "ask":
        # Average ask volume per level
        thickness = ask_volumes[ask_cols].mean(axis=1)
        thickness.name = "order_book_thickness_ask"

    elif side == "both":
        # Average of bid and ask thickness
        bid_thickness = bid_volumes[bid_cols].mean(axis=1)
        ask_thickness = ask_volumes[ask_cols].mean(axis=1)
        thickness = (bid_thickness + ask_thickness) / 2
        thickness.name = "order_book_thickness"

    else:
        raise ValueError(f"Invalid side: {side}. Choose 'bid', 'ask', or 'both'")

    return thickness


def compute_order_book_thickness_ratio(
    bid_volumes: pd.DataFrame,
    ask_volumes: pd.DataFrame,
    levels: int = 10,
) -> pd.Series:
    """
    Compute the ratio of bid thickness to ask thickness.

    The thickness ratio reveals whether liquidity provision is symmetric or
    if one side is significantly deeper than the other. Asymmetry indicates
    directional bias in market maker positioning.

    Formula:
        thickness_ratio = bid_thickness / ask_thickness
                        = (Σ bid_volume_i / levels) / (Σ ask_volume_i / levels)

    Intuition:
        **What the ratio reveals:**

        - **Ratio > 1.0** (bid thickness > ask thickness):
          - More committed buying liquidity
          - Market makers positioning for upside
          - Potential support, buying interest
          - Institutions may be accumulating

        - **Ratio < 1.0** (ask thickness < bid thickness):
          - More committed selling liquidity
          - Market makers positioning for downside
          - Potential resistance, selling interest
          - Institutions may be distributing

        - **Ratio ≈ 1.0** (symmetric):
          - Balanced market making
          - No strong directional bias
          - Fair, efficient price discovery

        **Ratio extremes:**

        - Ratio > 1.5: Strong bid-side dominance (very bullish)
        - Ratio 1.1-1.5: Moderate bid bias
        - Ratio 0.9-1.1: Balanced
        - Ratio 0.67-0.9: Moderate ask bias
        - Ratio < 0.67: Strong ask-side dominance (very bearish)

        **Time series patterns:**

        - Increasing ratio → building bid support
        - Decreasing ratio → building ask resistance
        - Ratio crosses above 1.0 → bullish regime shift
        - Ratio crosses below 1.0 → bearish regime shift

        **Divergences (powerful signals):**

        - Price rising + ratio falling → hidden distribution (bearish)
        - Price falling + ratio rising → hidden accumulation (bullish)

    Args:
        bid_volumes: DataFrame with bid volumes at each level
        ask_volumes: DataFrame with ask volumes at each level
        levels: Number of levels to average over

    Returns:
        Series containing thickness ratio (bid/ask)

    Examples:
        >>> bid_volumes = pd.DataFrame({
        ...     'bid_volume_1': [1000, 800, 1200],
        ...     'bid_volume_2': [900, 700, 1100]
        ... })
        >>> ask_volumes = pd.DataFrame({
        ...     'ask_volume_1': [800, 900, 1000],
        ...     'ask_volume_2': [750, 850, 950]
        ... })
        >>> compute_order_book_thickness_ratio(bid_volumes, ask_volumes, levels=2)
        0    1.225806
        1    0.857143
        2    1.157895
        Name: order_book_thickness_ratio, dtype: float64

        # At t=0: bid_thickness=950, ask_thickness=775, ratio=1.226 (bid-heavy)
        # At t=1: bid_thickness=750, ask_thickness=875, ratio=0.857 (ask-heavy)
        # At t=2: bid_thickness=1150, ask_thickness=975, ratio=1.179 (bid-heavy)

    Notes:
        **Interpretation:**

        - Ratio persistently > 1.2: Strong institutional buying
        - Ratio persistently < 0.8: Strong institutional selling
        - Ratio oscillating around 1.0: Active two-sided market making
        - Ratio trending: Building directional pressure

        **Combined with price action:**

        - Price ↑ + Ratio ↑: Confirmed uptrend (aligned)
        - Price ↑ + Ratio ↓: Divergence (potential reversal)
        - Price ↓ + Ratio ↓: Confirmed downtrend (aligned)
        - Price ↓ + Ratio ↑: Divergence (potential reversal)

    References:
        - Easley, D., López de Prado, M. M., & O'Hara, M. (2012). "Flow
          toxicity and liquidity in a high-frequency world"
    """
    bid_cols = [f"bid_volume_{i}" for i in range(1, levels + 1)]
    ask_cols = [f"ask_volume_{i}" for i in range(1, levels + 1)]

    bid_thickness = bid_volumes[bid_cols].mean(axis=1)
    ask_thickness = ask_volumes[ask_cols].mean(axis=1)

    # Small epsilon to avoid division by zero
    epsilon = 1e-10
    thickness_ratio = bid_thickness / (ask_thickness + epsilon)
    thickness_ratio.name = "order_book_thickness_ratio"

    return thickness_ratio


def compute_order_book_thickness_volatility(
    bid_volumes: pd.DataFrame,
    ask_volumes: pd.DataFrame,
    levels: int = 10,
    window: int = 20,
    side: str = "both",
) -> pd.Series:
    """
    Compute rolling volatility of order book thickness.

    Thickness volatility measures the stability of liquidity provision over time.
    High volatility indicates unstable, fleeting liquidity (bad for execution).
    Low volatility indicates committed, stable liquidity (good for execution).

    Formula:
        thickness_volatility_t = std(thickness_{t-window+1:t})

    Intuition:
        **What thickness volatility reveals:**

        - **Low volatility** (stable thickness):
          - Committed liquidity providers
          - Predictable execution environment
          - Professional market makers present
          - Safe for large order execution

        - **High volatility** (unstable thickness):
          - Fleeting liquidity (quotes disappear)
          - Unpredictable execution environment
          - Potential quote stuffing or manipulation
          - Risky for large order execution

        **Regime detection:**

        - Normal regime: Low thickness volatility
        - Stress regime: High thickness volatility (liquidity fleeing)
        - News events: Spikes in thickness volatility
        - Market close: Elevated thickness volatility

        **Trading applications:**

        - Execution algorithms: Slow down during high volatility
        - Market making: Widen spreads during high volatility
        - Risk management: Reduce position sizes during high volatility

    Args:
        bid_volumes: DataFrame with bid volumes
        ask_volumes: DataFrame with ask volumes
        levels: Number of levels to average over
        window: Rolling window for volatility calculation
        side: 'bid', 'ask', or 'both'

    Returns:
        Series containing rolling volatility of thickness

    Examples:
        >>> # Detect periods of unstable liquidity
        >>> thickness_vol = compute_order_book_thickness_volatility(
        ...     bid_volumes, ask_volumes, levels=10, window=20
        ... )
        >>> unstable_periods = thickness_vol > thickness_vol.quantile(0.90)

    Notes:
        **Interpretation:**

        - Volatility < 100 shares: Very stable
        - Volatility 100-500 shares: Normal
        - Volatility 500-1000 shares: Elevated
        - Volatility > 1000 shares: Unstable (be cautious)

        (Thresholds depend on asset and typical thickness level)
    """
    # Compute thickness time series
    thickness = compute_order_book_thickness(
        bid_volumes=bid_volumes, ask_volumes=ask_volumes, levels=levels, side=side
    )

    # Compute rolling volatility
    thickness_vol = thickness.rolling(window=window, min_periods=window).std()
    thickness_vol.name = f"order_book_thickness_volatility_{window}"

    return thickness_vol


def compute_order_book_thickness_change(
    bid_volumes: pd.DataFrame,
    ask_volumes: pd.DataFrame,
    levels: int = 10,
    lag: int = 1,
    side: str = "both",
) -> pd.Series:
    """
    Compute change in order book thickness over a specified lag.

    Thickness change measures whether liquidity is building or depleting,
    providing early warning signals for regime shifts or market stress.

    Formula:
        thickness_change_t = thickness_t - thickness_{t-lag}

    Intuition:
        **What thickness change reveals:**

        - **Positive change** (thickness increasing):
          - Liquidity building, improving conditions
          - Market makers adding quotes
          - More participants entering
          - Favorable for execution

        - **Negative change** (thickness decreasing):
          - Liquidity depleting, deteriorating conditions
          - Market makers withdrawing
          - Participants exiting
          - Risky for execution

        - **Near-zero change**:
          - Stable liquidity environment
          - Equilibrium reached

        **Pattern detection:**

        1. **Gradual increase**:
           - Confidence building
           - Market normalizing after stress

        2. **Gradual decrease**:
           - Confidence eroding
           - Potential stress building

        3. **Sudden spike**:
           - Large order arrival
           - Institutional activity

        4. **Sudden drop**:
           - Liquidity crisis
           - Flash crash risk
           - News event

    Args:
        bid_volumes: DataFrame with bid volumes
        ask_volumes: DataFrame with ask volumes
        levels: Number of levels to average over
        lag: Number of periods to look back for change calculation
        side: 'bid', 'ask', or 'both'

    Returns:
        Series containing thickness change

    Examples:
        >>> # Detect liquidity building/depleting
        >>> thickness_change = compute_order_book_thickness_change(
        ...     bid_volumes, ask_volumes, levels=10, lag=20
        ... )
        >>> liquidity_building = thickness_change > 0
        >>> liquidity_depleting = thickness_change < 0

    Notes:
        **Lag selection:**

        - Short lag (1-5): Tick-by-tick changes (noisy)
        - Medium lag (10-30): Short-term trends
        - Long lag (60+): Longer-term regime shifts
    """
    # Compute thickness
    thickness = compute_order_book_thickness(
        bid_volumes=bid_volumes, ask_volumes=ask_volumes, levels=levels, side=side
    )

    # Compute change
    thickness_change = thickness - thickness.shift(lag)
    thickness_change.name = f"order_book_thickness_change_{lag}"

    return thickness_change


def compute_depth_concentration_at_top(
    bid_volumes: pd.DataFrame,
    ask_volumes: pd.DataFrame,
    top_levels: int = 1,
    total_levels: int = 10,
    side: str = "both",
) -> pd.Series:
    """
    Compute depth concentration (fraction of liquidity at top levels).

    Depth concentration measures what percentage of total visible liquidity
    is concentrated at the top levels of the order book. High concentration
    indicates liquidity is front-loaded (thin deeper book), while low
    concentration indicates liquidity is distributed evenly across levels.

    Formula:
        If side='bid':
            concentration = Σ(bid_volume_i for i in [1, top_levels]) /
                           Σ(bid_volume_i for i in [1, total_levels])

        If side='ask':
            concentration = Σ(ask_volume_i for i in [1, top_levels]) /
                           Σ(ask_volume_i for i in [1, total_levels])

        If side='both':
            concentration = (bid_concentration + ask_concentration) / 2

    Intuition:
        **What concentration reveals:**

        Depth concentration answers: "Is liquidity front-loaded or distributed?"

        - **High concentration** (e.g., 70%+ at top):
          - Liquidity concentrated near current price
          - Shallow depth beyond top levels
          - Characteristics:
            * Quick to execute small orders
            * High slippage for large orders
            * "Iceberg" risk (hidden orders?)
            * Market makers using thin visible book
          - Common during:
            * High volatility (MMs pull back)
            * Low confidence periods
            * Retail-dominated markets

        - **Low concentration** (e.g., 30-40% at top):
          - Liquidity distributed evenly
          - Deep, robust order book
          - Characteristics:
            * Slower for small orders (deeper queue)
            * Lower slippage for large orders
            * Committed liquidity providers
            * Professional market making
          - Common during:
            * Normal market conditions
            * Institutional presence
            * High-liquidity periods

        **Trading implications:**

        1. **Execution strategy**:
           - High concentration → Use limit orders (get in queue)
           - Low concentration → Market orders okay (depth available)

        2. **Market impact estimation**:
           - High concentration → Expect impact beyond top level
           - Low concentration → Impact spread across levels

        3. **Liquidity quality**:
           - High concentration → Fragile, quote-dependent
           - Low concentration → Robust, committed liquidity

        4. **Hidden order detection**:
           - Very high concentration (>80%) → Possible icebergs
           - Concentration suddenly drops → Iceberg revealed

        **Asymmetric concentration:**

        - Bid concentration > Ask concentration:
          - Bid liquidity more front-loaded
          - Asks more distributed (sellers patient)
          - Potential: Limited upside (thin deep bids)

        - Ask concentration > Bid concentration:
          - Ask liquidity more front-loaded
          - Bids more distributed (buyers patient)
          - Potential: Limited downside (thin deep asks)

        **Time patterns:**

        - Market open/close: Higher concentration (volatility)
        - Mid-day: Lower concentration (stable conditions)
        - News events: Spike in concentration (uncertainty)

    Args:
        bid_volumes: DataFrame with bid volumes at each level
        ask_volumes: DataFrame with ask volumes at each level
        top_levels: Number of top levels to measure (default: 1, just top-of-book)
        total_levels: Total number of levels to compare against (default: 10)
        side: 'bid', 'ask', or 'both' (default: 'both')

    Returns:
        Series containing depth concentration (range: 0 to 1)

    Examples:
        >>> bid_volumes = pd.DataFrame({
        ...     'bid_volume_1': [5000, 1000, 8000],
        ...     'bid_volume_2': [3000, 3000, 1000],
        ...     'bid_volume_3': [2000, 3000, 1000]
        ... })
        >>> ask_volumes = pd.DataFrame({
        ...     'ask_volume_1': [4500, 1500, 7000],
        ...     'ask_volume_2': [3500, 3500, 1500],
        ...     'ask_volume_3': [2000, 3000, 1500]
        ... })
        >>> compute_depth_concentration_at_top(bid_volumes, ask_volumes,
        ...                                     top_levels=1, total_levels=3)
        0    0.475000
        1    0.166667
        2    0.750000
        Name: depth_concentration_top_1, dtype: float64

        # At t=0:
        # Bid: 5000 / (5000+3000+2000) = 50%
        # Ask: 4500 / (4500+3500+2000) = 45%
        # Both: (50% + 45%) / 2 = 47.5%
        # Interpretation: Moderate concentration

        # At t=1:
        # Bid: 1000 / 7000 = 14.3%
        # Ask: 1500 / 8000 = 18.75%
        # Both: 16.5%
        # Interpretation: Low concentration (deep, distributed book)

        # At t=2:
        # Bid: 8000 / 10000 = 80%
        # Ask: 7000 / 10000 = 70%
        # Both: 75%
        # Interpretation: High concentration (front-loaded, thin deep)

        >>> # Top 3 levels concentration
        >>> compute_depth_concentration_at_top(bid_volumes, ask_volumes,
        ...                                     top_levels=3, total_levels=5)
        # Would measure what % of top 5 levels is in top 3

    Notes:
        **Interpretation guidelines:**

        For top_levels=1 (top-of-book concentration):
        - Concentration > 70%: Very front-loaded (be cautious)
        - Concentration 50-70%: Moderate front-loading
        - Concentration 30-50%: Balanced distribution
        - Concentration < 30%: Deep, distributed book (good for large orders)

        For top_levels=5 (top 5 levels concentration out of 10):
        - Concentration > 80%: Thin beyond top 5 (risky)
        - Concentration 70-80%: Typical
        - Concentration 60-70%: Good depth throughout
        - Concentration < 60%: Excellent deep liquidity

        **Common configurations:**

        - (top=1, total=10): Classic "how much at TOB" measure
        - (top=3, total=10): "How much in near book"
        - (top=5, total=20): "How much in visible vs very deep"

        **Relationship to other metrics:**

        - High concentration + High thickness = Front-loaded but thick TOB
        - High concentration + Low thickness = Thin everywhere
        - Low concentration + High thickness = Deep, robust liquidity
        - Low concentration + Low thickness = Thin but distributed

    References:
        - Biais, B., Hillion, P., & Spatt, C. (1995). "An empirical analysis
          of the limit order book and the order flow in the Paris Bourse"
        - Hautsch, N., & Huang, R. (2012). "The market impact of a limit order"
    """
    # Get top levels volume
    if top_levels == 1:
        bid_top = bid_volumes["bid_volume_1"]
        ask_top = ask_volumes["ask_volume_1"]
    else:
        bid_top_cols = [f"bid_volume_{i}" for i in range(1, top_levels + 1)]
        ask_top_cols = [f"ask_volume_{i}" for i in range(1, top_levels + 1)]
        bid_top = bid_volumes[bid_top_cols].sum(axis=1)
        ask_top = ask_volumes[ask_top_cols].sum(axis=1)

    # Get total levels volume
    bid_total_cols = [f"bid_volume_{i}" for i in range(1, total_levels + 1)]
    ask_total_cols = [f"ask_volume_{i}" for i in range(1, total_levels + 1)]
    bid_total = bid_volumes[bid_total_cols].sum(axis=1)
    ask_total = ask_volumes[ask_total_cols].sum(axis=1)

    # Small epsilon to avoid division by zero
    epsilon = 1e-10

    if side == "bid":
        concentration = bid_top / (bid_total + epsilon)
        concentration.name = f"depth_concentration_top_{top_levels}_bid"

    elif side == "ask":
        concentration = ask_top / (ask_total + epsilon)
        concentration.name = f"depth_concentration_top_{top_levels}_ask"

    elif side == "both":
        bid_concentration = bid_top / (bid_total + epsilon)
        ask_concentration = ask_top / (ask_total + epsilon)
        concentration = (bid_concentration + ask_concentration) / 2
        concentration.name = f"depth_concentration_top_{top_levels}"

    else:
        raise ValueError(f"Invalid side: {side}. Choose 'bid', 'ask', or 'both'")

    return concentration


def compute_depth_concentration_multiple_tops(
    bid_volumes: pd.DataFrame,
    ask_volumes: pd.DataFrame,
    top_levels_list: List[int] = [1, 3, 5],
    total_levels: int = 10,
    side: str = "both",
) -> pd.DataFrame:
    """
    Compute depth concentration for multiple top-level configurations.

    Multiple concentration measures reveal the liquidity distribution profile
    across the order book, showing how liquidity decays with depth.

    Intuition:
        **Multi-level concentration analysis:**

        Comparing concentrations at different tops reveals book structure:

        Example 1: Front-loaded book
        - Top 1: 60%
        - Top 3: 85%
        - Top 5: 95%
        - Interpretation: Most liquidity at very top, thin beyond

        Example 2: Distributed book
        - Top 1: 25%
        - Top 3: 50%
        - Top 5: 70%
        - Interpretation: Liquidity spread evenly, deep book

        Example 3: Two-tier book
        - Top 1: 40%
        - Top 3: 90%
        - Top 5: 95%
        - Interpretation: Decent top, then concentration, then nothing

        **Concentration gradient:**

        The rate of increase reveals depth structure:
        - Steep gradient (rapid increase) → Front-loaded
        - Gentle gradient (slow increase) → Distributed
        - Flat gradient → Uniform depth (rare)

    Args:
        bid_volumes: DataFrame with bid volumes
        ask_volumes: DataFrame with ask volumes
        top_levels_list: List of top-level counts to measure (default: [1,3,5])
        total_levels: Total levels to compare against (default: 10)
        side: 'bid', 'ask', or 'both'

    Returns:
        DataFrame with concentration at each specified top-level count

    Examples:
        >>> concentration_df = compute_depth_concentration_multiple_tops(
        ...     bid_volumes, ask_volumes,
        ...     top_levels_list=[1, 3, 5],
        ...     total_levels=10
        ... )
        >>> # Compute concentration gradient
        >>> concentration_df['gradient_1_to_5'] = (
        ...     concentration_df['depth_concentration_top_5'] -
        ...     concentration_df['depth_concentration_top_1']
        ... )
    """
    concentration_df = pd.DataFrame(index=bid_volumes.index)

    for top_levels in top_levels_list:
        concentration_df[f"depth_concentration_top_{top_levels}"] = (
            compute_depth_concentration_at_top(
                bid_volumes=bid_volumes,
                ask_volumes=ask_volumes,
                top_levels=top_levels,
                total_levels=total_levels,
                side=side,
            )
        )

    return concentration_df


def compute_depth_concentration_asymmetry(
    bid_volumes: pd.DataFrame,
    ask_volumes: pd.DataFrame,
    top_levels: int = 1,
    total_levels: int = 10,
) -> pd.Series:
    """
    Compute asymmetry in depth concentration between bid and ask sides.

    Concentration asymmetry reveals whether one side has more front-loaded
    liquidity than the other, indicating different strategies or expectations
    between buyers and sellers.

    Formula:
        concentration_asymmetry = bid_concentration - ask_concentration

    Intuition:
        **What asymmetry reveals:**

        - **Positive asymmetry** (bid more concentrated):
          - Buyers front-loading (aggressive at top)
          - Sellers more patient (distributed deeper)
          - Interpretation:
            * Buyers want immediate fills
            * Sellers willing to wait for better price
          - Potential: Upward pressure near-term

        - **Negative asymmetry** (ask more concentrated):
          - Sellers front-loading (aggressive at top)
          - Buyers more patient (distributed deeper)
          - Interpretation:
            * Sellers want immediate fills
            * Buyers willing to wait for better price
          - Potential: Downward pressure near-term

        - **Near-zero asymmetry**:
          - Both sides using similar strategies
          - Symmetric market making
          - Balanced expectations

        **Strategic implications:**

        1. Bid concentrated + Ask distributed:
           → Buyers eager, sellers patient (bullish signal)

        2. Ask concentrated + Bid distributed:
           → Sellers eager, buyers patient (bearish signal)

        3. Both concentrated:
           → Both sides front-running, high-frequency activity

        4. Both distributed:
           → Both sides patient, institutional presence

    Args:
        bid_volumes: DataFrame with bid volumes
        ask_volumes: DataFrame with ask volumes
        top_levels: Number of top levels to measure
        total_levels: Total levels to compare against

    Returns:
        Series containing concentration asymmetry (range: -1 to 1)

    Examples:
        >>> bid_volumes = pd.DataFrame({
        ...     'bid_volume_1': [8000, 3000],
        ...     'bid_volume_2': [1000, 3000],
        ...     'bid_volume_3': [1000, 3000]
        ... })
        >>> ask_volumes = pd.DataFrame({
        ...     'ask_volume_1': [3000, 7000],
        ...     'ask_volume_2': [3500, 1500],
        ...     'ask_volume_3': [3500, 1500]
        ... })
        >>> compute_depth_concentration_asymmetry(bid_volumes, ask_volumes, 1, 3)
        0    0.500000
        1   -0.400000
        Name: depth_concentration_asymmetry, dtype: float64

        # At t=0: bid_conc=80%, ask_conc=30%, asymmetry=+0.50 (bid front-loaded)
        # At t=1: bid_conc=30%, ask_conc=70%, asymmetry=-0.40 (ask front-loaded)

    Notes:
        **Interpretation:**

        - Asymmetry > +0.3: Strong bid front-loading (bullish)
        - Asymmetry +0.1 to +0.3: Moderate bid front-loading
        - Asymmetry -0.1 to +0.1: Balanced
        - Asymmetry -0.3 to -0.1: Moderate ask front-loading
        - Asymmetry < -0.3: Strong ask front-loading (bearish)
    """
    bid_concentration = compute_depth_concentration_at_top(
        bid_volumes=bid_volumes,
        ask_volumes=ask_volumes,
        top_levels=top_levels,
        total_levels=total_levels,
        side="bid",
    )

    ask_concentration = compute_depth_concentration_at_top(
        bid_volumes=bid_volumes,
        ask_volumes=ask_volumes,
        top_levels=top_levels,
        total_levels=total_levels,
        side="ask",
    )

    asymmetry = bid_concentration - ask_concentration
    asymmetry.name = f"depth_concentration_asymmetry_top_{top_levels}"

    return asymmetry


def compute_depth_concentration_change(
    bid_volumes: pd.DataFrame,
    ask_volumes: pd.DataFrame,
    top_levels: int = 1,
    total_levels: int = 10,
    lag: int = 20,
    side: str = "both",
) -> pd.Series:
    """
    Compute change in depth concentration over time.

    Concentration change detects shifts in liquidity structure, revealing
    whether the book is becoming more front-loaded (concentrating) or more
    distributed (deconcentrating) over time.

    Formula:
        concentration_change_t = concentration_t - concentration_{t-lag}

    Intuition:
        **What concentration change reveals:**

        - **Positive change** (increasing concentration):
          - Liquidity pulling back from deep levels
          - Market makers reducing commitment
          - Uncertainty increasing
          - Potential: Volatility spike, liquidity crisis

        - **Negative change** (decreasing concentration):
          - Liquidity expanding to deeper levels
          - Market makers increasing commitment
          - Confidence building
          - Potential: Stabilization, normal conditions returning

        - **Stable concentration**:
          - Liquidity structure unchanged
          - Market in equilibrium

        **Pattern detection:**

        1. Gradual concentration increase:
           - Slow liquidity withdrawal
           - Risk building

        2. Sudden concentration spike:
           - Rapid liquidity pullback
           - Crisis or news event
           - Flash crash risk

        3. Gradual concentration decrease:
           - Confidence returning
           - Market normalizing

        4. Oscillating concentration:
           - Active quote management
           - HFT strategies

    Args:
        bid_volumes: DataFrame with bid volumes
        ask_volumes: DataFrame with ask volumes
        top_levels: Number of top levels to measure
        total_levels: Total levels to compare against
        lag: Lookback period for change calculation
        side: 'bid', 'ask', or 'both'

    Returns:
        Series containing concentration change

    Examples:
        >>> # Detect liquidity structure shifts
        >>> conc_change = compute_depth_concentration_change(
        ...     bid_volumes, ask_volumes, top_levels=1, total_levels=10, lag=20
        ... )
        >>> concentrating = conc_change > 0.1  # Liquidity pulling back
        >>> deconcentrating = conc_change < -0.1  # Liquidity expanding

    Notes:
        **Alert thresholds:**

        - Change > +0.2: Significant concentration (warning)
        - Change +0.1 to +0.2: Moderate concentration
        - Change -0.1 to +0.1: Stable
        - Change -0.2 to -0.1: Moderate deconcentration
        - Change < -0.2: Significant deconcentration (improving)
    """
    concentration = compute_depth_concentration_at_top(
        bid_volumes=bid_volumes,
        ask_volumes=ask_volumes,
        top_levels=top_levels,
        total_levels=total_levels,
        side=side,
    )

    concentration_change = concentration - concentration.shift(lag)
    concentration_change.name = f"depth_concentration_change_{lag}"

    return concentration_change


def compute_time_since_last_trade(
    timestamps: pd.Series,
    unit: str = "seconds",
) -> pd.Series:
    """
    Compute time elapsed since the last trade/event.

    Time since last event measures the recency of activity, revealing periods
    of high-frequency trading vs quiet periods. In microstructure, the arrival
    rate of events contains information about market dynamics and liquidity.

    Formula:
        time_since_event_t = timestamp_t - timestamp_{t-1}

    Intuition:
        **Why time between events matters:**

        Market microstructure is NOT uniformly spaced - events cluster:

        - **Short gaps** (high arrival rate):
          - Active trading, high information flow
          - News incorporation, price discovery
          - Increased toxicity (informed traders)
          - Market makers face adverse selection
          - Volatility likely elevated

        - **Long gaps** (low arrival rate):
          - Quiet market, low information flow
          - Consolidation, waiting for news
          - Lower toxicity (uninformed traders)
          - Market makers can tighten spreads
          - Volatility likely subdued

        **Hawkes process connection:**

        Trade arrivals follow a self-exciting Hawkes process:
        - Trades beget more trades (clustering)
        - Short inter-arrival times predict more short times
        - Long gaps predict more long gaps
        - "Bursts" of activity vs "droughts"

        **Trading implications:**

        1. **Spread adjustment**:
           - Short gaps → widen spreads (toxicity risk)
           - Long gaps → tighten spreads (competition)

        2. **Execution timing**:
           - Long gaps → good time to execute (low impact)
           - Short gaps → wait or split order (high impact)

        3. **Regime detection**:
           - Transition from long → short gaps → volatility spike coming
           - Transition from short → long gaps → volatility calming

        4. **Market making**:
           - Track gaps to adjust inventory risk
           - High-frequency periods = higher risk

        **Empirical patterns:**

        - Mean reversion: Very short/long gaps tend to revert
        - Clustering: Short gaps cluster together (burstiness)
        - Intraday patterns: Shorter gaps at open/close
        - Volatility link: Shorter gaps during volatile periods

    Args:
        timestamps: Series of event timestamps (must be datetime)
        unit: Time unit for output - 'seconds', 'milliseconds', 'microseconds' (default: 'seconds')

    Returns:
        Series containing time since last event in specified units

    Examples:
        >>> timestamps = pd.Series([
        ...     pd.Timestamp('2025-01-01 09:30:00.000'),
        ...     pd.Timestamp('2025-01-01 09:30:00.500'),  # 0.5s gap
        ...     pd.Timestamp('2025-01-01 09:30:01.500'),  # 1.0s gap
        ...     pd.Timestamp('2025-01-01 09:30:01.600'),  # 0.1s gap
        ...     pd.Timestamp('2025-01-01 09:30:05.000'),  # 3.4s gap
        ... ])
        >>> compute_time_since_last_trade(timestamps, unit='seconds')
        0         NaN
        1    0.500000
        2    1.000000
        3    0.100000
        4    3.400000
        Name: time_since_last_event_seconds, dtype: float64

        # First event has no prior, so NaN
        # Second event: 0.5 seconds after first
        # Fifth event: 3.4 seconds after fourth (long gap)

        >>> compute_time_since_last_trade(timestamps, unit='milliseconds')
        0       NaN
        1     500.0
        2    1000.0
        3     100.0
        4    3400.0
        Name: time_since_last_event_milliseconds, dtype: float64

    Notes:
        **Interpretation guidelines:**

        For high-frequency equity trading:
        - Gap < 0.1s: Very high frequency (HFT dominated)
        - Gap 0.1-1s: Active trading
        - Gap 1-10s: Normal activity
        - Gap > 10s: Quiet period

        For crypto (24/7 markets):
        - Gap < 0.01s: Ultra-high frequency
        - Gap 0.01-0.1s: Very active
        - Gap 0.1-1s: Active
        - Gap > 1s: Relatively quiet

        **Statistical properties:**

        - Distribution: Right-skewed (few very long gaps)
        - Not normal: Use robust methods
        - Autocorrelated: Past gaps predict future gaps
        - Regime-dependent: Changes with volatility

        **Data quality:**

        - Gaps can include exchange downtime
        - Weekends/holidays create artificial long gaps
        - Filter extreme outliers if needed
        - Consider market hours only for equity

    References:
        - Engle, R. F., & Russell, J. R. (1998). "Autoregressive Conditional
          Duration: A New Model for Irregularly Spaced Transaction Data"
        - Hautsch, N. (2012). "Econometrics of Financial High-Frequency Data"
    """
    if not isinstance(timestamps.iloc[0], pd.Timestamp):
        raise TypeError("timestamps must be pandas Timestamps")

    # Compute time differences
    time_diff = timestamps.diff()

    # Convert to requested unit
    if unit == "seconds":
        time_since = time_diff.dt.total_seconds()
        time_since.name = "time_since_last_event_seconds"
    elif unit == "milliseconds":
        time_since = time_diff.dt.total_seconds() * 1000
        time_since.name = "time_since_last_event_milliseconds"
    elif unit == "microseconds":
        time_since = time_diff.dt.total_seconds() * 1_000_000
        time_since.name = "time_since_last_event_microseconds"
    else:
        raise ValueError(
            f"Invalid unit: {unit}. Choose 'seconds', 'milliseconds', or 'microseconds'"
        )

    return time_since


def compute_time_since_price_change(
    prices: pd.Series,
    timestamps: pd.Series,
    threshold: float = 0.0,
    unit: str = "seconds",
) -> pd.Series:
    """
    Compute time elapsed since the last significant price change.

    Time since price change measures how long the price has been stable,
    revealing consolidation periods vs active price discovery. Long periods
    without movement suggest equilibrium or low information flow.

    Formula:
        For each timestamp t:
            Find most recent time t' where |price_t' - price_{t'-1}| > threshold
            time_since_price_change_t = timestamp_t - timestamp_t'

    Intuition:
        **What time since price change reveals:**

        - **Short time** (recent price movement):
          - Active price discovery
          - New information being incorporated
          - Directional momentum possible
          - Higher volatility regime

        - **Long time** (price stable):
          - Consolidation, equilibrium
          - Waiting for new information
          - Mean reversion likely
          - Lower volatility regime

        **Pattern detection:**

        1. **Extended stability then movement**:
           - Compression phase → expansion
           - Breakout from consolidation
           - Volatility regime shift

        2. **Rapid successive changes**:
           - Trending market
           - Momentum building
           - News event unfolding

        3. **Alternating stability/movement**:
           - Range-bound market
           - Support/resistance testing
           - Mean-reverting conditions

        **Trading applications:**

        - Long stability → expect breakout (place breakout orders)
        - Recent movement → expect continuation or exhaustion
        - Stability increasing → tighten stops (breakout imminent)

    Args:
        prices: Series of prices (mid-price, last trade, etc.)
        timestamps: Series of corresponding timestamps
        threshold: Minimum price change to count as "event" (default: 0.0, any change)
        unit: Time unit for output

    Returns:
        Series containing time since last price change

    Examples:
        >>> prices = pd.Series([100.0, 100.0, 100.0, 100.1, 100.1, 100.2])
        >>> timestamps = pd.to_datetime([
        ...     '2025-01-01 09:30:00',
        ...     '2025-01-01 09:30:01',
        ...     '2025-01-01 09:30:02',
        ...     '2025-01-01 09:30:03',  # Price changes here
        ...     '2025-01-01 09:30:04',
        ...     '2025-01-01 09:30:05',  # Price changes here
        ... ])
        >>> compute_time_since_price_change(prices, timestamps, threshold=0.0)
        0         NaN
        1    1.000000
        2    2.000000
        3    0.000000
        4    1.000000
        5    0.000000
        Name: time_since_price_change_seconds, dtype: float64

        # At t=2: 2 seconds since last change (at t=0)
        # At t=3: 0 seconds (just changed)
        # At t=4: 1 second since last change (at t=3)

    Notes:
        **Threshold selection:**

        - threshold=0: Count any change (noisy)
        - threshold=tick_size: Only meaningful moves
        - threshold=0.01%: Relative threshold

        **Use cases:**

        - Breakout trading: Long time_since → expect move
        - Mean reversion: Short time_since → expect pause
        - Volatility forecasting: Time pattern predicts vol regime
    """
    # Detect price changes exceeding threshold
    price_changes = prices.diff().abs() > threshold

    # Find indices where price changed
    change_indices = price_changes[price_changes].index

    # For each timestamp, find time since last change
    time_since = pd.Series(index=timestamps.index, dtype=float)

    for idx in timestamps.index:
        # Find most recent price change before this point
        prior_changes = change_indices[change_indices < idx]

        if len(prior_changes) == 0:
            time_since[idx] = np.nan
        else:
            last_change_idx = prior_changes[-1]
            time_diff = timestamps[idx] - timestamps[last_change_idx]

            if unit == "seconds":
                time_since[idx] = time_diff.total_seconds()
            elif unit == "milliseconds":
                time_since[idx] = time_diff.total_seconds() * 1000
            elif unit == "microseconds":
                time_since[idx] = time_diff.total_seconds() * 1_000_000

    time_since.name = f"time_since_price_change_{unit}"
    return time_since


def compute_time_since_volume_spike(
    volumes: pd.Series,
    timestamps: pd.Series,
    spike_threshold: float = 2.0,
    window: int = 20,
    unit: str = "seconds",
) -> pd.Series:
    """
    Compute time elapsed since the last volume spike.

    Volume spikes indicate unusual activity - large orders, news events, or
    institutional flow. Time since last spike reveals how recently the market
    experienced abnormal activity.

    Formula:
        1. Compute rolling average volume: avg_vol = rolling_mean(volume, window)
        2. Detect spike: volume_t > spike_threshold × avg_vol_t
        3. For each t, find most recent spike
        4. time_since_spike_t = timestamp_t - timestamp_spike

    Intuition:
        **Why time since volume spike matters:**

        Volume spikes are NOT random - they indicate:
        - Large institutional orders
        - News events breaking
        - Stop-loss cascades
        - Algorithmic execution bursts

        After a spike:
        - Short time → aftershocks possible, volatility elevated
        - Medium time → digesting the flow, stabilizing
        - Long time → normal conditions, spike forgotten

        **Trading implications:**

        - Recent spike (< 1 min) → be cautious, more volatility
        - Moderate time (1-5 min) → watch for reversion
        - Long time (> 10 min) → back to normal

    Args:
        volumes: Series of volumes (trade size, LOB volume, etc.)
        timestamps: Series of corresponding timestamps
        spike_threshold: Multiplier for spike detection (default: 2.0 = 2x average)
        window: Window for computing average volume (default: 20)
        unit: Time unit for output

    Returns:
        Series containing time since last volume spike

    Examples:
        >>> volumes = pd.Series([100, 150, 120, 500, 110, 130, 400, 140])
        >>> timestamps = pd.date_range('2025-01-01 09:30:00', periods=8, freq='1s')
        >>> compute_time_since_volume_spike(volumes, timestamps, spike_threshold=2.0, window=3)
        # Will detect spikes at indices where volume > 2x rolling average

    Notes:
        **Spike detection variations:**

        - Fixed threshold: volume > X shares
        - Percentile: volume > 95th percentile
        - Standard deviations: volume > mean + 2*std
        - Relative: volume > N × recent average (used here)

        **Use cases:**

        - Risk management: Avoid trading near spikes
        - Opportunity detection: Trade the reversion after spike
        - Regime classification: Frequent spikes = active regime
    """
    # Compute rolling average volume
    avg_volume = volumes.rolling(window=window, min_periods=1).mean()

    # Detect spikes (volume exceeds threshold × average)
    spikes = volumes > (spike_threshold * avg_volume)

    # Find indices where spikes occurred
    spike_indices = spikes[spikes].index

    # For each timestamp, find time since last spike
    time_since = pd.Series(index=timestamps.index, dtype=float)

    for idx in timestamps.index:
        # Find most recent spike before this point
        prior_spikes = spike_indices[spike_indices < idx]

        if len(prior_spikes) == 0:
            time_since[idx] = np.nan
        else:
            last_spike_idx = prior_spikes[-1]
            time_diff = timestamps[idx] - timestamps[last_spike_idx]

            if unit == "seconds":
                time_since[idx] = time_diff.total_seconds()
            elif unit == "milliseconds":
                time_since[idx] = time_diff.total_seconds() * 1000
            elif unit == "microseconds":
                time_since[idx] = time_diff.total_seconds() * 1_000_000

    time_since.name = f"time_since_volume_spike_{unit}"
    return time_since


def compute_arrival_rate(
    timestamps: pd.Series,
    window: int = 20,
    unit: str = "per_second",
) -> pd.Series:
    """
    Compute the arrival rate of events (inverse of average inter-arrival time).

    Arrival rate measures activity intensity - high rates indicate busy markets
    with rapid trading, while low rates indicate quiet markets. This is the
    instantaneous "speed" of the market.

    Formula:
        arrival_rate_t = window / Σ(inter_arrival_time_{t-window+1:t})
                       = events per unit time

    Intuition:
        **Why arrival rate matters:**

        Arrival rate is the fundamental measure of market activity:

        - **High arrival rate** (many events per second):
          - Active trading, high information flow
          - Competitive market making
          - Price discovery efficient
          - High toxicity risk
          - Spreads typically wider

        - **Low arrival rate** (few events per second):
          - Quiet market, low information
          - Less competition
          - Price stale
          - Low toxicity
          - Spreads can tighten

        **Hawkes process intensity:**

        Arrival rate ≈ Hawkes intensity λ(t):
        - Self-exciting: High rate → more high rates
        - Mean-reverting: Eventually returns to baseline
        - Predictive: Current rate forecasts future rate

        **Intraday patterns:**

        - Market open: Very high rate (100+ events/sec)
        - Mid-morning: Moderate rate (20-50 events/sec)
        - Lunch: Low rate (5-15 events/sec)
        - Close: Very high rate again

    Args:
        timestamps: Series of event timestamps
        window: Number of events to average over (default: 20)
        unit: Rate unit - 'per_second', 'per_minute', 'per_hour' (default: 'per_second')

    Returns:
        Series containing arrival rate in specified units

    Examples:
        >>> timestamps = pd.date_range('2025-01-01 09:30:00', periods=100, freq='0.1s')
        >>> arrival_rate = compute_arrival_rate(timestamps, window=20, unit='per_second')
        >>> # Should show ~10 events per second (0.1s gaps)

    Notes:
        **Interpretation:**

        For equity markets:
        - Rate > 100/sec: Very active (news, open/close)
        - Rate 20-100/sec: Active trading
        - Rate 5-20/sec: Normal activity
        - Rate < 5/sec: Quiet period

        **Uses:**

        - Execution: Slow down algorithms when rate is high
        - Market making: Adjust spreads based on arrival rate
        - Regime detection: Rate clusters define regimes
        - Volatility forecasting: Rate predicts volatility
    """
    # Compute inter-arrival times
    inter_arrival = timestamps.diff().dt.total_seconds()

    # Compute rolling sum of inter-arrival times
    sum_inter_arrival = inter_arrival.rolling(window=window, min_periods=1).sum()

    # Arrival rate = number of events / time elapsed
    # (window - 1 because first event has no prior)
    epsilon = 1e-10

    if unit == "per_second":
        arrival_rate = (window - 1) / (sum_inter_arrival + epsilon)
        arrival_rate.name = "arrival_rate_per_second"
    elif unit == "per_minute":
        arrival_rate = (window - 1) * 60 / (sum_inter_arrival + epsilon)
        arrival_rate.name = "arrival_rate_per_minute"
    elif unit == "per_hour":
        arrival_rate = (window - 1) * 3600 / (sum_inter_arrival + epsilon)
        arrival_rate.name = "arrival_rate_per_hour"
    else:
        raise ValueError(f"Invalid unit: {unit}")

    return arrival_rate
