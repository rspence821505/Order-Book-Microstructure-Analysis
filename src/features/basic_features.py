"""
Basic microstructure feature engineering.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Optional


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
        if any(p in col for p in ["mid_price", "weighted_mid"])
    ]
    spread_features = [col for col in engineered if "spread" in col]
    return_features = [col for col in engineered if "return" in col]
    imbalance_features = [col for col in engineered if "imbalance" in col]
    depth_features = [col for col in engineered if "depth" in col]
    queue_features = [col for col in engineered if "queue" in col]
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
