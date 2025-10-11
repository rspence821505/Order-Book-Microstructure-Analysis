# Limit Order Book Feature Engineering Reference

Complete guide to market microstructure features for HFT machine learning models.

---

## 📊 Basic Features

### Spread

**What It Is:**
The difference between the best ask price and the best bid price, representing the immediate cost of a round-trip trade.

**Equation:**

```
spread = ask_px_1 - bid_px_1
```

**Motivation:**
Measures the cost of immediate execution and market impact. The spread compensates market makers for providing liquidity and bearing inventory risk.

**Market Interpretation:**
Wider spreads indicate lower liquidity, higher volatility, or increased information asymmetry. Narrow spreads suggest competitive, liquid markets with many active market makers. During calm periods, spreads tighten; during volatile or uncertain periods, spreads widen as market makers protect themselves.

**Related Concepts:**
Transaction costs, bid-ask bounce, market making profitability, adverse selection costs, inventory risk.

**Usage in Your Pipeline:**

```python
df['spread'] = df['ask_px_1'] - df['bid_px_1']
```

**Why This Is Powerful:**
Spread is the most fundamental liquidity measure and highly predictive of short-term volatility. It's a key input for execution cost estimation and regime classification. Changes in spread often precede price movements.

**Academic References:**

- Glosten & Milgrom (1985) - "Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders"
- Roll (1984) - "A Simple Implicit Measure of the Effective Bid-Ask Spread"

**Recommendation for Phase 2:**
Essential baseline feature. Compute alongside relative_spread for cross-time comparisons. Use in Week 8 HMM as a primary regime indicator.

---

### Mid Price

**What It Is:**
The midpoint between the best bid and best ask prices, serving as a reference "fair value" for the asset.

**Equation:**

```
mid_price = (ask_px_1 + bid_px_1) / 2
```

**Motivation:**
Provides a central tendency price that averages out the bid-ask spread. Used as the baseline for computing returns, price movements, and other derived features.

**Market Interpretation:**
Represents the theoretical fair value in a frictionless market. Mid-price movements indicate genuine price discovery rather than bid-ask bounce noise. Most ML models predict mid-price direction rather than actual trade prices.

**Related Concepts:**
Price discovery, efficient market hypothesis, microprice (volume-weighted alternative), fair value estimation.

**Usage in Your Pipeline:**

```python
df['mid_price'] = (df['ask_px_1'] + df['bid_px_1']) / 2
```

**Why This Is Powerful:**
Foundation for all return calculations and price movement features. Essential for normalizing other features and creating prediction targets in classification models.

**Academic References:**

- Hasbrouck (1991) - "Measuring the Information Content of Stock Trades"
- Cont, Stoikov & Talreja (2010) - "A Stochastic Model for Order Book Dynamics"

**Recommendation for Phase 2:**
Compute immediately after loading data. Use as the basis for all return features and prediction targets (e.g., predict mid_price direction at t+5 seconds).

---

### Relative Spread

**What It Is:**
The bid-ask spread expressed as a percentage of the mid price, enabling cross-asset and cross-time comparisons.

**Equation:**

```
relative_spread = spread / mid_price
```

**Motivation:**
Normalizes spread by price level. A $1 spread on a $100 asset is very different from a $1 spread on a $50,000 asset. Relative spread makes comparisons meaningful.

**Market Interpretation:**
Typical values for liquid crypto: 0.01%-0.05% (1-5 basis points). Values >0.1% signal illiquidity or stress. Rising relative spread indicates deteriorating market quality. Useful for comparing liquidity across different price regimes.

**Related Concepts:**
Liquidity measurement, market quality metrics, basis points, transaction cost analysis.

**Usage in Your Pipeline:**

```python
df['relative_spread'] = df['spread'] / df['mid_price']
```

**Why This Is Powerful:**
Makes spread comparable across time as price levels change. Essential for detecting abnormal liquidity conditions and regime transitions. More stationary than absolute spread.

**Academic References:**

- Chordia, Roll & Subrahmanyam (2000) - "Commonality in Liquidity"
- Hasbrouck (2009) - "Trading Costs and Returns for U.S. Equities"

**Recommendation for Phase 2:**
Include alongside absolute spread. Use in PCA (Week 7) and as a key feature for HMM regime detection (Week 8) to identify liquidity-driven states.

---

### Weighted Mid Price

**What It Is:**
A volume-weighted midpoint that adjusts for liquidity imbalance at the top of the book, providing a more accurate fair value estimate than simple mid price.

**Equation:**

```
weighted_mid = (ask_px_1 × bid_qty_1 + bid_px_1 × ask_qty_1) / (bid_qty_1 + ask_qty_1)
```

**Motivation:**
When one side of the book has significantly more volume, the simple mid-price doesn't reflect the true balance of supply and demand. Weighted mid shifts toward the side with deeper liquidity.

**Market Interpretation:**
If bid_qty >> ask_qty, weighted_mid shifts below simple mid (downward pressure expected). If ask_qty >> bid_qty, weighted_mid shifts above simple mid (upward pressure expected). The difference between weighted_mid and mid_price is itself a predictive signal.

**Related Concepts:**
Microprice, volume-weighted prices, informed order flow, price pressure effects, queue position.

**Usage in Your Pipeline:**

```python
df['weighted_mid'] = compute_weighted_mid(df)
```

**Why This Is Powerful:**
Superior to simple mid price for short-term price prediction. Research shows weighted mid has lower variance and better forecasting properties. The spread between weighted_mid and mid_price captures immediate directional pressure.

**Academic References:**

- Stoikov & Waeber (2016) - "Reducing Transaction Costs with Low-Latency Trading Algorithms"
- Cartea, Jaimungal & Penalva (2015) - "Algorithmic and High-Frequency Trading"

**Recommendation for Phase 2:**
Essential for Phase 3 mid-price prediction models. Create additional feature: `df['weighted_mid_spread'] = df['weighted_mid'] - df['mid_price']` as a directional indicator.

---

## 📈 Volatility & Price Movement

### Rolling Volatility

**What It Is:**
The standard deviation of returns over a rolling time window, capturing recent price variability and risk.

**Equation:**

```
volatility_Ns = std(return_1s, window=N)
```

where return_1s are 1-second log returns.

**Motivation:**
Markets transition between calm and volatile regimes. Volatility clustering means high volatility persists. Capturing recent volatility helps identify regime changes and adjust trading strategies.

**Market Interpretation:**
Low volatility (< 10 bps/sec) indicates stable, predictable markets suitable for aggressive market making. High volatility (> 50 bps/sec) signals information arrival, uncertainty, or potential regime shift. Market makers widen spreads and reduce depth during high volatility.

**Related Concepts:**
GARCH models, volatility clustering, regime-switching models, VIX, realized volatility, market microstructure noise.

**Usage in Your Pipeline:**

```python
df['volatility_10s'] = compute_rolling_volatility(df, window=10)
df['volatility_60s'] = compute_rolling_volatility(df, window=60)
```

**Why This Is Powerful:**
Multiple volatility windows capture different timescales. Essential for HMM regime detection (Week 8) as primary state variable. Strong predictor of future volatility and spread widening. Tree models use volatility to segment data.

**Academic References:**

- Engle (1982) - "Autoregressive Conditional Heteroscedasticity"
- Andersen et al. (2003) - "Modeling and Forecasting Realized Volatility"

**Recommendation for Phase 2:**
Compute at 10s, 30s, and 60s windows. Use as primary feature for HMM regime classification. Volatility regime is the most important market state to identify.

---

### Returns (Multiple Lags)

**What It Is:**
Logarithmic price changes over various time horizons, capturing momentum and mean-reversion patterns at different scales.

**Equation:**

```
return_Ns = ln(mid_price_t / mid_price_{t-N})
```

**Motivation:**
Returns are more stationary than prices and capture relative changes. Multiple lags reveal momentum (trending) vs mean-reversion (bouncing) behavior. Essential features for price direction prediction.

**Market Interpretation:**
Short lags (1-5s) capture microstructure noise and mean-reversion. Medium lags (10-30s) capture momentum. Long lags (60s+) capture trend. Positive autocorrelation suggests momentum; negative suggests mean-reversion. High-frequency returns exhibit mean-reversion due to bid-ask bounce.

**Related Concepts:**
Momentum strategies, mean reversion, autocorrelation, predictability, market efficiency, bid-ask bounce effects.

**Usage in Your Pipeline:**

```python
df['return_1s'] = compute_returns(df, lag=1, return_type='log')
df['return_5s'] = compute_returns(df, lag=5, return_type='log')
df['return_10s'] = compute_returns(df, lag=10, return_type='log')
df['return_30s'] = compute_returns(df, lag=30, return_type='log')
df['return_60s'] = compute_returns(df, lag=60, return_type='log')
```

**Why This Is Powerful:**
Returns are among the most predictive features for tree models and neural networks. Multiple lags let models learn the optimal lookback period. Can create forward returns as prediction targets: `return_forward_5s` for supervised learning.

**Academic References:**

- Lo & MacKinlay (1988) - "Stock Market Prices Do Not Follow Random Walks"
- Jegadeesh & Titman (1993) - "Returns to Buying Winners and Selling Losers"

**Recommendation for Phase 2:**
Include 5 different lag values (1s, 5s, 10s, 30s, 60s). Use for both features (past returns) and targets (forward returns). Critical for Phase 3 boosted tree models.

---

## 📐 Order Book Shape & Slope

### Book Slope / Volume Decay Rate

**What It Is:**
A measure of how quickly liquidity deteriorates (volume decays) as you move away from the best bid/ask. Quantifies the steepness of the order book depth profile.

**Equation:**

**Linear slope (price-volume):**

```
slope_bid = (bid_px_1 - bid_px_N) / sum(bid_qty_1, ..., bid_qty_N)
```

**Exponential decay rate:**

```
qty_i ≈ qty_1 × exp(-α × i)
slope = -α  (fitted decay parameter)
```

**Log-volume gradient:**

```
gradient = Δlog(qty) / Δlevel = mean(log(qty_{i} / qty_{i+1}))
```

**Motivation:**
Book slope reveals market depth quality and price impact. Steep slopes indicate thin liquidity where large orders have high execution costs. Flat slopes indicate deep, resilient liquidity. Market makers adjust book slope based on their confidence and market conditions.

**Market Interpretation:**
**Steep slope (high α > 0.5):** Liquidity concentrated at top, high price impact, fragile market. Seen during volatile periods or defensive market making. Signals caution and reduced depth provision.

**Flat slope (low α < 0.2):** Liquidity spread evenly, low price impact, resilient market. Seen during calm periods with competitive market making. Signals confidence and aggressive depth provision.

**Asymmetric slopes (bid_slope ≠ ask_slope):** Steeper ask slope suggests market makers expect upward move (pulling ask liquidity). Steeper bid slope suggests expected downward move (pulling bid liquidity). Directional predictor.

**Related Concepts:**
Almgren-Chriss execution model, temporary price impact, Kyle's lambda, liquidity supply curve, market resilience, market maker sentiment.

**Usage in Your Pipeline:**

```python
# Linear slope
df['slope_bid'] = compute_book_slope_linear(df, n_levels=10, side='bid')
df['slope_ask'] = compute_book_slope_linear(df, n_levels=10, side='ask')

# Exponential decay rate (slower but more accurate)
df['decay_rate_bid'] = compute_volume_decay_rate(df, n_levels=10, side='bid')
df['decay_rate_ask'] = compute_volume_decay_rate(df, n_levels=10, side='ask')

# Log-gradient (fastest alternative)
df['log_gradient_bid'] = compute_log_volume_gradient(df, n_levels=10, side='bid')
df['log_gradient_ask'] = compute_log_volume_gradient(df, n_levels=10, side='ask')

# Directional signal
df['slope_asymmetry'] = compute_slope_asymmetry(df, n_levels=10)
df['slope_avg'] = (df['slope_bid'] + df['slope_ask']) / 2
```

**Why This Is Powerful:**
Book slope captures **different information** than depth imbalance or queue features. Depth imbalance measures _how much_ liquidity exists; queue depth measures _where_ it is; book slope measures _how it deteriorates_. This orthogonal information is valuable for PCA and regime detection.

**For PCA (Week 7):** Slope contributes unique variance about market depth quality.

**For HMM (Week 8):** Different regimes have distinct slope signatures. Calm = flat slopes (α ≈ 0.1-0.2), Volatile = steep slopes (α ≈ 0.5-1.0).

**For Prediction (Week 13):** Slope asymmetry is highly predictive of short-term price direction. Tree models learn: "if ask_slope > bid_slope + threshold, predict upward move."

**Academic References:**

- Cont, Kukanov & Stoikov (2014) - "The Price Impact of Order Book Events"
- Bouchaud et al. (2009) - "How Markets Slowly Digest Changes in Supply and Demand"
- Tóth et al. (2011) - "Anomalous Price Impact and the Critical Nature of Liquidity in Financial Markets"
- Donier, Bonart & Bouchaud (2015) - "A Fully Consistent, Minimal Model for Non-Linear Market Impact"

**Recommendation for Phase 2:**
Start with linear slope (fastest). Add log-gradient if you need percentage-based interpretation. Include slope_asymmetry as a directional feature. Use slope_avg to identify market stress in HMM. This will be interview-discussion-worthy: "I measured how liquidity degrades away from mid-price to capture market maker sentiment."

---

### Order Book Thickness

**What It Is:**
The average quantity available per level in the order book, indicating overall book density and liquidity provision intensity.

**Equation:**

```
thickness_bid = mean(bid_qty_1, ..., bid_qty_N)
thickness_ask = mean(ask_qty_1, ..., ask_qty_N)
```

**Motivation:**
Thickness reveals whether market makers are providing dense, continuous liquidity or sparse, patchy liquidity. Complements slope by measuring average depth rather than decay rate.

**Market Interpretation:**
Thick books (high average qty) absorb orders without large price moves. Thin books are fragile to shocks. Increasing thickness suggests growing market maker confidence. Decreasing thickness signals caution or inventory constraints. Asymmetric thickness (bid_thick ≠ ask_thick) provides directional signals.

**Related Concepts:**
Liquidity provision strategies, market maker inventory management, resilience, market depth, order splitting behavior.

**Usage in Your Pipeline:**

```python
df['bid_thickness'] = compute_book_thickness(df, side='bid', n_levels=10)
df['ask_thickness'] = compute_book_thickness(df, side='ask', n_levels=10)
df['thickness_asymmetry'] = df['bid_thickness'] - df['ask_thickness']
df['thickness_avg'] = (df['bid_thickness'] + df['ask_thickness']) / 2
```

**Why This Is Powerful:**
Thickness is complementary to slope. You can have thick books with steep slopes (concentrated liquidity) or thin books with flat slopes (sparse but even distribution). Together they fully characterize book shape. Useful for identifying market maker strategy changes.

**Academic References:**

- Foucault, Kadan & Kandel (2005) - "Limit Order Book as a Market for Liquidity"
- Rosu (2009) - "A Dynamic Model of the Limit Order Book"

**Recommendation for Phase 2:**
Include alongside slope features. Use thickness_avg to filter regimes: thick + flat = healthy market, thin + steep = stressed market. Good feature for Week 8 HMM.

---

## 💧 Liquidity Measures

### VWAP (Volume-Weighted Average Price)

**What It Is:**
The average price weighted by volume across multiple levels, representing realistic execution cost for larger orders that walk the book.

**Equation:**

```
vwap_bid = sum(bid_px_i × bid_qty_i) / sum(bid_qty_i)  for i=1..N
vwap_ask = sum(ask_px_i × ask_qty_i) / sum(ask_qty_i)  for i=1..N
```

**Motivation:**
Simple mid-price or best bid/ask don't reflect execution cost for large orders. VWAP accounts for the need to trade across multiple levels, providing realistic cost estimates.

**Market Interpretation:**
VWAP deviates from mid-price when the book is imbalanced or thin. Large VWAP_ask - mid_price spread indicates high cost to buy. Large mid_price - VWAP_bid spread indicates high cost to sell. VWAP spread > simple spread indicates depth is shallow beyond top level.

**Related Concepts:**
Execution algorithms (VWAP benchmarking), slippage, market impact, trade cost analysis, price impact models.

**Usage in Your Pipeline:**

```python
df['vwap_bid'] = compute_vwap(df, side='bid', n_levels=5)
df['vwap_ask'] = compute_vwap(df, side='ask', n_levels=5)
df['vwap_mid'] = (df['vwap_bid'] + df['vwap_ask']) / 2
df['vwap_spread'] = df['vwap_ask'] - df['vwap_bid']
```

**Why This Is Powerful:**
More realistic execution cost measure than simple spread. VWAP features are essential for Phase 5 execution algorithms. Can compute VWAP at different depths (top 3, top 5, top 10) to capture execution cost at different sizes. The difference between VWAP and mid is a measure of market depth quality.

**Academic References:**

- Almgren & Chriss (2001) - "Optimal Execution of Portfolio Transactions"
- Berkowitz, Logue & Noser (1988) - "The Total Cost of Transactions on the NYSE"

**Recommendation for Phase 2:**
Compute at top 5 levels. Use VWAP_spread instead of simple spread for more accurate cost measurement. Create feature `vwap_deviation = vwap_mid - mid_price` as depth quality indicator.

---

### Effective Spread

**What It Is:**
The difference between volume-weighted average prices on bid and ask sides, representing actual trading cost for realistic-sized orders rather than just top-of-book spread.

**Equation:**

```
effective_spread = vwap_ask - vwap_bid
```

**Motivation:**
Simple spread only considers the best bid/ask, but real execution requires trading across multiple levels. Effective spread accounts for depth effects and provides accurate cost measurement.

**Market Interpretation:**
Effective spread > simple spread indicates thin book beyond top level (high depth-dependent costs). Effective spread ≈ simple spread indicates uniform, deep book. Rising effective spread signals deteriorating depth. Important for comparing liquidity quality across time periods or assets.

**Related Concepts:**
Market quality measurement, realized transaction costs, depth quality, liquidity fragmentation, execution cost analysis.

**Usage in Your Pipeline:**

```python
df['effective_spread'] = df['vwap_ask'] - df['vwap_bid']
df['spread_ratio'] = df['effective_spread'] / df['spread']  # How much worse than top-of-book?
```

**Why This Is Powerful:**
More accurate measure of true trading costs than simple spread. The ratio `effective_spread / spread` reveals depth quality: values near 1.0 indicate robust depth; values > 2.0 indicate thin book beyond best prices. Essential for real-world execution cost modeling.

**Academic References:**

- Bessembinder & Kaufman (1997) - "A Comparison of Trade Execution Costs for NYSE and NASDAQ-Listed Stocks"
- Hasbrouck (2009) - "Trading Costs and Returns for U.S. Equities"

**Recommendation for Phase 2:**
Use alongside simple spread. The spread_ratio feature (effective/simple) is particularly useful for regime detection—thin books have high ratios. Good feature for Phase 3 tree models.

---

### Microprice

**What It Is:**
A volume-weighted mid-price that uses quantities at the best bid and ask to predict the next mid-price movement. More responsive to imbalances than simple mid-price.

**Equation:**

```
microprice = (ask_px_1 × bid_qty_1 + bid_px_1 × ask_qty_1) / (bid_qty_1 + ask_qty_1)
```

**Motivation:**
When bid_qty >> ask_qty, the next price movement is more likely upward (buying pressure), so microprice shifts above mid. When ask_qty >> bid_qty, selling pressure dominates, and microprice shifts below mid. This creates a more accurate "fair value" estimate.

**Market Interpretation:**
Microprice - mid_price reveals directional pressure. Positive difference = buying pressure (microprice above mid). Negative difference = selling pressure (microprice below mid). Research shows microprice has lower variance than mid and better short-term forecasting properties.

**Related Concepts:**
High-frequency prediction, informed order flow, adverse selection, price pressure, queue position theory, optimal pricing for market makers.

**Usage in Your Pipeline:**

```python
df['microprice'] = compute_microprice(df)
df['microprice_deviation'] = df['microprice'] - df['mid_price']
```

**Why This Is Powerful:**
Microprice is one of the best short-term price predictors in academic literature. The microprice_deviation feature captures immediate directional pressure. Essential for Phase 3 and Phase 4 prediction models. Some studies show microprice alone can achieve 55-60% directional accuracy for next-tick prediction.

**Academic References:**

- Stoikov & Waeber (2016) - "Reducing Transaction Costs with Low-Latency Trading Algorithms"
- Cartea, Jaimungal & Penalva (2015) - "Algorithmic and High-Frequency Trading"
- Cont, Kukanov & Stoikov (2014) - "The Price Impact of Order Book Events"

**Recommendation for Phase 2:**
Essential feature. Compute both microprice and microprice_deviation. Use as baseline predictor in Phase 1 capstone. Will be a strong feature in all subsequent models.

---

## 🌊 Order Book Pressure & Flow

### Order Flow Imbalance (OFI)

**What It Is:**
The net change in bid and ask quantities between consecutive snapshots, capturing buying and selling pressure from order arrivals, cancellations, and executions.

**Equation:**

```
ofi_t = Δbid_qty - Δask_qty
```

where Δ denotes change from previous snapshot, summed across levels.

**Motivation:**
OFI captures order flow—the actual trading activity and liquidity changes. Positive OFI means more bids arrived or asks were consumed (bullish). Negative OFI means more asks arrived or bids were consumed (bearish). This flow information is highly predictive of short-term price movements.

**Market Interpretation:**
OFI > 0: Net buying pressure, expect upward price movement. OFI < 0: Net selling pressure, expect downward movement. Large |OFI| indicates aggressive trading or information arrival. OFI clustering (sustained positive or negative) suggests directional momentum. Mean-reverting OFI suggests noise trading.

**Related Concepts:**
Order flow toxicity, informed trading, price impact, adverse selection, market microstructure theory (Glosten-Milgrom, Kyle models).

**Usage in Your Pipeline:**

```python
df['ofi'] = compute_order_flow_imbalance(df, n_levels=10)
df['ofi_normalized'] = df['ofi'] / df['queue_total_cumulative']  # Scale by book size
```

**Why This Is Powerful:**
OFI is one of the most predictive features in HFT literature. It captures **flow** (changes over time) whereas depth imbalance captures **state** (current snapshot). Together they provide complementary information. OFI is essential for Phase 1 capstone and remains important through all phases.

**Academic References:**

- Cont, Kukanov & Stoikov (2014) - "The Price Impact of Order Book Events"
- Lipton, Pesavento & Sotiropoulos (2013) - "Trade Arrival Dynamics and Quote Imbalance in a Limit Order Book"
- Hautsch & Huang (2012) - "The Market Impact of a Limit Order"

**Recommendation for Phase 2:**
Core feature. Compute at multiple levels (top 3, top 5, all levels) to capture different scales of order flow. Normalize by total book volume for comparability. Use raw OFI and normalized OFI as separate features.

**Interview Talking Points**
"Order flow imbalance measures the relative difference between bid and ask volumes. I computed it as (bid - ask) / (bid + ask), which normalizes to the range [-1, 1]. I also experimented with multi-level imbalance aggregating across the top 5 levels and volume-weighted imbalance with exponential decay to give more weight to levels closer to the mid-price. Imbalance is a strong predictor of short-term price movements because it captures supply/demand dynamics directly."

---

### Depth Imbalance

**What It Is:**
The normalized difference between total bid and ask quantities, measuring current supply/demand imbalance across multiple price levels.

**Equation:**

```
depth_imbalance = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)
```

where totals are summed across N levels.

**Motivation:**
Captures the overall state of the order book. Unlike OFI (which measures changes), depth imbalance measures the current snapshot of supply vs demand. Persistent imbalances often precede price movements.

**Market Interpretation:**
Values near +1: Strong buy-side liquidity, potential resistance to downward moves, or trapped long positions. Values near -1: Strong sell-side liquidity, potential resistance to upward moves, or trapped short positions. Values near 0: Balanced book, fair price discovery. Extreme values often revert as market makers rebalance.

**Related Concepts:**
Queue position, inventory risk, price pressure theory, market maker inventory management, supply-demand equilibrium.

**Usage in Your Pipeline:**

```python
df['depth_imbalance'] = compute_depth_imbalance(df, n_levels=10)
```

**Why This Is Powerful:**
Complements OFI by capturing state vs flow. Research shows depth imbalance at multiple depths (top 3, top 5, all levels) each provide unique predictive power. Essential for HMM regime detection—different regimes show different imbalance patterns. Good feature for tree models.

**Academic References:**

- Cao, Hansch & Wang (2009) - "The Information Content of an Open Limit-Order Book"
- Cartea & Penalva (2012) - "Where is the Value in High Frequency Trading?"

**Recommendation for Phase 2:**
Compute alongside OFI. Include imbalance at multiple depths (see Liquidity Imbalance at Multiple Depths below). Use as key feature for regime detection in Week 8 HMM.

---

### Liquidity Imbalance at Multiple Depths

**What It Is:**
Depth imbalance computed separately at each depth level (cumulative from level 1 to level N), revealing how imbalance evolves as you go deeper into the book.

**Equation:**

```
liquidity_imbalance_depth_N = (sum_bid_1_to_N - sum_ask_1_to_N) / (sum_bid_1_to_N + sum_ask_1_to_N)
```

**Motivation:**
Different depths reveal different information. Imbalance at level 1 shows immediate pressure. Imbalance at level 5 shows near-term positioning. Imbalance at level 10 shows institutional or informed positioning. Creating a depth profile reveals **where in the book** the pressure lies.

**Market Interpretation:**
**Top-heavy imbalance (high at level 1-3, neutral deeper):** High-frequency traders or aggressive market makers creating short-term pressure. Often mean-reverts quickly.

**Deep imbalance (neutral at top, high at levels 5-10):** Institutional orders or informed traders positioning for larger moves. More persistent signal.

**Uniform imbalance (same across all depths):** Strong directional consensus. Most reliable for prediction.

**Divergent imbalance (opposite signs at different depths):** Conflicting signals, often noisy or transitional state.

**Related Concepts:**
Order book depth profile, informed vs uninformed traders, institutional order placement, iceberg orders, price impact at scale.

**Usage in Your Pipeline:**

```python
# Create imbalance profile at multiple depths
for depth in [1, 2, 3, 5, 10]:
    df[f'liquidity_imbalance_depth_{depth}'] = compute_liquidity_imbalance_at_depth(df, depth=depth)

# Optional: Create depth gradient
df['imbalance_gradient'] = df['liquidity_imbalance_depth_10'] - df['liquidity_imbalance_depth_1']
```

**Why This Is Powerful:**
**For PCA (Week 7):** These depth-wise features reveal **where** the important variance lies. PCA may discover that depths 1-3 vary together (high-frequency regime) or depths 5-10 vary together (institutional regime).

**For HMM (Week 8):** Different market regimes show imbalance at different depths. Calm regime = balanced at all depths. High-frequency regime = imbalance concentrated at top. Institutional regime = deep imbalance with neutral top.

**For Prediction (Week 13):** Tree models can learn: "if top imbalance is positive but deep imbalance is negative, predict mean-reversion." This depth structure adds predictive power beyond simple depth imbalance.

**Academic References:**

- Cao, Hansch & Wang (2009) - "The Information Content of an Open Limit-Order Book"
- Hautsch & Huang (2012) - "The Market Impact of a Limit Order"
- Cartea, Jaimungal & Penalva (2015) - "Algorithmic and High-Frequency Trading" (discusses multi-level effects)

**Recommendation for Phase 2:**
**Required for your capstone.** Compute at depths [1, 2, 3, 5, 10]. These 5 features provide a complete depth profile. In your capstone writeup, visualize how imbalance evolves across depths during different market conditions. This will be an excellent interview discussion topic.

---

### Order Book Pressure (Cumulative Depth-Weighted Imbalance)

**What It Is:**
A sophisticated variant of depth imbalance that weights levels by their proximity to the mid-price, emphasizing actionable liquidity near the top of the book while including deeper levels with reduced influence.

**Equation:**
**Inverse distance weighting:**

```
OBP = sum_{i=1}^{N} (1/i) × (bid_qty_i - ask_qty_i)
```

**Exponential decay weighting:**

```
OBP = sum_{i=1}^{N} exp(-λ × i) × (bid_qty_i - ask_qty_i)
```

where λ controls decay rate (typical values: 0.3-1.0)

**Normalized version (for [-1, 1] range):**

```
OBP_norm = sum w_i(bid_qty_i - ask_qty_i) / sum w_i(bid_qty_i + ask_qty_i)
```

**Motivation:**
Unlike simple depth imbalance (which treats all levels equally), order book pressure recognizes that levels closer to mid-price are more actionable and informationally relevant. A 100 BTC order at level 1 has far more immediate price impact than 100 BTC at level 10. OBP captures this economic reality by weighting imbalances by depth.

**Market Interpretation:**
**Positive OBP:** Weighted buying pressure, particularly at actionable levels near mid-price. Suggests upward price movement more likely. Larger magnitude indicates stronger directional pressure.

**Negative OBP:** Weighted selling pressure concentrated where it matters most. Suggests downward price movement.

**High magnitude OBP:** Strong directional signal in the executable book. More reliable than simple depth imbalance for prediction.

**Low magnitude OBP:** Balanced book with no clear pressure. Neutral or ranging market.

**Key insight:** OBP correctly emphasizes liquidity at the top of the book. A book with 50 BTC at level 1 bid and 10 BTC at level 1 ask will show strong positive OBP even if deeper levels are balanced, because the top-level imbalance is what drives immediate price action.

**Related Concepts:**
Kyle's lambda (price impact parameter), market depth elasticity, informed trading pressure, actionable liquidity, short-term price forecasting, execution cost models.

**Usage in Your Pipeline:**

```python
# Order book pressure with different weighting schemes
df['obp_inverse'] = compute_order_book_pressure(df, n_levels=10, weighting='inverse')
df['obp_exponential'] = compute_order_book_pressure(df, n_levels=10, weighting='exponential')
df['obp_normalized'] = compute_normalized_order_book_pressure(df, n_levels=10, weighting='inverse')

# Optional: Multiple decay rates to capture different horizons
df['obp_fast_decay'] = compute_order_book_pressure(df, n_levels=5, weighting='exponential', lambda_decay=1.0)
df['obp_slow_decay'] = compute_order_book_pressure(df, n_levels=10, weighting='exponential', lambda_decay=0.3)
```

**Why This Is Powerful:**
**Superior predictive power:** Research shows OBP predicts short-term price movements better than unweighted depth imbalance because it emphasizes executable, actionable liquidity rather than treating all depth equally.

**Multi-scale analysis:** Different weighting schemes capture different trading horizons. Fast exponential decay (λ=1.0) focuses on ultra-short-term (next 1-5 seconds), considering only top 3-5 levels. Slow decay (λ=0.3) or inverse weighting captures longer-term positioning by including deeper book.

**Complements other features:** You now have three complementary views:

- **OFI:** Changes over time (flow)
- **Depth imbalance:** Total inventory (state)
- **OBP:** Actionable directional pressure (weighted state)

**For PCA (Week 7):** OBP contributes unique variance by weighting depth by relevance. May load differently than simple depth imbalance.

**For HMM (Week 8):** Different regimes show different OBP patterns. Calm markets have low OBP variance; volatile markets show rapid OBP swings.

**For Prediction (Week 13):** Tree models can learn regime-dependent rules: "In high-volatility regime, use fast-decay OBP; in calm regime, use slow-decay OBP."

**Academic References:**

- Cont, Stoikov & Talreja (2010) - "A Stochastic Model for Order Book Dynamics"
- Cartea, Jaimungal & Penalva (2015) - "Algorithmic and High-Frequency Trading"
- Sirignano (2016) - "Deep Learning for Limit Order Books"
- Huang & Polak (2011) - "LOBSTER: Limit Order Book System - The Efficient Reconstructor"

**Recommendation for Phase 2:**
Start with inverse weighting (simplest, interpretable). Add normalized version for [-1, 1] scaling. If you have time in Week 9, experiment with exponential decay at different λ values and compare predictive power. This feature will be excellent for interviews: "I weighted order book imbalance by depth to emphasize actionable liquidity, improving prediction accuracy over naive depth measures."

---

### Queue Imbalance (Multiple Levels)

**What It Is:**
Depth imbalance computed at specific aggregation levels (top 1, top 5, cumulative), revealing how supply/demand imbalance manifests at different depths of the order book.

**Equation:**

```
queue_imbalance_topN = (queue_bid_topN - queue_ask_topN) / (queue_bid_topN + queue_ask_topN)
```

where queue_bid_topN is sum of bid quantities from level 1 to N.

**Motivation:**
Similar to liquidity imbalance at multiple depths, but focuses on standard aggregation levels (1, 5, all) that correspond to different trading contexts. Top 1 = immediate executable liquidity, Top 5 = near-term resilience, Cumulative = total visible depth.

**Market Interpretation:**
**queue_imbalance_top1:** Immediate pressure at best bid/ask. Most volatile, responds to every order arrival/cancellation. High-frequency signal.

**queue_imbalance_top5:** Near-term pressure within ~$1-2 of mid. More stable than top1. Captures medium-frequency positioning.

**queue_imbalance_cumulative:** Overall book state across all visible levels. Most stable, captures long-term positioning and institutional orders.

**Divergence patterns:**

- **Top1 positive, cumulative negative:** Short-term buying pressure against deeper selling—expect mean reversion
- **All levels positive:** Uniform buying pressure—expect continuation
- **Top1 neutral, cumulative positive:** Deep hidden demand—potential breakout setup

**Related Concepts:**
Queue position theory, inventory risk, market making strategies, depth profile analysis, regime-dependent trading.

**Usage in Your Pipeline:**

```python
# Queue imbalance at standard aggregation levels
df['queue_imbalance_top1'] = compute_queue_imbalance(df, level='top1')
df['queue_imbalance_top5'] = compute_queue_imbalance(df, level='top5')
df['queue_imbalance_cumulative'] = compute_queue_imbalance(df, level='cumulative')

# Divergence features
df['queue_imbalance_divergence'] = df['queue_imbalance_top1'] - df['queue_imbalance_cumulative']
```

**Why This Is Powerful:**
These three levels provide a complete summary of the depth profile without needing individual level-by-level features. More parsimonious than computing imbalance at every depth. Each captures a different timescale and trader type.

**For PCA (Week 7):** These three features may load on different principal components, revealing that different market participants (HFT, institutional) drive variance at different depths.

**For HMM (Week 8):** Regime signatures often appear in the relationship between these three. Calm regime = all three move together. Volatile regime = top1 diverges from deeper levels.

**For Tree Models (Week 13):** Can learn interaction rules: "if top1 > 0.5 and cumulative < 0, predict reversal."

**Academic References:**

- Cao, Hansch & Wang (2009) - "The Information Content of an Open Limit-Order Book"
- Hautsch & Huang (2012) - "The Market Impact of a Limit Order"

**Recommendation for Phase 2:**
Compute all three. These are essential features that provide a complete summary of book imbalance at different scales. Include the divergence feature to capture top vs deep conflicts.

---

### Cumulative Volume Imbalance

**What It Is:**
The absolute (not normalized) difference in total volume between bid and ask sides across all levels, measuring raw supply/demand asymmetry in quantity terms.

**Equation:**

```
cvi = sum_{i=1}^{N} (bid_qty_i - ask_qty_i)
```

**Motivation:**
Unlike normalized depth imbalance (which scales to [-1,1]), cumulative volume imbalance preserves the magnitude of the imbalance. A book with 100 BTC bid vs 50 BTC ask is different from 10 BTC bid vs 5 BTC ask, even though both have the same normalized imbalance ratio.

**Market Interpretation:**
**Large positive CVI:** Significant excess bid-side liquidity. Could indicate: (1) trapped long positions unable to exit, (2) institutional accumulation, (3) defensive market making after downward move.

**Large negative CVI:** Excess ask-side liquidity. Could indicate: (1) trapped shorts, (2) institutional distribution, (3) defensive market making after upward move.

**Near-zero CVI:** Balanced book regardless of total size. Healthy two-sided market.

**Scaling with book size:** CVI should be considered relative to total book depth. 50 BTC imbalance is huge for a thin book but small for a deep book.

**Related Concepts:**
Market making inventory accumulation, liquidity traps, position unwinding, institutional order placement, inventory risk management.

**Usage in Your Pipeline:**

```python
df['cumulative_volume_imbalance'] = compute_cumulative_volume_imbalance(df, n_levels=10)

# Normalized version for cross-time comparison
df['cvi_normalized'] = df['cumulative_volume_imbalance'] / df['queue_total_cumulative']
```

**Why This Is Powerful:**
CVI captures **magnitude** while depth_imbalance captures **ratio**. Both are informative. Large CVI even with neutral ratio suggests massive book size (high confidence). Small CVI with extreme ratio suggests thin, imbalanced book (fragile state). Together they fully characterize the imbalance situation.

**Academic References:**

- Cont, Stoikov & Talreja (2010) - "A Stochastic Model for Order Book Dynamics"
- Rosu (2009) - "A Dynamic Model of the Limit Order Book"

**Recommendation for Phase 2:**
Compute both raw CVI and normalized version. Use raw CVI to detect absolute imbalance magnitude; use normalized for regime comparison. Particularly useful for identifying trapped positions or large institutional orders.

---

### Order Flow Toxicity

**What It Is:**
A measure of adverse selection risk, quantifying how much informed traders (who move prices with small volume) are active versus uninformed traders (large volume, small price impact).

**Equation:**

```
toxicity ≈ |Δprice| / volume_executed
```

or using VPIN (Volume-Synchronized Probability of Informed Trading):

```
VPIN = |buy_volume - sell_volume| / total_volume
```

over recent time window.

**Motivation:**
When informed traders are active, small orders have large price impacts (high toxicity). When uninformed traders dominate, large orders have small price impacts (low toxicity). Market makers widen spreads and reduce depth when toxicity is high to protect against adverse selection.

**Market Interpretation:**
**High toxicity (>threshold):** Informed trading detected. Price movements are driven by small volumes. High information asymmetry. Market makers should be defensive (wide spreads, shallow depth).

**Low toxicity:** Uninformed order flow. Price movements require large volumes. Low information asymmetry. Market makers can be aggressive.

**Rising toxicity:** Information event approaching or occurring. Signals regime transition to volatile/informed state.

**Related Concepts:**
Adverse selection, informed vs uninformed trading, Glosten-Milgrom model, Kyle's lambda, VPIN, flash crashes, market maker risk management.

**Usage in Your Pipeline:**

```python
df['order_flow_toxicity'] = compute_flow_toxicity(df, window=10)

# VPIN alternative
df['vpin'] = compute_vpin(df, window=50)
```

**Why This Is Powerful:**
Toxicity is a sophisticated feature that captures information asymmetry. High toxicity periods are dangerous for market makers but profitable for informed traders. Including toxicity helps models identify when prediction is easier (high toxicity = clear direction) vs harder (low toxicity = noise).

**For HMM (Week 8):** Toxicity is an excellent regime indicator. Can define regimes as: calm/low-toxicity, volatile/high-toxicity, informed/extreme-toxicity.

**For Risk Management (Phase 5):** Essential for optimal execution and market making. High toxicity = reduce position size, widen spreads.

**Academic References:**

- Easley, López de Prado & O'Hara (2012) - "Flow Toxicity and Liquidity in a High Frequency World"
- Easley, Kiefer & O'Hara (1997) - "One Day in the Life of a Very Common Stock"
- Andersen & Bondarenko (2014) - "VPIN and the Flash Crash"

**Recommendation for Phase 2:**
Advanced feature—add in Week 9 if time permits. Requires computing volume and price changes together. Start with simple version (|Δprice|/volume). VPIN requires trade direction classification, which is more complex. This will be impressive in interviews as it shows understanding of informed trading theory.

---

## 📏 Queue Depth Features

### Queue Depth (Top 1, Top 5, Cumulative)

**What It Is:**
The total quantity available at specific depth levels of the order book, measuring immediate, near-term, and total liquidity respectively.

**Equation:**

```
queue_bid_top1 = bid_qty_1
queue_bid_top5 = sum(bid_qty_1, ..., bid_qty_5)
queue_bid_cumulative = sum(bid_qty_1, ..., bid_qty_N)

queue_total_top1 = queue_bid_top1 + queue_ask_top1
queue_total_top5 = queue_bid_top5 + queue_ask_top5
queue_total_cumulative = queue_bid_cumulative + queue_ask_cumulative
```

**Motivation:**
Measures available liquidity at different scales. Top 1 = immediate execution capacity (can I fill my order right now?). Top 5 = near-term resilience (will the book absorb moderate flow?). Cumulative = total visible depth (what's the maximum visible liquidity?).

**Market Interpretation:**
**High queue_total_top1:** Dense liquidity at best prices. Low immediate execution cost. Competitive market making.

**Low queue_total_top1:** Thin top of book. High execution cost. Cautious or defensive market making. Potential for large bid-ask bounce.

**queue_total_top5 >> queue_total_top1:** Liquidity concentrated just beyond best prices. "Hidden depth" or stepped pricing strategy.

**queue_total_cumulative >> queue_total_top5:** Deep book with liquidity spread across many levels. Resilient to large orders.

**Declining queue depths over time:** Liquidity withdrawal, regime transition to volatile state, or pre-announcement positioning.

**Related Concepts:**
Execution cost estimation, market depth, order placement strategy, market maker inventory, liquidity provision incentives.

**Usage in Your Pipeline:**

```python
df = compute_queue_depth(df, n_levels=10)
# This adds 9 columns: queue_bid_top1, queue_ask_top1, queue_total_top1,
#                      queue_bid_top5, queue_ask_top5, queue_total_top5,
#                      queue_bid_cumulative, queue_ask_cumulative, queue_total_cumulative
```

**Why This Is Powerful:**
Queue depth is fundamental for execution algorithms and market making. These three aggregation levels provide complete liquidity summary without redundancy. Essential inputs for:

- Price impact estimation
- Optimal execution (Phase 5 RL agents)
- Regime classification (thin vs thick book regimes)
- Feature normalization (normalize imbalances by queue depth)

**For PCA (Week 7):** Queue depths at different levels may have different variance patterns, revealing liquidity provision strategies.

**For HMM (Week 8):** Calm regime = high stable queue depths. Volatile regime = low volatile queue depths. Transition points show rapid depth changes.

**Academic References:**

- Foucault, Kadan & Kandel (2005) - "Limit Order Book as a Market for Liquidity"
- Parlour (1998) - "Price Dynamics in Limit Order Markets"

**Recommendation for Phase 2:**
Essential features. Compute all three aggregation levels. Use queue_total_cumulative to normalize other features (e.g., OFI / queue_total gives scale-independent OFI).

---

### Depth Concentration

**What It Is:**
The fraction of total liquidity concentrated at the very top of the book, measuring how front-loaded the order book is.

**Equation:**

```
depth_concentration = queue_total_top1 / queue_total_cumulative
```

**Motivation:**
Reveals whether liquidity is concentrated at best prices (aggressive, competitive market making) or spread across many levels (defensive, cautious provision). Different market maker strategies and market conditions produce different concentration patterns.

**Market Interpretation:**
**High concentration (>0.4):** Most liquidity at top of book. Could indicate: (1) aggressive market making competing for rebates, (2) fleeting liquidity that cancels quickly, (3) high-frequency traders dominating, (4) narrow spread environment.

**Low concentration (<0.15):** Liquidity spread evenly or concentrated deeper. Could indicate: (1) defensive market making expecting volatility, (2) institutional orders placed away from market, (3) wide spread environment, (4) low confidence in current price.

**Rising concentration:** Market makers becoming more aggressive or confident. Potentially transitioning to calmer regime.

**Falling concentration:** Liquidity withdrawal from top, defensive positioning. Warning sign of potential volatility or stress.

**Related Concepts:**
Market maker strategy, fleeting liquidity, quote stuffing, maker-taker pricing models, high-frequency trading patterns, liquidity concentration risk.

**Usage in Your Pipeline:**

```python
df['depth_concentration'] = compute_depth_concentration(df)

# Separate bid and ask concentration to detect asymmetry
df['bid_concentration'] = df['queue_bid_top1'] / df['queue_bid_cumulative']
df['ask_concentration'] = df['queue_ask_top1'] / df['queue_ask_cumulative']
df['concentration_asymmetry'] = df['bid_concentration'] - df['ask_concentration']
```

**Why This Is Powerful:**
Concentration captures market maker behavior and confidence in ways that raw queue depth cannot. Two books can have same total depth but very different concentration—revealing different market states.

**High depth + high concentration:** Aggressive competitive environment
**High depth + low concentration:** Institutional depth positioned away from market
**Low depth + high concentration:** Fleeting HFT liquidity
**Low depth + low concentration:** Defensive, scared market

**For HMM (Week 8):** Concentration is an excellent regime indicator. Can define regimes based on concentration levels.

**For Risk Assessment:** High concentration = liquidity could vanish quickly (fleeting). Low concentration = stable but potentially expensive execution.

**Academic References:**

- Hendershott, Jones & Menkveld (2011) - "Does Algorithmic Trading Improve Liquidity?"
- Hasbrouck & Saar (2013) - "Low-Latency Trading"

**Recommendation for Phase 2:**
Compute for overall book and separately for bid/ask sides. The asymmetry feature (bid_concentration - ask_concentration) provides directional signal: if bid side more concentrated, market makers are more aggressive on bids (expect upward support).

---

## 💥 Price Impact Features

### Price Impact (Multiple Volumes)

**What It Is:**
The estimated cost of executing an order of given size, measured as the deviation from mid-price to the volume-weighted average execution price when walking the book.

**Equation:**

```
execution_price_V = VWAP to execute volume V
price_impact_V = (execution_price_V - mid_price) / mid_price
```

For buy order: walk up the ask side until cumulative volume ≥ V
For sell order: walk down the bid side until cumulative volume ≥ V

**Motivation:**
Captures market depth quality and resilience. Shows how much price degrades as you execute larger orders. Directly related to execution cost and optimal trading strategies. Different volumes reveal depth at different scales.

**Market Interpretation:**
**Low price_impact:** Deep, liquid book. Large orders can be executed with minimal cost. Resilient market.

**High price_impact:** Thin book. Even moderate orders significantly move the execution price. Fragile, illiquid market.

**Increasing impact with volume:** Normal concave relationship. Impact grows but at decreasing rate (square-root law).

**Super-linear impact:** Impact grows faster than square-root. Indicates very thin book beyond a certain depth threshold.

**Asymmetric impact (buy_impact ≠ sell_impact):** Directional signal. Higher buy impact = thin ask side = resistance to upward moves. Higher sell impact = thin bid side = resistance to downward moves.

**Related Concepts:**
Almgren-Chriss execution model, temporary vs permanent impact, square-root law, market depth elasticity, optimal execution (TWAP, VWAP, POV strategies).

**Usage in Your Pipeline:**

```python
# Price impact at different volumes
df['price_impact_1btc'] = compute_price_impact(df, volume=1.0, side='buy')
df['price_impact_5btc'] = compute_price_impact(df, volume=5.0, side='buy')
df['price_impact_10btc'] = compute_price_impact(df, volume=10.0, side='buy')

# Sell-side impacts
df['price_impact_1btc_sell'] = compute_price_impact(df, volume=1.0, side='sell')

# Impact ratio (measure of non-linearity)
df['impact_ratio_5_1'] = df['price_impact_5btc'] / df['price_impact_1btc']  # Should be ~√5 ≈ 2.24
```

**Why This Is Powerful:**
Price impact directly measures the quality

**Market Interpretation:**
**Market open (9:30-10:00 AM EST):** High volatility, wide spreads, information from overnight news. For crypto (24/7): US market open still creates activity spike.

**Mid-day (11:00 AM - 2:00 PM):** Lower volatility, tighter spreads, more predictable. Lowest activity period.

**Market close (3:30-4:00 PM EST):** High volatility, portfolio rebalancing, index replication trades. For crypto: US close creates activity change.

**Overnight/Asian hours (crypto):** Different liquidity providers, often thinner books, different volatility patterns.

**Weekend (crypto):** Significantly different patterns—lower institutional participation, thinner liquidity, higher retail percentage.

**Related Concepts:**
U-shaped volatility pattern, market microstructure noise variation by time, informed trading timing, index rebalancing, settlement effects, global market interconnections.

**Usage in Your Pipeline:**

```python
# Simple integer encoding
df['hour'] = df.index.hour
df['minute'] = df.index.minute
df['day_of_week'] = df.index.dayofweek  # 0=Monday, 6=Sunday

# Cyclic encoding (better for ML models)
df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
df['minute_sin'] = np.sin(2 * np.pi * df.index.minute / 60)
df['minute_cos'] = np.cos(2 * np.pi * df.index.minute / 60)

# Categorical: trading session
df['session'] = df.index.hour.map(lambda h: 'asian' if h < 8 else 'european' if h < 13 else 'us' if h < 21 else 'overnight')
```

**Why This Is Powerful:**
**Cyclic encoding is critical:** Hour 23 and hour 0 are adjacent but numerically far apart. Sin/cos encoding preserves cyclical nature, improving model performance especially for neural networks.

**Interaction with other features:** Tree models can learn: "if hour=10 and volatility < X, use strategy A; if hour=15, use strategy B." Different times have different optimal strategies.

**For HMM (Week 8):** Can condition regime transitions on time of day. Some regimes only occur during specific trading hours.

**For Neural Networks (Week 14-18):** Time embeddings are essential. Use sin/cos encoding for continuous time representation.

**Academic References:**

- Admati & Pfleiderer (1988) - "A Theory of Intraday Patterns: Volume and Price Variability"
- Andersen & Bollerslev (1997) - "Intraday Periodicity and Volatility Persistence in Financial Markets"
- Engle & Russell (1998) - "Autoregressive Conditional Duration: A New Model for Irregularly Spaced Transaction Data"

**Recommendation for Phase 2:**
Add in Week 6-7. Use cyclic encoding (sin/cos) for continuous models. Keep integer encoding for tree models (they can split on hour directly). Essential for any model deployed in real markets where time-of-day effects are strong.

---

## 📐 Higher-Order Statistics

### Book Skewness

**What It Is:**
The third moment of the quantity distribution across order book levels, measuring asymmetry in how volume is distributed.

**Equation:**

```
skewness_bid = (1/N) × sum((qty_i - mean_qty)³) / std_qty³
```

for levels i = 1 to N.

**Motivation:**
Skewness reveals the shape of liquidity distribution. Positive skew = few large orders dominate. Negative skew = many small orders. Captures order placement strategies and presence of large resting orders (institutional, icebergs).

**Market Interpretation:**
**Positive skewness (right-skewed):** Few levels have very large quantities, most have small. Suggests: (1) large institutional orders at specific levels, (2) iceberg orders revealing themselves, (3) concentrated liquidity provision.

**Negative skewness (left-skewed):** Many small orders, few outliers. Suggests: (1) order splitting behavior, (2) distributed market making, (3) retail-dominated flow.

**Near-zero skewness (symmetric):** Uniform distribution or normal-like. Balanced, competitive market making.

**Asymmetric skewness (bid_skew ≠ ask_skew):** Different strategies on different sides. High bid skew + low ask skew might indicate institutional buyer with large resting bid vs distributed ask-side liquidity.

**Related Concepts:**
Order splitting strategies, iceberg detection, institutional order identification, market maker heterogeneity, fat tails in liquidity distribution.

**Usage in Your Pipeline:**

```python
df['bid_skewness'] = compute_book_skewness(df, side='bid', n_levels=10)
df['ask_skewness'] = compute_book_skewness(df, side='ask', n_levels=10)
df['skewness_asymmetry'] = df['bid_skewness'] - df['ask_skewness']
```

**Why This Is Powerful:**
Skewness captures **shape** of the distribution that mean and variance miss. Two books can have identical total depth and imbalance but very different skewness—revealing different underlying order placement strategies.

**For Regime Detection (Week 8):** Different market participants (retail, HFT, institutional) create different skewness patterns. Can identify participant mix from skewness.

**For Informed Trading Detection:** Large skewness values often correlate with large informed orders. Unusual skewness spikes may predict price movements.

**Academic References:**

- Large (2007) - "Measuring the Resiliency of an Electronic Limit Order Book"
- Cont & de Larrard (2013) - "Price Dynamics in a Markovian Limit Order Market"

**Recommendation for Phase 2:**
Optional advanced feature for Week 9. Computationally simple (just scipy.stats.skew) but conceptually sophisticated. Good for PCA to capture distributional shape. Excellent interview topic: "I computed higher-order moments to detect institutional order placement patterns."

---

### Book Kurtosis

**What It Is:**
The fourth moment of the quantity distribution, measuring tail heaviness (presence of extreme values/outliers) in order quantities.

**Equation:**

```
kurtosis_bid = (1/N) × sum((qty_i - mean_qty)⁴) / std_qty⁴ - 3
```

(excess kurtosis, where 3 is subtracted so normal distribution has kurtosis=0)

**Motivation:**
Kurtosis captures tail behavior—how often extreme order sizes appear. High kurtosis = fat tails = frequent large orders. Low kurtosis = thin tails = uniform order sizes. Reveals presence of outlier orders that might signal informed trading or institutional positioning.

**Market Interpretation:**
**High kurtosis (>3):** Fat tails, frequent large orders. Suggests: (1) presence of institutional orders, (2) heterogeneous market participants, (3) occasional large informed trades, (4) iceberg orders periodically revealing size.

**Low/negative kurtosis (<0):** Thin tails, uniform distribution. Suggests: (1) homogeneous market making, (2) order size limits or regulations, (3) algorithmic order splitting creating uniform sizes.

**Increasing kurtosis:** Large orders entering the book. Potential information event or institutional positioning. May precede price movement.

**Kurtosis spikes:** Occasional very large orders. Monitor for execution or cancellation—can signal informed activity.

**Related Concepts:**
Fat tails, outlier detection, large order identification, informed trading signals, iceberg orders, institutional flow detection.

**Usage in Your Pipeline:**

```python
df['bid_kurtosis'] = compute_book_kurtosis(df, side='bid', n_levels=10)
df['ask_kurtosis'] = compute_book_kurtosis(df, side='ask', n_levels=10)
df['kurtosis_asymmetry'] = df['bid_kurtosis'] - df['ask_kurtosis']
```

**Why This Is Powerful:**
Kurtosis captures tail events that are often the most informative. Large orders are more likely to be informed. Detecting when fat tails appear (high kurtosis regime) helps identify information arrival.

**Combined with skewness:** Together, skewness and kurtosis fully characterize the shape of the distribution beyond mean/variance. Can identify specific distribution families (e.g., Pareto, exponential, gamma).

**For Outlier Detection:** High kurtosis periods warrant closer inspection. May contain predictive large orders.

**Academic References:**

- Mandelbrot (1963) - "The Variation of Certain Speculative Prices" (fat tails in finance)
- Cont (2001) - "Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues"

**Recommendation for Phase 2:**
Optional advanced feature for Week 9. Pair with skewness for complete distribution characterization. Most useful when you suspect large informed orders—kurtosis will spike when they appear. Good for filtering: "high kurtosis + high OFI = strong signal."

---

## 🔄 Market Resilience Features

### Book Resilience

**What It Is:**
The rate at which order book depth replenishes after being depleted by trades or cancellations, measuring how quickly market makers refill liquidity.

**Equation:**

```
resilience = Δdepth / Δtime
```

specifically: rate of liquidity recovery after a depletion event.

Or: time constant τ from exponential recovery model:

```
depth(t) = depth_final - (depth_final - depth_min) × exp(-t/τ)
resilience = 1/τ
```

**Motivation:**
Resilient markets quickly recover from shocks. Fragile markets take long to refill depth. Resilience measures market maker competition, confidence, and willingness to provide liquidity. Critical for understanding how markets absorb large orders.

**Market Interpretation:**
**High resilience (fast recovery):** Competitive market making, multiple liquidity providers, high market maker confidence. Market can absorb repeated large orders without lasting damage.

**Low resilience (slow recovery):** Few market makers, low confidence, risk aversion. Large orders create lasting depth holes. Vulnerable to predatory trading.

**Decreasing resilience over time:** Market makers becoming more cautious, potentially sensing information asymmetry or inventory risk. Warning sign of regime transition.

**Asymmetric resilience (bid ≠ ask):** One side recovers faster. Faster bid recovery = market makers eager to buy (bullish signal). Faster ask recovery = eager to sell (bearish signal).

**Related Concepts:**
Liquidity provision speed, market maker competition, post-trade depth dynamics, flash crashes, liquidity crises, predatory trading, adverse selection recovery.

**Usage in Your Pipeline:**

```python
df['book_resilience'] = compute_book_resilience(df, window=5)
df['bid_resilience'] = compute_side_resilience(df, side='bid', window=5)
df['ask_resilience'] = compute_side_resilience(df, side='ask', window=5)
df['resilience_asymmetry'] = df['bid_resilience'] - df['ask_resilience']
```

**Why This Is Powerful:**
Resilience is a dynamic feature that captures market response to events. Static book features don't reveal how quickly the market adapts. Resilience is forward-looking—tells you how the market will behave after the next large order.

**For Risk Management (Phase 5):** Low resilience = dangerous to trade large size. High resilience = safe to be aggressive.

**For Regime Detection (Week 8):** Resilience is excellent regime indicator. Calm = high resilience. Stressed = low resilience. Flash crash precursors show declining resilience.

**For Optimal Execution:** Resilience determines optimal execution speed. High resilience = can trade fast without permanent impact. Low resilience = must trade slowly.

**Academic References:**

- Large (2007) - "Measuring the Resiliency of an Electronic Limit Order Market"
- Biais, Hillion & Spatt (1995) - "An Empirical Analysis of the Limit Order Book and the Order Flow in the Paris Bourse"
- Cont, Kukanov & Stoikov (2014) - "The Price Impact of Order Book Events"

**Recommendation for Phase 2:**
Advanced feature for Week 9 if time permits. Requires tracking depth changes over time and identifying depletion-recovery cycles. Computationally more intensive but highly informative. Excellent for Phase 5 RL market making agents—resilience determines optimal quote placement.

---

### Depth Depletion Rate

**What It Is:**
The rate at which total liquidity is being consumed or withdrawn from the book, indicating aggressive order flow or liquidity withdrawal.

**Equation:**

```
depletion_rate = -Δ(queue_total) / Δtime
```

measured when queue_total is decreasing (negative during depletion).

Can also measure separately:

```
bid_depletion_rate = -Δ(queue_bid_cumulative) / Δtime
ask_depletion_rate = -Δ(queue_ask_cumulative) / Δtime
```

**Motivation:**
Rapid depletion signals aggressive trading, informed order flow, or market maker withdrawal. Helps identify liquidity crises before they become severe. Predicts potential price movements when one side depletes faster than the other.

**Market Interpretation:**
**High depletion rate:** Aggressive order flow consuming liquidity faster than market makers replenish. Often precedes price movement in depletion direction. Can signal information event.

**Low/zero depletion:** Stable book, balanced flow, market makers keeping up with demand. Calm market state.

**Accelerating depletion:** Positive feedback loop—market makers withdrawing as aggression increases. Flash crash precursor pattern.

**Asymmetric depletion (bid >> ask or vice versa):** Strong directional flow. Depleting bid side = aggressive selling pressure, expect downward price move. Depleting ask side = aggressive buying, expect upward move.

**Depletion without price movement:** Potentially trapped orders or market makers testing. Can precede reversal if depletion stops and refills.

**Related Concepts:**
Liquidity crises, predatory trading, informed order flow, adverse selection, flash crashes, market maker withdrawal dynamics, inventory risk.

**Usage in Your Pipeline:**

```python
df['depth_depletion_rate'] = compute_depth_depletion(df, window=10)
df['bid_depletion_rate'] = compute_side_depletion(df, side='bid', window=10)
df['ask_depletion_rate'] = compute_side_depletion(df, side='ask', window=10)
df['depletion_asymmetry'] = df['bid_depletion_rate'] - df['ask_depletion_rate']

# Depletion acceleration (second derivative)
df['depletion_acceleration'] = df['depth_depletion_rate'].diff()
```

**Why This Is Powerful:**
Depletion is an early warning indicator. By the time depth is fully depleted, it's too late—price has already moved. Monitoring depletion rate allows prediction of where the market is heading.

**For Prediction (Week 13):** Depletion asymmetry is highly predictive of next-tick price direction. Strong directional signal.

**For Risk Management (Phase 5):** High depletion = exit market making positions, widen spreads. Low depletion = safe to provide liquidity.

**For Flash Crash Detection:** Accelerating depletion (positive depletion_acceleration) is a key flash crash precursor. Can trigger circuit breakers or position unwinding.

**Academic References:**

- Easley, López de Prado & O'Hara (2012) - "Flow Toxicity and Liquidity in a High Frequency World"
- Kirilenko et al. (2017) - "The Flash Crash: High-Frequency Trading in an Electronic Market"
- Cont & Wagalath (2016) - "Running for the Exit: Distressed Selling and Endogenous Correlation in Financial Markets"

**Recommendation for Phase 2:**
Implement in Week 9 as advanced feature. Requires tracking queue_total over time and computing rate of change. Use rolling window (5-10 seconds) to smooth noise. The depletion_asymmetry feature is particularly valuable for directional prediction in Phase 3 models.

---

## 📦 Volume Concentration (Herfindahl Index)

**What It Is:**
A measure of how concentrated liquidity is across order book levels, borrowed from economics (market concentration index), applied to order book depth distribution.

**Equation:**

```
HHI_bid = sum((bid_qty_i / total_bid_qty)² for i=1..N)
HHI_ask = sum((ask_qty_i / total_ask_qty)² for i=1..N)
```

Range: **1/N to 1**

- **1/N:** Perfect uniform distribution (all levels have equal quantity)
- **1:** Complete concentration (all quantity at one level)

**Motivation:**
Reveals whether liquidity is concentrated at a few levels (fragile, vulnerable) or spread evenly (resilient, robust). Different market maker strategies and market conditions produce different concentration patterns. Concentration affects execution risk and market stability.

**Market Interpretation:**
**High HHI (>0.3):** Volume concentrated at few levels. Could indicate: (1) aggressive market making competing for rebates at best prices, (2) fleeting HFT liquidity that may vanish, (3) large resting orders at specific prices, (4) defensive concentration after uncertainty.

**Low HHI (~1/N, e.g., 0.1 for 10 levels):** Volume evenly distributed. Could indicate: (1) defensive market making expecting volatility, (2) institutional orders spread across levels, (3) algorithmic liquidity provision, (4) high confidence in current price range.

**Rising concentration:** Liquidity pulling toward top of book. May signal: (1) increased confidence (competitive aggressive quotes), (2) increased caution (pulling deep liquidity), (3) HFT dominance increasing.

**Falling concentration:** Liquidity spreading out. May signal: (1) defensive positioning for volatility, (2) institutional depth building, (3) reduced confidence in current price.

**Asymmetric concentration (bid_HHI ≠ ask_HHI):** Different strategies on each side. High bid concentration + low ask concentration might indicate aggressive market making on bids (expect support) vs distributed asks (expect resistance).

**Related Concepts:**
Market maker strategy, fleeting liquidity, quote stuffing, maker-taker pricing, HFT patterns, liquidity concentration risk, iceberg orders, flash crash vulnerability.

**Usage in Your Pipeline:**

```python
df['volume_concentration_bid'] = compute_volume_concentration(df, n_levels=10, side='bid')
df['volume_concentration_ask'] = compute_volume_concentration(df, n_levels=10, side='ask')
df['volume_concentration_avg'] = (df['volume_concentration_bid'] + df['volume_concentration_ask']) / 2
df['volume_concentration_asymmetry'] = df['volume_concentration_bid'] - df['volume_concentration_ask']
```

**Why This Is Powerful:**
**Orthogonal to other features:** Concentration captures distribution **shape** that total depth or imbalance miss. Two books can have same total depth but very different concentration—revealing different risk profiles.

**Concentration + depth = complete picture:**

- High depth + high concentration = aggressive competitive environment (good for small orders, risky for large)
- High depth + low concentration = institutional depth distributed (good for large orders)
- Low depth + high concentration = fleeting HFT liquidity (dangerous, may vanish)
- Low depth + low concentration = defensive, scared market (expensive execution at all sizes)

**For PCA (Week 7):** Concentration contributes unique variance about distribution shape.

**For HMM (Week 8):** Different regimes have distinct concentration signatures. Calm = low concentration (spread out). Volatile = high concentration (pulled to top). Can use concentration as primary regime classifier.

**For Risk Assessment:** High concentration = liquidity could vanish quickly. Monitor for sudden drops. Low concentration = more stable but potentially expensive execution deeper in book.

**Academic References:**

- Herfindahl (1950) - "Concentration in the Steel Industry" (original HHI concept)
- Hendershott, Jones & Menkveld (2011) - "Does Algorithmic Trading Improve Liquidity?"
- Hasbrouck & Saar (2013) - "Low-Latency Trading"
- Brogaard, Hendershott & Riordan (2014) - "High-Frequency Trading and Price Discovery"

**Recommendation for Phase 2:**
**Essential advanced feature.** Compute for both bid and ask sides separately. Include asymmetry and average. Use in Week 8 HMM as key regime indicator. Will be excellent interview topic: "I applied the Herfindahl concentration index to order book depth to measure liquidity fragility and market maker strategy."

---

## 🎯 Feature Selection & Implementation Guidelines

### For PCA (Week 7):

**Aim for 20-40 features before dimensionality reduction:**

- All basic features (spread, mid, returns, volatility)
- Queue depth at 3 levels (top1, top5, cumulative)
- Imbalance features (depth, OFI, OBP, queue at 3 levels)
- Liquidity imbalance at 5 depths [1,2,3,5,10]
- Slope, thickness, concentration
- Price impact at 3 volumes
- VWAP, microprice
- Time features (hour, minute cyclic encoding)

### For HMM Regime Detection (Week 8):

**Focus on regime-indicative features:**

- Volatility (10s, 60s windows) - PRIMARY
- Spread, relative_spread
- Queue depth (top1, cumulative) + changes
- Depth imbalance, OBP
- Volume concentration - IMPORTANT
- Book slope/gradient
- Price impact
- Time of day

### For Tree Models (Week 10-13):

**Include ALL features - trees handle multicollinearity:**

- All features from above
- Raw + normalized versions
- Lagged features (t-1, t-5, t-10)
- Interaction candidates (spread × volatility, OFI × depth)
- Trees will select important ones via feature importance

### For Neural Networks (Week 14-18):

**Normalize/standardize all features:**

- Use PCA-reduced features (5-15 components) for MLP
- For LSTM: sequences of 10-20 timesteps of top features
- For CNN: 2D representation of book (price × volume)
- Remove highly correlated pairs manually
- Use time cyclic encoding (sin/cos)

---

## 📚 Implementation Priority by Week

### Week 6 (LOB Features Basics):

spread, mid_price, relative_spread, weighted_mid, returns (5 lags), volatility (2 windows), queue_depth (3 levels), depth_imbalance, OFI

### Week 7 (PCA Preparation):

- VWAP, microprice, price_impact (3 volumes), liquidity_imbalance (5 depths), queue_imbalance (3 levels), book_slope, thickness, time features (hour/minute)

### Week 8 (HMM Preparation):

- volume_concentration, OBP, multi-level spreads, avg_spread

### Week 9 (Advanced/Optional):

- skewness, kurtosis, toxicity, resilience, depletion_rate, time_since_event features

---

## 🔗 Key Academic References

### Core Market Microstructure:

- Glosten & Milgrom (1985) - Bid-ask spread theory
- Kyle (1985) - Informed trading model
- Hasbrouck (1991) - Information content of trades

### Order Book Dynamics:

- Cont, Stoikov & Talreja (2010) - Stochastic LOB model
- Cont, Kukanov & Stoikov (2014) - Price impact of events
- Cartea, Jaimungal & Penalva (2015) - Algorithmic trading

### Optimal Execution:

- Almgren & Chriss (2001) - Optimal execution framework
- Bouchaud et al. (2009) - Price impact models

### High-Frequency Trading:

- Hendershott, Jones & Menkveld (2011) - Algorithmic trading effects
- Hasbrouck & Saar (2013) - Low-latency trading
- Easley, López de Prado & O'Hara (2012) - Flow toxicity

### Machine Learning Applications:

- Sirignano (2016) - Deep learning for LOB
- Huang & Polak (2011) - LOBSTER data
- Ntakaris et al. (2018) - Feature engineering for LOB prediction

---

## 💡 Interview Discussion Points

When presenting your Phase 2 capstone, emphasize:

1. **Feature engineering sophistication:** "I computed 40+ microstructure features including weighted price measures, depth-dependent imbalance profiles, order book pressure with exponential decay weighting, and volume concentration metrics."

2. **Theoretical grounding:** "Each feature is motivated by market microstructure theory—OFI captures informed order flow from Cont et al., OBP implements Kyle's lambda concept, concentration applies Herfindahl index to liquidity distribution."

3. **Multi-scale analysis:** "I captured book state at multiple scales: immediate (top1), near-term (top5), and deep (cumulative), revealing how different trader types position across depths."

4. **PCA results:** "PCA revealed that X% of variance is explained by the first 5 components, which primarily capture [volatility regime, depth imbalance, slope characteristics]."

5. **HMM regime insights:** "The HMM identified 3 distinct market states characterized by [high vol + steep slopes + high concentration] vs [low vol + flat slopes + distributed liquidity]."

This demonstrates both technical skill and domain understanding—exactly what quant recruiters seek. of the book and execution cost. Essential for Phase 5 execution algorithms. Computing impact at multiple volumes (1, 5, 10 BTC) reveals the depth profile: small-order vs large-order execution environment.

**Impact ratios test market microstructure theory:** Square-root law predicts impact_5btc / impact_1btc ≈ √5 ≈ 2.24. Deviations indicate unusual book structure—super-linear growth suggests liquidity cliff; sub-linear suggests unusually deep book.

**For Regime Detection (Week 8):** Different regimes have dramatically different impact profiles. Calm = low impact. Volatile = high impact. Transition = rapidly changing impact.

**For Optimal Execution (Phase 5):** Direct input to execution algorithms. High impact = need to trade slowly (TWAP). Low impact = can trade aggressively.

**Academic References:**

- Almgren & Chriss (2001) - "Optimal Execution of Portfolio Transactions"
- Bouchaud, Farmer & Lillo (2009) - "How Markets Slowly Digest Changes in Supply and Demand"
- Kyle (1985) - "Continuous Auctions and Insider Trading"
- Tóth et al. (2011) - "Anomalous Price Impact and the Critical Nature of Liquidity"

**Recommendation for Phase 2:**
Compute at 3-5 different volumes to capture the impact curve. Use 1 BTC (small retail), 5 BTC (medium), 10 BTC (large institutional). Include both buy and sell impacts if computational budget allows. The impact_ratio feature is particularly elegant for interviews: "I computed the square-root law coefficient to test microstructure theory."

---

## 📊 Multi-Level Spreads

### Spread at Multiple Levels

**What It Is:**
The price difference between bid and ask at each depth level (not just level 1), showing how spread widens as you go deeper into the book.

**Equation:**

```
spread_level_N = ask_px_N - bid_px_N
```

**Motivation:**
While the best bid-ask spread (level 1) is most commonly quoted, deeper spreads reveal book quality and liquidity provision strategies. Widening spreads at deeper levels indicate thinning liquidity. Consistent spreads across levels indicate robust market making.

**Market Interpretation:**
**Constant spread across levels:** Market makers maintaining tight, uniform spreads throughout the book. High confidence, aggressive liquidity provision. Seen in calm, competitive markets.

**Widening spreads deeper:** Normal pattern. Market makers less willing to commit capital far from current price. Indicates uncertainty about fair value at those levels.

**Dramatically widening spreads:** Thin, fragile book. Liquidity dries up quickly beyond top few levels. High execution risk for larger orders.

**Spread contraction at deeper levels:** Unusual pattern, possibly institutional resting orders or temporary depth at specific price targets.

**Related Concepts:**
Market depth quality, spread decomposition, latent liquidity, limit order placement strategies, price uncertainty.

**Usage in Your Pipeline:**

```python
# Individual level spreads
df['spread_level_2'] = df['ask_px_2'] - df['bid_px_2']
df['spread_level_5'] = df['ask_px_5'] - df['bid_px_5']
df['spread_level_10'] = df['ask_px_10'] - df['bid_px_10']

# Spread ratios (how much wider than top?)
df['spread_ratio_5'] = df['spread_level_5'] / df['spread']
df['spread_ratio_10'] = df['spread_level_10'] / df['spread']
```

**Why This Is Powerful:**
Multi-level spreads capture depth quality in a price-centric way (whereas queue depth captures it in volume-centric way). Together they provide complete depth characterization.

**Spread_ratio features are particularly informative:** ratio ≈ 1.0 means tight book at all levels; ratio >> 1.0 means spread blowout deeper in book. This ratio is more stable than absolute spreads during price changes.

**For Execution Algorithms:** Spread profile determines slicing strategy. Tight deep spreads = can use larger clips. Wide deep spreads = must use smaller clips.

**Academic References:**

- Foucault, Kadan & Kandel (2005) - "Limit Order Book as a Market for Liquidity"
- Parlour (1998) - "Price Dynamics in Limit Order Markets"

**Recommendation for Phase 2:**
Compute spreads at levels 2, 5, and 10. Include spread_ratio features for normalization. Use in Week 8 HMM to characterize regime depth quality: calm regime = low spread ratios, volatile regime = high spread ratios.

---

### Average Spread (Top N Levels)

**What It Is:**
The mean spread across multiple levels, summarizing overall book tightness beyond just the best bid-ask.

**Equation:**

```
avg_spread_topN = mean(spread_level_i for i=1..N)
```

**Motivation:**
Single number summarizing spread quality across the book. More robust than level-1 spread alone, which can be spoofed or fleeting. Average spread captures true depth of competitive pricing.

**Market Interpretation:**
**Low average spread:** Competitive market making across multiple levels. Good liquidity depth. Favorable execution environment.

**High average spread:** Wide spreads throughout the book. Poor liquidity. High execution costs at all sizes.

**avg_spread >> spread_level_1:** Spread blows out beyond top level. Top-of-book liquidity is misleading—don't trust it for large orders.

**avg_spread ≈ spread_level_1:** Uniform tight spreads. Robust, deep market. Safe for larger execution.

**Related Concepts:**
Market quality metrics, depth-adjusted spreads, effective spreads, liquidity scoring.

**Usage in Your Pipeline:**

```python
df['avg_spread_top5'] = compute_avg_spread(df, n_levels=5)
df['avg_spread_top10'] = compute_avg_spread(df, n_levels=10)

# Spread degradation rate
df['spread_degradation'] = (df['avg_spread_top5'] - df['spread']) / df['spread']
```

**Why This Is Powerful:**
Single metric for comparing market quality across time or assets. The spread_degradation feature quantifies how much worse spreads get beyond the top—key for execution sizing. Good feature for PCA and regime detection.

**Academic References:**

- Chordia, Roll & Subrahmanyam (2000) - "Commonality in Liquidity"
- Hasbrouck (2009) - "Trading Costs and Returns for U.S. Equities"

**Recommendation for Phase 2:**
Compute at top 5 and top 10. Include spread_degradation as a depth quality indicator. Use alongside price_impact features for complete execution cost picture.

---

## ⏰ Time-Based Features

### Time Since Event

**What It Is:**
Elapsed time since the last occurrence of a specific market event (spread change, significant price move, depth change), capturing temporal patterns in information arrival and market activity.

**Equation:**

```
time_since_event = t_current - t_last_event
```

where event can be: spread widening/tightening beyond threshold, mid-price move > X bps, depth change > Y%, volatility spike, etc.

**Motivation:**
Information doesn't arrive uniformly. Events cluster temporally (Hawkes process). Long gaps suggest quiet periods; short gaps indicate active trading or information flow. Time since last event helps models understand current market state.

**Market Interpretation:**
**Short time since last event (<5 seconds):** Active market, information flow, potentially elevated volatility. Events are clustering. Hawkes self-exciting behavior.

**Long time since last event (>60 seconds):** Quiet market, low information arrival, typically lower volatility. Mean-reversion regime.

**Accelerating events (decreasing gaps):** Information cascade or volatility feedback loop beginning. Potentially leading to regime transition.

**Decelerating events (increasing gaps):** Market calming down after information event. Transitioning back to quiet regime.

**Related Concepts:**
Hawkes processes (self-exciting point processes), information arrival, event clustering, volatility timing, market microstructure noise.

**Usage in Your Pipeline:**

```python
df['time_since_spread_change'] = compute_time_since_event(df, event='spread_change', threshold=0.01)
df['time_since_price_move'] = compute_time_since_event(df, event='price_move', threshold=1.0)
df['time_since_depth_change'] = compute_time_since_event(df, event='depth_change', threshold=0.1)

# Event frequency (inverse of gap)
df['event_frequency'] = 1.0 / (df['time_since_price_move'] + 1)  # +1 to avoid division by zero
```

**Why This Is Powerful:**
Time features capture information dynamics that static book features miss. Models can learn: "if time_since_price_move < 10 seconds and OFI > 0, expect continuation; if time_since_price_move > 60 seconds, expect mean reversion."

**For HMM (Week 8):** Time since events is excellent regime indicator. Calm regime = long gaps. Active regime = short gaps. Can use event frequency as primary emission variable.

**For Hawkes Modeling (Week 9):** Direct input to Hawkes process estimation. Captures self-exciting dynamics.

**Academic References:**

- Hawkes (1971) - "Spectra of Some Self-Exciting and Mutually Exciting Point Processes"
- Bacry, Mastromatteo & Muzy (2015) - "Hawkes Processes in Finance"
- Filimonov & Sornette (2012) - "Quantifying Reflexivity in Financial Markets"

**Recommendation for Phase 2:**
Implement for Week 9 when studying Hawkes processes. Focus on time_since_price_move and time_since_spread_change. Computationally simple but conceptually sophisticated—good interview topic about information arrival models.

---

### Intraday Patterns (Hour, Minute)

**What It Is:**
Time-of-day indicators capturing intraday seasonality in volatility, liquidity, and trading activity patterns.

**Equation:**

```
hour = timestamp.hour
minute = timestamp.minute
seconds_since_midnight = (hour × 3600) + (minute × 60) + second
```

Can also encode cyclically:

```
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

**Motivation:**
Markets exhibit strong intraday seasonality. Volatility and spreads follow U-shaped patterns (high at open/close, low mid-day). Liquidity varies by time of day. European vs US trading hours affect crypto differently. Capturing time-of-day helps models adapt to predictable patterns.

**Market Interpretation:**
**Market open (9:30-10:00 AM EST for stocks):** High# Limit Order Book Feature Engineering Reference

Complete guide to market microstructure features for HFT machine learning models.

---

## 📊 Basic Features

### Spread

**What It Is:**
The difference between the best ask price and the best bid price, representing the immediate cost of a round-trip trade.

**Equation:**

```
spread = ask_px_1 - bid_px_1
```

**Motivation:**
Measures the cost of immediate execution and market impact. The spread compensates market makers for providing liquidity and bearing inventory risk.

**Market Interpretation:**
Wider spreads indicate lower liquidity, higher volatility, or increased information asymmetry. Narrow spreads suggest competitive, liquid markets with many active market makers. During calm periods, spreads tighten; during volatile or uncertain periods, spreads widen as market makers protect themselves.

**Related Concepts:**
Transaction costs, bid-ask bounce, market making profitability, adverse selection costs, inventory risk.

**Usage in Your Pipeline:**

```python
df['spread'] = df['ask_px_1'] - df['bid_px_1']
```

**Why This Is Powerful:**
Spread is the most fundamental liquidity measure and highly predictive of short-term volatility. It's a key input for execution cost estimation and regime classification. Changes in spread often precede price movements.

**Academic References:**

- Glosten & Milgrom (1985) - "Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders"
- Roll (1984) - "A Simple Implicit Measure of the Effective Bid-Ask Spread"

**Recommendation for Phase 2:**
Essential baseline feature. Compute alongside relative_spread for cross-time comparisons. Use in Week 8 HMM as a primary regime indicator.

---

### Mid Price

**What It Is:**
The midpoint between the best bid and best ask prices, serving as a reference "fair value" for the asset.

**Equation:**

```
mid_price = (ask_px_1 + bid_px_1) / 2
```

**Motivation:**
Provides a central tendency price that averages out the bid-ask spread. Used as the baseline for computing returns, price movements, and other derived features.

**Market Interpretation:**
Represents the theoretical fair value in a frictionless market. Mid-price movements indicate genuine price discovery rather than bid-ask bounce noise. Most ML models predict mid-price direction rather than actual trade prices.

**Related Concepts:**
Price discovery, efficient market hypothesis, microprice (volume-weighted alternative), fair value estimation.

**Usage in Your Pipeline:**

```python
df['mid_price'] = (df['ask_px_1'] + df['bid_px_1']) / 2
```

**Why This Is Powerful:**
Foundation for all return calculations and price movement features. Essential for normalizing other features and creating prediction targets in classification models.

**Academic References:**

- Hasbrouck (1991) - "Measuring the Information Content of Stock Trades"
- Cont, Stoikov & Talreja (2010) - "A Stochastic Model for Order Book Dynamics"

**Recommendation for Phase 2:**
Compute immediately after loading data. Use as the basis for all return features and prediction targets (e.g., predict mid_price direction at t+5 seconds).

---

### Relative Spread

**What It Is:**
The bid-ask spread expressed as a percentage of the mid price, enabling cross-asset and cross-time comparisons.

**Equation:**

```
relative_spread = spread / mid_price
```

**Motivation:**
Normalizes spread by price level. A $1 spread on a $100 asset is very different from a $1 spread on a $50,000 asset. Relative spread makes comparisons meaningful.

**Market Interpretation:**
Typical values for liquid crypto: 0.01%-0.05% (1-5 basis points). Values >0.1% signal illiquidity or stress. Rising relative spread indicates deteriorating market quality. Useful for comparing liquidity across different price regimes.

**Related Concepts:**
Liquidity measurement, market quality metrics, basis points, transaction cost analysis.

**Usage in Your Pipeline:**

```python
df['relative_spread'] = df['spread'] / df['mid_price']
```

**Why This Is Powerful:**
Makes spread comparable across time as price levels change. Essential for detecting abnormal liquidity conditions and regime transitions. More stationary than absolute spread.

**Academic References:**

- Chordia, Roll & Subrahmanyam (2000) - "Commonality in Liquidity"
- Hasbrouck (2009) - "Trading Costs and Returns for U.S. Equities"

**Recommendation for Phase 2:**
Include alongside absolute spread. Use in PCA (Week 7) and as a key feature for HMM regime detection (Week 8) to identify liquidity-driven states.

---

### Weighted Mid Price

**What It Is:**
A volume-weighted midpoint that adjusts for liquidity imbalance at the top of the book, providing a more accurate fair value estimate than simple mid price.

**Equation:**

```
weighted_mid = (ask_px_1 × bid_qty_1 + bid_px_1 × ask_qty_1) / (bid_qty_1 + ask_qty_1)
```

**Motivation:**
When one side of the book has significantly more volume, the simple mid-price doesn't reflect the true balance of supply and demand. Weighted mid shifts toward the side with deeper liquidity.

**Market Interpretation:**
If bid_qty >> ask_qty, weighted_mid shifts below simple mid (downward pressure expected). If ask_qty >> bid_qty, weighted_mid shifts above simple mid (upward pressure expected). The difference between weighted_mid and mid_price is itself a predictive signal.

**Related Concepts:**
Microprice, volume-weighted prices, informed order flow, price pressure effects, queue position.

**Usage in Your Pipeline:**

```python
df['weighted_mid'] = compute_weighted_mid(df)
```

**Why This Is Powerful:**
Superior to simple mid price for short-term price prediction. Research shows weighted mid has lower variance and better forecasting properties. The spread between weighted_mid and mid_price captures immediate directional pressure.

**Academic References:**

- Stoikov & Waeber (2016) - "Reducing Transaction Costs with Low-Latency Trading Algorithms"
- Cartea, Jaimungal & Penalva (2015) - "Algorithmic and High-Frequency Trading"

**Recommendation for Phase 2:**
Essential for Phase 3 mid-price prediction models. Create additional feature: `df['weighted_mid_spread'] = df['weighted_mid'] - df['mid_price']` as a directional indicator.

---

## 📈 Volatility & Price Movement

### Rolling Volatility

**What It Is:**
The standard deviation of returns over a rolling time window, capturing recent price variability and risk.

**Equation:**

```
volatility_Ns = std(return_1s, window=N)
```

where return_1s are 1-second log returns.

**Motivation:**
Markets transition between calm and volatile regimes. Volatility clustering means high volatility persists. Capturing recent volatility helps identify regime changes and adjust trading strategies.

**Market Interpretation:**
Low volatility (< 10 bps/sec) indicates stable, predictable markets suitable for aggressive market making. High volatility (> 50 bps/sec) signals information arrival, uncertainty, or potential regime shift. Market makers widen spreads and reduce depth during high volatility.

**Related Concepts:**
GARCH models, volatility clustering, regime-switching models, VIX, realized volatility, market microstructure noise.

**Usage in Your Pipeline:**

```python
df['volatility_10s'] = compute_rolling_volatility(df, window=10)
df['volatility_60s'] = compute_rolling_volatility(df, window=60)
```

**Why This Is Powerful:**
Multiple volatility windows capture different timescales. Essential for HMM regime detection (Week 8) as primary state variable. Strong predictor of future volatility and spread widening. Tree models use volatility to segment data.

**Academic References:**

- Engle (1982) - "Autoregressive Conditional Heteroscedasticity"
- Andersen et al. (2003) - "Modeling and Forecasting Realized Volatility"

**Recommendation for Phase 2:**
Compute at 10s, 30s, and 60s windows. Use as primary feature for HMM regime classification. Volatility regime is the most important market state to identify.

---

### Returns (Multiple Lags)

**What It Is:**
Logarithmic price changes over various time horizons, capturing momentum and mean-reversion patterns at different scales.

**Equation:**

```
return_Ns = ln(mid_price_t / mid_price_{t-N})
```

**Motivation:**
Returns are more stationary than prices and capture relative changes. Multiple lags reveal momentum (trending) vs mean-reversion (bouncing) behavior. Essential features for price direction prediction.

**Market Interpretation:**
Short lags (1-5s) capture microstructure noise and mean-reversion. Medium lags (10-30s) capture momentum. Long lags (60s+) capture trend. Positive autocorrelation suggests momentum; negative suggests mean-reversion. High-frequency returns exhibit mean-reversion due to bid-ask bounce.

**Related Concepts:**
Momentum strategies, mean reversion, autocorrelation, predictability, market efficiency, bid-ask bounce effects.

**Usage in Your Pipeline:**

```python
df['return_1s'] = compute_returns(df, lag=1, return_type='log')
df['return_5s'] = compute_returns(df, lag=5, return_type='log')
df['return_10s'] = compute_returns(df, lag=10, return_type='log')
df['return_30s'] = compute_returns(df, lag=30, return_type='log')
df['return_60s'] = compute_returns(df, lag=60, return_type='log')
```

**Why This Is Powerful:**
Returns are among the most predictive features for tree models and neural networks. Multiple lags let models learn the optimal lookback period. Can create forward returns as prediction targets: `return_forward_5s` for supervised learning.

**Academic References:**

- Lo & MacKinlay (1988) - "Stock Market Prices Do Not Follow Random Walks"
- Jegadeesh & Titman (1993) - "Returns to Buying Winners and Selling Losers"

**Recommendation for Phase 2:**
Include 5 different lag values (1s, 5s, 10s, 30s, 60s). Use for both features (past returns) and targets (forward returns). Critical for Phase 3 boosted tree models.

---

## 📐 Order Book Shape & Slope

### Book Slope / Volume Decay Rate

**What It Is:**
A measure of how quickly liquidity deteriorates (volume decays) as you move away from the best bid/ask. Quantifies the steepness of the order book depth profile.

**Equation:**

**Linear slope (price-volume):**

```
slope_bid = (bid_px_1 - bid_px_N) / sum(bid_qty_1, ..., bid_qty_N)
```

**Exponential decay rate:**

```
qty_i ≈ qty_1 × exp(-α × i)
slope = -α  (fitted decay parameter)
```

**Log-volume gradient:**

```
gradient = Δlog(qty) / Δlevel = mean(log(qty_{i} / qty_{i+1}))
```

**Motivation:**
Book slope reveals market depth quality and price impact. Steep slopes indicate thin liquidity where large orders have high execution costs. Flat slopes indicate deep, resilient liquidity. Market makers adjust book slope based on their confidence and market conditions.

**Market Interpretation:**
**Steep slope (high α > 0.5):** Liquidity concentrated at top, high price impact, fragile market. Seen during volatile periods or defensive market making. Signals caution and reduced depth provision.

**Flat slope (low α < 0.2):** Liquidity spread evenly, low price impact, resilient market. Seen during calm periods with competitive market making. Signals confidence and aggressive depth provision.

**Asymmetric slopes (bid_slope ≠ ask_slope):** Steeper ask slope suggests market makers expect upward move (pulling ask liquidity). Steeper bid slope suggests expected downward move (pulling bid liquidity). Directional predictor.

**Related Concepts:**
Almgren-Chriss execution model, temporary price impact, Kyle's lambda, liquidity supply curve, market resilience, market maker sentiment.

**Usage in Your Pipeline:**

```python
# Linear slope
df['slope_bid'] = compute_book_slope_linear(df, n_levels=10, side='bid')
df['slope_ask'] = compute_book_slope_linear(df, n_levels=10, side='ask')

# Exponential decay rate (slower but more accurate)
df['decay_rate_bid'] = compute_volume_decay_rate(df, n_levels=10, side='bid')
df['decay_rate_ask'] = compute_volume_decay_rate(df, n_levels=10, side='ask')

# Log-gradient (fastest alternative)
df['log_gradient_bid'] = compute_log_volume_gradient(df, n_levels=10, side='bid')
df['log_gradient_ask'] = compute_log_volume_gradient(df, n_levels=10, side='ask')

# Directional signal
df['slope_asymmetry'] = compute_slope_asymmetry(df, n_levels=10)
df['slope_avg'] = (df['slope_bid'] + df['slope_ask']) / 2
```

**Why This Is Powerful:**
Book slope captures **different information** than depth imbalance or queue features. Depth imbalance measures _how much_ liquidity exists; queue depth measures _where_ it is; book slope measures _how it deteriorates_. This orthogonal information is valuable for PCA and regime detection.

**For PCA (Week 7):** Slope contributes unique variance about market depth quality.

**For HMM (Week 8):** Different regimes have distinct slope signatures. Calm = flat slopes (α ≈ 0.1-0.2), Volatile = steep slopes (α ≈ 0.5-1.0).

**For Prediction (Week 13):** Slope asymmetry is highly predictive of short-term price direction. Tree models learn: "if ask_slope > bid_slope + threshold, predict upward move."

**Academic References:**

- Cont, Kukanov & Stoikov (2014) - "The Price Impact of Order Book Events"
- Bouchaud et al. (2009) - "How Markets Slowly Digest Changes in Supply and Demand"
- Tóth et al. (2011) - "Anomalous Price Impact and the Critical Nature of Liquidity in Financial Markets"
- Donier, Bonart & Bouchaud (2015) - "A Fully Consistent, Minimal Model for Non-Linear Market Impact"

**Recommendation for Phase 2:**
Start with linear slope (fastest). Add log-gradient if you need percentage-based interpretation. Include slope_asymmetry as a directional feature. Use slope_avg to identify market stress in HMM. This will be interview-discussion-worthy: "I measured how liquidity degrades away from mid-price to capture market maker sentiment."

---

### Order Book Thickness

**What It Is:**
The average quantity available per level in the order book, indicating overall book density and liquidity provision intensity.

**Equation:**

```
thickness_bid = mean(bid_qty_1, ..., bid_qty_N)
thickness_ask = mean(ask_qty_1, ..., ask_qty_N)
```

**Motivation:**
Thickness reveals whether market makers are providing dense, continuous liquidity or sparse, patchy liquidity. Complements slope by measuring average depth rather than decay rate.

**Market Interpretation:**
Thick books (high average qty) absorb orders without large price moves. Thin books are fragile to shocks. Increasing thickness suggests growing market maker confidence. Decreasing thickness signals caution or inventory constraints. Asymmetric thickness (bid_thick ≠ ask_thick) provides directional signals.

**Related Concepts:**
Liquidity provision strategies, market maker inventory management, resilience, market depth, order splitting behavior.

**Usage in Your Pipeline:**

```python
df['bid_thickness'] = compute_book_thickness(df, side='bid', n_levels=10)
df['ask_thickness'] = compute_book_thickness(df, side='ask', n_levels=10)
df['thickness_asymmetry'] = df['bid_thickness'] - df['ask_thickness']
df['thickness_avg'] = (df['bid_thickness'] + df['ask_thickness']) / 2
```

**Why This Is Powerful:**
Thickness is complementary to slope. You can have thick books with steep slopes (concentrated liquidity) or thin books with flat slopes (sparse but even distribution). Together they fully characterize book shape. Useful for identifying market maker strategy changes.

**Academic References:**

- Foucault, Kadan & Kandel (2005) - "Limit Order Book as a Market for Liquidity"
- Rosu (2009) - "A Dynamic Model of the Limit Order Book"

**Recommendation for Phase 2:**
Include alongside slope features. Use thickness_avg to filter regimes: thick + flat = healthy market, thin + steep = stressed market. Good feature for Week 8 HMM.

---

## 💧 Liquidity Measures

### VWAP (Volume-Weighted Average Price)

**What It Is:**
The average price weighted by volume across multiple levels, representing realistic execution cost for larger orders that walk the book.

**Equation:**

```
vwap_bid = sum(bid_px_i × bid_qty_i) / sum(bid_qty_i)  for i=1..N
vwap_ask = sum(ask_px_i × ask_qty_i) / sum(ask_qty_i)  for i=1..N
```

**Motivation:**
Simple mid-price or best bid/ask don't reflect execution cost for large orders. VWAP accounts for the need to trade across multiple levels, providing realistic cost estimates.

**Market Interpretation:**
VWAP deviates from mid-price when the book is imbalanced or thin. Large VWAP_ask - mid_price spread indicates high cost to buy. Large mid_price - VWAP_bid spread indicates high cost to sell. VWAP spread > simple spread indicates depth is shallow beyond top level.

**Related Concepts:**
Execution algorithms (VWAP benchmarking), slippage, market impact, trade cost analysis, price impact models.

**Usage in Your Pipeline:**

```python
df['vwap_bid'] = compute_vwap(df, side='bid', n_levels=5)
df['vwap_ask'] = compute_vwap(df, side='ask', n_levels=5)
df['vwap_mid'] = (df['vwap_bid'] + df['vwap_ask']) / 2
df['vwap_spread'] = df['vwap_ask'] - df['vwap_bid']
```

**Why This Is Powerful:**
More realistic execution cost measure than simple spread. VWAP features are essential for Phase 5 execution algorithms. Can compute VWAP at different depths (top 3, top 5, top 10) to capture execution cost at different sizes. The difference between VWAP and mid is a measure of market depth quality.

**Academic References:**

- Almgren & Chriss (2001) - "Optimal Execution of Portfolio Transactions"
- Berkowitz, Logue & Noser (1988) - "The Total Cost of Transactions on the NYSE"

**Recommendation for Phase 2:**
Compute at top 5 levels. Use VWAP_spread instead of simple spread for more accurate cost measurement. Create feature `vwap_deviation = vwap_mid - mid_price` as depth quality indicator.

---

### Effective Spread

**What It Is:**
The difference between volume-weighted average prices on bid and ask sides, representing actual trading cost for realistic-sized orders rather than just top-of-book spread.

**Equation:**

```
effective_spread = vwap_ask - vwap_bid
```

**Motivation:**
Simple spread only considers the best bid/ask, but real execution requires trading across multiple levels. Effective spread accounts for depth effects and provides accurate cost measurement.

**Market Interpretation:**
Effective spread > simple spread indicates thin book beyond top level (high depth-dependent costs). Effective spread ≈ simple spread indicates uniform, deep book. Rising effective spread signals deteriorating depth. Important for comparing liquidity quality across time periods or assets.

**Related Concepts:**
Market quality measurement, realized transaction costs, depth quality, liquidity fragmentation, execution cost analysis.

**Usage in Your Pipeline:**

```python
df['effective_spread'] = df['vwap_ask'] - df['vwap_bid']
df['spread_ratio'] = df['effective_spread'] / df['spread']  # How much worse than top-of-book?
```

**Why This Is Powerful:**
More accurate measure of true trading costs than simple spread. The ratio `effective_spread / spread` reveals depth quality: values near 1.0 indicate robust depth; values > 2.0 indicate thin book beyond best prices. Essential for real-world execution cost modeling.

**Academic References:**

- Bessembinder & Kaufman (1997) - "A Comparison of Trade Execution Costs for NYSE and NASDAQ-Listed Stocks"
- Hasbrouck (2009) - "Trading Costs and Returns for U.S. Equities"

**Recommendation for Phase 2:**
Use alongside simple spread. The spread_ratio feature (effective/simple) is particularly useful for regime detection—thin books have high ratios. Good feature for Phase 3 tree models.

---

### Microprice

**What It Is:**
A volume-weighted mid-price that uses quantities at the best bid and ask to predict the next mid-price movement. More responsive to imbalances than simple mid-price.

**Equation:**

```
microprice = (ask_px_1 × bid_qty_1 + bid_px_1 × ask_qty_1) / (bid_qty_1 + ask_qty_1)
```

**Motivation:**
When bid_qty >> ask_qty, the next price movement is more likely upward (buying pressure), so microprice shifts above mid. When ask_qty >> bid_qty, selling pressure dominates, and microprice shifts below mid. This creates a more accurate "fair value" estimate.

**Market Interpretation:**
Microprice - mid_price reveals directional pressure. Positive difference = buying pressure (microprice above mid). Negative difference = selling pressure (microprice below mid). Research shows microprice has lower variance than mid and better short-term forecasting properties.

**Related Concepts:**
High-frequency prediction, informed order flow, adverse selection, price pressure, queue position theory, optimal pricing for market makers.

**Usage in Your Pipeline:**

```python
df['microprice'] = compute_microprice(df)
df['microprice_deviation'] = df['microprice'] - df['mid_price']
```

**Why This Is Powerful:**
Microprice is one of the best short-term price predictors in academic literature. The microprice_deviation feature captures immediate directional pressure. Essential for Phase 3 and Phase 4 prediction models. Some studies show microprice alone can achieve 55-60% directional accuracy for next-tick prediction.

**Academic References:**

- Stoikov & Waeber (2016) - "Reducing Transaction Costs with Low-Latency Trading Algorithms"
- Cartea, Jaimungal & Penalva (2015) - "Algorithmic and High-Frequency Trading"
- Cont, Kukanov & Stoikov (2014) - "The Price Impact of Order Book Events"

**Recommendation for Phase 2:**
Essential feature. Compute both microprice and microprice_deviation. Use as baseline predictor in Phase 1 capstone. Will be a strong feature in all subsequent models.

---

## 🌊 Order Book Pressure & Flow

### Order Flow Imbalance (OFI)

**What It Is:**
The net change in bid and ask quantities between consecutive snapshots, capturing buying and selling pressure from order arrivals, cancellations, and executions.

**Equation:**

```
ofi_t = Δbid_qty - Δask_qty
```

where Δ denotes change from previous snapshot, summed across levels.

**Motivation:**
OFI captures order flow—the actual trading activity and liquidity changes. Positive OFI means more bids arrived or asks were consumed (bullish). Negative OFI means more asks arrived or bids were consumed (bearish). This flow information is highly predictive of short-term price movements.

**Market Interpretation:**
OFI > 0: Net buying pressure, expect upward price movement. OFI < 0: Net selling pressure, expect downward movement. Large |OFI| indicates aggressive trading or information arrival. OFI clustering (sustained positive or negative) suggests directional momentum. Mean-reverting OFI suggests noise trading.

**Related Concepts:**
Order flow toxicity, informed trading, price impact, adverse selection, market microstructure theory (Glosten-Milgrom, Kyle models).

**Usage in Your Pipeline:**

```python
df['ofi'] = compute_order_flow_imbalance(df, n_levels=10)
df['ofi_normalized'] = df['ofi'] / df['queue_total_cumulative']  # Scale by book size
```

**Why This Is Powerful:**
OFI is one of the most predictive features in HFT literature. It captures **flow** (changes over time) whereas depth imbalance captures **state** (current snapshot). Together they provide complementary information. OFI is essential for Phase 1 capstone and remains important through all phases.

**Academic References:**

- Cont, Kukanov & Stoikov (2014) - "The Price Impact of Order Book Events"
- Lipton, Pesavento & Sotiropoulos (2013) - "Trade Arrival Dynamics and Quote Imbalance in a Limit Order Book"
- Hautsch & Huang (2012) - "The Market Impact of a Limit Order"

**Recommendation for Phase 2:**
Core feature. Compute at multiple levels (top 3, top 5, all levels) to capture different scales of order flow. Normalize by total book volume for comparability. Use raw OFI and normalized OFI as separate features.

---

### Depth Imbalance

**What It Is:**
The normalized difference between total bid and ask quantities, measuring current supply/demand imbalance across multiple price levels.

**Equation:**

```
depth_imbalance = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)
```

where totals are summed across N levels.

**Motivation:**
Captures the overall state of the order book. Unlike OFI (which measures changes), depth imbalance measures the current snapshot of supply vs demand. Persistent imbalances often precede price movements.

**Market Interpretation:**
Values near +1: Strong buy-side liquidity, potential resistance to downward moves, or trapped long positions. Values near -1: Strong sell-side liquidity, potential resistance to upward moves, or trapped short positions. Values near 0: Balanced book, fair price discovery. Extreme values often revert as market makers rebalance.

**Related Concepts:**
Queue position, inventory risk, price pressure theory, market maker inventory management, supply-demand equilibrium.

**Usage in Your Pipeline:**

```python
df['depth_imbalance'] = compute_depth_imbalance(df, n_levels=10)
```

**Why This Is Powerful:**
Complements OFI by capturing state vs flow. Research shows depth imbalance at multiple depths (top 3, top 5, all levels) each provide unique predictive power. Essential for HMM regime detection—different regimes show different imbalance patterns. Good feature for tree models.

**Academic References:**

- Cao, Hansch & Wang (2009) - "The Information Content of an Open Limit-Order Book"
- Cartea & Penalva (2012) - "Where is the Value in High Frequency Trading?"

**Recommendation for Phase 2:**
Compute alongside OFI. Include imbalance at multiple depths (see Liquidity Imbalance at Multiple Depths below). Use as key feature for regime detection in Week 8 HMM.

---

### Liquidity Imbalance at Multiple Depths

**What It Is:**
Depth imbalance computed separately at each depth level (cumulative from level 1 to level N), revealing how imbalance evolves as you go deeper into the book.

**Equation:**

```
liquidity_imbalance_depth_N = (sum_bid_1_to_N - sum_ask_1_to_N) / (sum_bid_1_to_N + sum_ask_1_to_N)
```

**Motivation:**
Different depths reveal different information. Imbalance at level 1 shows immediate pressure. Imbalance at level 5 shows near-term positioning. Imbalance at level 10 shows institutional or informed positioning. Creating a depth profile reveals **where in the book** the pressure lies.

**Market Interpretation:**
**Top-heavy imbalance (high at level 1-3, neutral deeper):** High-frequency traders or aggressive market makers creating short-term pressure. Often mean-reverts quickly.

**Deep imbalance (neutral at top, high at levels 5-10):** Institutional orders or informed traders positioning for larger moves. More persistent signal.

**Uniform imbalance (same across all depths):** Strong directional consensus. Most reliable for prediction.

**Divergent imbalance (opposite signs at different depths):** Conflicting signals, often noisy or transitional state.

**Related Concepts:**
Order book depth profile, informed vs uninformed traders, institutional order placement, iceberg orders, price impact at scale.

**Usage in Your Pipeline:**

```python
# Create imbalance profile at multiple depths
for depth in [1, 2, 3, 5, 10]:
    df[f'liquidity_imbalance_depth_{depth}'] = compute_liquidity_imbalance_at_depth(df, depth=depth)

# Optional: Create depth gradient
df['imbalance_gradient'] = df['liquidity_imbalance_depth_10'] - df['liquidity_imbalance_depth_1']
```

**Why This Is Powerful:**
**For PCA (Week 7):** These depth-wise features reveal **where** the important variance lies. PCA may discover that depths 1-3 vary together (high-frequency regime) or depths 5-10 vary together (institutional regime).

**For HMM (Week 8):** Different market regimes show imbalance at different depths. Calm regime = balanced at all depths. High-frequency regime = imbalance concentrated at top. Institutional regime = deep imbalance with neutral top.

**For Prediction (Week 13):** Tree models can learn: "if top imbalance is positive but deep imbalance is negative, predict mean-reversion." This depth structure adds predictive power beyond simple depth imbalance.

**Academic References:**

- Cao, Hansch & Wang (2009) - "The Information Content of an Open Limit-Order Book"
- Hautsch & Huang (2012) - "The Market Impact of a Limit Order"
- Cartea, Jaimungal & Penalva (2015) - "Algorithmic and High-Frequency Trading" (discusses multi-level effects)

**Recommendation for Phase 2:**
**Required for your capstone.** Compute at depths [1, 2, 3, 5, 10]. These 5 features provide a complete depth profile. In your capstone writeup, visualize how imbalance evolves across depths during different market conditions. This will be an excellent interview discussion topic.

---

### Order Book Pressure (Cumulative Depth-Weighted Imbalance)

**What It Is:**
A sophisticated variant of depth imbalance that weights levels by their proximity to the mid-price, emphasizing actionable liquidity near the top of the book while including deeper levels with reduced influence.

**Equation:**
**Inverse distance weighting:**

```
OBP = sum_{i=1}^{N} (1/i) × (bid_qty_i - ask_qty_i)
```

**Exponential decay weighting:**

```
OBP = sum_{i=1}^{N} exp(-λ × i) × (bid_qty_i - ask_qty_i)
```

where λ controls decay rate (typical values: 0.3-1.0)

**Normalized version (for [-1, 1] range):**

```
OBP_norm = sum w_i(bid_qty_i - ask_qty_i) / sum w_i(bid_qty_i + ask_qty_i)
```

**Motivation:**
Unlike simple depth imbalance (which treats all levels equally), order book pressure recognizes that levels closer to mid-price are more actionable and informationally relevant. A 100 BTC order at level 1 has far more immediate price impact than 100 BTC at level 10. OBP captures this economic reality by weighting imbalances by depth.

**Market Interpretation:**
**Positive OBP:** Weighted buying pressure, particularly at actionable levels near mid-price. Suggests upward price movement more likely. Larger magnitude indicates stronger directional pressure.

**Negative OBP:** Weighted selling pressure concentrated where it matters most. Suggests downward price movement.

**High magnitude OBP:** Strong directional signal in the executable book. More reliable than simple depth imbalance for prediction.

**Low magnitude OBP:** Balanced book with no clear pressure. Neutral or ranging market.

**Key insight:** OBP correctly emphasizes liquidity at the top of the book. A book with 50 BTC at level 1 bid and 10 BTC at level 1 ask will show strong positive OBP even if deeper levels are balanced, because the top-level imbalance is what drives immediate price action.

**Related Concepts:**
Kyle's lambda (price impact parameter), market depth elasticity, informed trading pressure, actionable liquidity, short-term price forecasting, execution cost models.

**Usage in Your Pipeline:**

```python
# Order book pressure with different weighting schemes
df['obp_inverse'] = compute_order_book_pressure(df, n_levels=10, weighting='inverse')
df['obp_exponential'] = compute_order_book_pressure(df, n_levels=10, weighting='exponential')
df['obp_normalized'] = compute_normalized_order_book_pressure(df, n_levels=10, weighting='inverse')

# Optional: Multiple decay rates to capture different horizons
df['obp_fast_decay'] = compute_order_book_pressure(df, n_levels=5, weighting='exponential', lambda_decay=1.0)
df['obp_slow_decay'] = compute_order_book_pressure(df, n_levels=10, weighting='exponential', lambda_decay=0.3)
```

**Why This Is Powerful:**
**Superior predictive power:** Research shows OBP predicts short-term price movements better than unweighted depth imbalance because it emphasizes executable, actionable liquidity rather than treating all depth equally.

**Multi-scale analysis:** Different weighting schemes capture different trading horizons. Fast exponential decay (λ=1.0) focuses on ultra-short-term (next 1-5 seconds), considering only top 3-5 levels. Slow decay (λ=0.3) or inverse weighting captures longer-term positioning by including deeper book.

**Complements other features:** You now have three complementary views:

- **OFI:** Changes over time (flow)
- **Depth imbalance:** Total inventory (state)
- **OBP:** Actionable directional pressure (weighted state)

**For PCA (Week 7):** OBP contributes unique variance by weighting depth by relevance. May load differently than simple depth imbalance.

**For HMM (Week 8):** Different regimes show different OBP patterns. Calm markets have low OBP variance; volatile markets show rapid OBP swings.

**For Prediction (Week 13):** Tree models can learn regime-dependent rules: "In high-volatility regime, use fast-decay OBP; in calm regime, use slow-decay OBP."

**Academic References:**

- Cont, Stoikov & Talreja (2010) - "A Stochastic Model for Order Book Dynamics"
- Cartea, Jaimungal & Penalva (2015) - "Algorithmic and High-Frequency Trading"
- Sirignano (2016) - "Deep Learning for Limit Order Books"
- Huang & Polak (2011) - "LOBSTER: Limit Order Book System - The Efficient Reconstructor"

**Recommendation for Phase 2:**
Start with inverse weighting (simplest, interpretable). Add normalized version for [-1, 1] scaling. If you have time in Week 9, experiment with exponential decay at different λ values and compare predictive power. This feature will be excellent for interviews: "I weighted order book imbalance by depth to emphasize actionable liquidity, improving prediction accuracy over naive depth measures."

---

### Queue Imbalance (Multiple Levels)

**What It Is:**
Depth imbalance computed at specific aggregation levels (top 1, top 5, cumulative), revealing how supply/demand imbalance manifests at different depths of the order book.

**Equation:**

```
queue_imbalance_topN = (queue_bid_topN - queue_ask_topN) / (queue_bid_topN + queue_ask_topN)
```

where queue_bid_topN is sum of bid quantities from level 1 to N.

**Motivation:**
Similar to liquidity imbalance at multiple depths, but focuses on standard aggregation levels (1, 5, all) that correspond to different trading contexts. Top 1 = immediate executable liquidity, Top 5 = near-term resilience, Cumulative = total visible depth.

**Market Interpretation:**
**queue_imbalance_top1:** Immediate pressure at best bid/ask. Most volatile, responds to every order arrival/cancellation. High-frequency signal.

**queue_imbalance_top5:** Near-term pressure within ~$1-2 of mid. More stable than top1. Captures medium-frequency positioning.

**queue_imbalance_cumulative:** Overall book state across all visible levels. Most stable, captures long-term positioning and institutional orders.

**Divergence patterns:**

- **Top1 positive, cumulative negative:** Short-term buying pressure against deeper selling—expect mean reversion
- **All levels positive:** Uniform buying pressure—expect continuation
- **Top1 neutral, cumulative positive:** Deep hidden demand—potential breakout setup

**Related Concepts:**
Queue position theory, inventory risk, market making strategies, depth profile analysis, regime-dependent trading.

**Usage in Your Pipeline:**

```python
# Queue imbalance at standard aggregation levels
df['queue_imbalance_top1'] = compute_queue_imbalance(df, level='top1')
df['queue_imbalance_top5'] = compute_queue_imbalance(df, level='top5')
df['queue_imbalance_cumulative'] = compute_queue_imbalance(df, level='cumulative')

# Divergence features
df['queue_imbalance_divergence'] = df['queue_imbalance_top1'] - df['queue_imbalance_cumulative']
```

**Why This Is Powerful:**
These three levels provide a complete summary of the depth profile without needing individual level-by-level features. More parsimonious than computing imbalance at every depth. Each captures a different timescale and trader type.

**For PCA (Week 7):** These three features may load on different principal components, revealing that different market participants (HFT, institutional) drive variance at different depths.

**For HMM (Week 8):** Regime signatures often appear in the relationship between these three. Calm regime = all three move together. Volatile regime = top1 diverges from deeper levels.

**For Tree Models (Week 13):** Can learn interaction rules: "if top1 > 0.5 and cumulative < 0, predict reversal."

**Academic References:**

- Cao, Hansch & Wang (2009) - "The Information Content of an Open Limit-Order Book"
- Hautsch & Huang (2012) - "The Market Impact of a Limit Order"

**Recommendation for Phase 2:**
Compute all three. These are essential features that provide a complete summary of book imbalance at different scales. Include the divergence feature to capture top vs deep conflicts.

---

### Cumulative Volume Imbalance

**What It Is:**
The absolute (not normalized) difference in total volume between bid and ask sides across all levels, measuring raw supply/demand asymmetry in quantity terms.

**Equation:**

```
cvi = sum_{i=1}^{N} (bid_qty_i - ask_qty_i)
```

**Motivation:**
Unlike normalized depth imbalance (which scales to [-1,1]), cumulative volume imbalance preserves the magnitude of the imbalance. A book with 100 BTC bid vs 50 BTC ask is different from 10 BTC bid vs 5 BTC ask, even though both have the same normalized imbalance ratio.

**Market Interpretation:**
**Large positive CVI:** Significant excess bid-side liquidity. Could indicate: (1) trapped long positions unable to exit, (2) institutional accumulation, (3) defensive market making after downward move.

**Large negative CVI:** Excess ask-side liquidity. Could indicate: (1) trapped shorts, (2) institutional distribution, (3) defensive market making after upward move.

**Near-zero CVI:** Balanced book regardless of total size. Healthy two-sided market.

**Scaling with book size:** CVI should be considered relative to total book depth. 50 BTC imbalance is huge for a thin book but small for a deep book.

**Related Concepts:**
Market making inventory accumulation, liquidity traps, position unwinding, institutional order placement, inventory risk management.

**Usage in Your Pipeline:**

```python
df['cumulative_volume_imbalance'] = compute_cumulative_volume_imbalance(df, n_levels=10)

# Normalized version for cross-time comparison
df['cvi_normalized'] = df['cumulative_volume_imbalance'] / df['queue_total_cumulative']
```

**Why This Is Powerful:**
CVI captures **magnitude** while depth_imbalance captures **ratio**. Both are informative. Large CVI even with neutral ratio suggests massive book size (high confidence). Small CVI with extreme ratio suggests thin, imbalanced book (fragile state). Together they fully characterize the imbalance situation.

**Academic References:**

- Cont, Stoikov & Talreja (2010) - "A Stochastic Model for Order Book Dynamics"
- Rosu (2009) - "A Dynamic Model of the Limit Order Book"

**Recommendation for Phase 2:**
Compute both raw CVI and normalized version. Use raw CVI to detect absolute imbalance magnitude; use normalized for regime comparison. Particularly useful for identifying trapped positions or large institutional orders.

---

### Order Flow Toxicity

**What It Is:**
A measure of adverse selection risk, quantifying how much informed traders (who move prices with small volume) are active versus uninformed traders (large volume, small price impact).

**Equation:**

```
toxicity ≈ |Δprice| / volume_executed
```

or using VPIN (Volume-Synchronized Probability of Informed Trading):

```
VPIN = |buy_volume - sell_volume| / total_volume
```

over recent time window.

**Motivation:**
When informed traders are active, small orders have large price impacts (high toxicity). When uninformed traders dominate, large orders have small price impacts (low toxicity). Market makers widen spreads and reduce depth when toxicity is high to protect against adverse selection.

**Market Interpretation:**
**High toxicity (>threshold):** Informed trading detected. Price movements are driven by small volumes. High information asymmetry. Market makers should be defensive (wide spreads, shallow depth).

**Low toxicity:** Uninformed order flow. Price movements require large volumes. Low information asymmetry. Market makers can be aggressive.

**Rising toxicity:** Information event approaching or occurring. Signals regime transition to volatile/informed state.

**Related Concepts:**
Adverse selection, informed vs uninformed trading, Glosten-Milgrom model, Kyle's lambda, VPIN, flash crashes, market maker risk management.

**Usage in Your Pipeline:**

```python
df['order_flow_toxicity'] = compute_flow_toxicity(df, window=10)

# VPIN alternative
df['vpin'] = compute_vpin(df, window=50)
```

**Why This Is Powerful:**
Toxicity is a sophisticated feature that captures information asymmetry. High toxicity periods are dangerous for market makers but profitable for informed traders. Including toxicity helps models identify when prediction is easier (high toxicity = clear direction) vs harder (low toxicity = noise).

**For HMM (Week 8):** Toxicity is an excellent regime indicator. Can define regimes as: calm/low-toxicity, volatile/high-toxicity, informed/extreme-toxicity.

**For Risk Management (Phase 5):** Essential for optimal execution and market making. High toxicity = reduce position size, widen spreads.

**Academic References:**

- Easley, López de Prado & O'Hara (2012) - "Flow Toxicity and Liquidity in a High Frequency World"
- Easley, Kiefer & O'Hara (1997) - "One Day in the Life of a Very Common Stock"
- Andersen & Bondarenko (2014) - "VPIN and the Flash Crash"

**Recommendation for Phase 2:**
Advanced feature—add in Week 9 if time permits. Requires computing volume and price changes together. Start with simple version (|Δprice|/volume). VPIN requires trade direction classification, which is more complex. This will be impressive in interviews as it shows understanding of informed trading theory.

---

## 📏 Queue Depth Features

### Queue Depth (Top 1, Top 5, Cumulative)

**What It Is:**
The total quantity available at specific depth levels of the order book, measuring immediate, near-term, and total liquidity respectively.

**Equation:**

```
queue_bid_top1 = bid_qty_1
queue_bid_top5 = sum(bid_qty_1, ..., bid_qty_5)
queue_bid_cumulative = sum(bid_qty_1, ..., bid_qty_N)

queue_total_top1 = queue_bid_top1 + queue_ask_top1
queue_total_top5 = queue_bid_top5 + queue_ask_top5
queue_total_cumulative = queue_bid_cumulative + queue_ask_cumulative
```

**Motivation:**
Measures available liquidity at different scales. Top 1 = immediate execution capacity (can I fill my order right now?). Top 5 = near-term resilience (will the book absorb moderate flow?). Cumulative = total visible depth (what's the maximum visible liquidity?).

**Market Interpretation:**
**High queue_total_top1:** Dense liquidity at best prices. Low immediate execution cost. Competitive market making.

**Low queue_total_top1:** Thin top of book. High execution cost. Cautious or defensive market making. Potential for large bid-ask bounce.

**queue_total_top5 >> queue_total_top1:** Liquidity concentrated just beyond best prices. "Hidden depth" or stepped pricing strategy.

**queue_total_cumulative >> queue_total_top5:** Deep book with liquidity spread across many levels. Resilient to large orders.

**Declining queue depths over time:** Liquidity withdrawal, regime transition to volatile state, or pre-announcement positioning.

**Related Concepts:**
Execution cost estimation, market depth, order placement strategy, market maker inventory, liquidity provision incentives.

**Usage in Your Pipeline:**

```python
df = compute_queue_depth(df, n_levels=10)
# This adds 9 columns: queue_bid_top1, queue_ask_top1, queue_total_top1,
#                      queue_bid_top5, queue_ask_top5, queue_total_top5,
#                      queue_bid_cumulative, queue_ask_cumulative, queue_total_cumulative
```

**Why This Is Powerful:**
Queue depth is fundamental for execution algorithms and market making. These three aggregation levels provide complete liquidity summary without redundancy. Essential inputs for:

- Price impact estimation
- Optimal execution (Phase 5 RL agents)
- Regime classification (thin vs thick book regimes)
- Feature normalization (normalize imbalances by queue depth)

**For PCA (Week 7):** Queue depths at different levels may have different variance patterns, revealing liquidity provision strategies.

**For HMM (Week 8):** Calm regime = high stable queue depths. Volatile regime = low volatile queue depths. Transition points show rapid depth changes.

**Academic References:**

- Foucault, Kadan & Kandel (2005) - "Limit Order Book as a Market for Liquidity"
- Parlour (1998) - "Price Dynamics in Limit Order Markets"

**Recommendation for Phase 2:**
Essential features. Compute all three aggregation levels. Use queue_total_cumulative to normalize other features (e.g., OFI / queue_total gives scale-independent OFI).

---

### Depth Concentration

**What It Is:**
The fraction of total liquidity concentrated at the very top of the book, measuring how front-loaded the order book is.

**Equation:**

```
depth_concentration = queue_total_top1 / queue_total_cumulative
```

**Motivation:**
Reveals whether liquidity is concentrated at best prices (aggressive, competitive market making) or spread across many levels (defensive, cautious provision). Different market maker strategies and market conditions produce different concentration patterns.

**Market Interpretation:**
**High concentration (>0.4):** Most liquidity at top of book. Could indicate: (1) aggressive market making competing for rebates, (2) fleeting liquidity that cancels quickly, (3) high-frequency traders dominating, (4) narrow spread environment.

**Low concentration (<0.15):** Liquidity spread evenly or concentrated deeper. Could indicate: (1) defensive market making expecting volatility, (2) institutional orders placed away from market, (3) wide spread environment, (4) low confidence in current price.

**Rising concentration:** Market makers becoming more aggressive or confident. Potentially transitioning to calmer regime.

**Falling concentration:** Liquidity withdrawal from top, defensive positioning. Warning sign of potential volatility or stress.

**Related Concepts:**
Market maker strategy, fleeting liquidity, quote stuffing, maker-taker pricing models, high-frequency trading patterns, liquidity concentration risk.

**Usage in Your Pipeline:**

```python
df['depth_concentration'] = compute_depth_concentration(df)

# Separate bid and ask concentration to detect asymmetry
df['bid_concentration'] = df['queue_bid_top1'] / df['queue_bid_cumulative']
df['ask_concentration'] = df['queue_ask_top1'] / df['queue_ask_cumulative']
df['concentration_asymmetry'] = df['bid_concentration'] - df['ask_concentration']
```

**Why This Is Powerful:**
Concentration captures market maker behavior and confidence in ways that raw queue depth cannot. Two books can have same total depth but very different concentration—revealing different market states.

**High depth + high concentration:** Aggressive competitive environment
**High depth + low concentration:** Institutional depth positioned away from market
**Low depth + high concentration:** Fleeting HFT liquidity
**Low depth + low concentration:** Defensive, scared market

**For HMM (Week 8):** Concentration is an excellent regime indicator. Can define regimes based on concentration levels.

**For Risk Assessment:** High concentration = liquidity could vanish quickly (fleeting). Low concentration = stable but potentially expensive execution.

**Academic References:**

- Hendershott, Jones & Menkveld (2011) - "Does Algorithmic Trading Improve Liquidity?"
- Hasbrouck & Saar (2013) - "Low-Latency Trading"

**Recommendation for Phase 2:**
Compute for overall book and separately for bid/ask sides. The asymmetry feature (bid_concentration - ask_concentration) provides directional signal: if bid side more concentrated, market makers are more aggressive on bids (expect upward support).

---

## 💥 Price Impact Features

### Price Impact (Multiple Volumes)

**What It Is:**
The estimated cost of executing an order of given size, measured as the deviation from mid-price to the volume-weighted average execution price when walking the book.

**Equation:**

```
execution_price_V = VWAP to execute volume V
price_impact_V = (execution_price_V - mid_price) / mid_price
```

For buy order: walk up the ask side until cumulative volume ≥ V
For sell order: walk down the bid side until cumulative volume ≥ V

**Motivation:**
Captures market depth quality and resilience. Shows how much price degrades as you execute larger orders. Directly related to execution cost and optimal trading strategies. Different volumes reveal depth at different scales.

**Market Interpretation:**
**Low price_impact:** Deep, liquid book. Large orders can be executed with minimal cost. Resilient market.

**High price_impact:** Thin book. Even moderate orders significantly move the execution price. Fragile, illiquid market.

**Increasing impact with volume:** Normal concave relationship. Impact grows but at decreasing rate (square-root law).

**Super-linear impact:** Impact grows faster than square-root. Indicates very thin book beyond a certain depth threshold.

**Asymmetric impact (buy_impact ≠ sell_impact):** Directional signal. Higher buy impact = thin ask side = resistance to upward moves. Higher sell impact = thin bid side = resistance to downward moves.

**Related Concepts:**
Almgren-Chriss execution model, temporary vs permanent impact, square-root law, market depth elasticity, optimal execution (TWAP, VWAP, POV strategies).

**Usage in Your Pipeline:**

```python
# Price impact at different volumes
df['price_impact_1btc'] = compute_price_impact(df, volume=1.0, side='buy')
df['price_impact_5btc'] = compute_price_impact(df, volume=5.0, side='buy')
df['price_impact_10btc'] = compute_price_impact(df, volume=10.0, side='buy')

# Sell-side impacts
df['price_impact_1btc_sell'] = compute_price_impact(df, volume=1.0, side='sell')

# Impact ratio (measure of non-linearity)
df['impact_ratio_5_1'] = df['price_impact_5btc'] / df['price_impact_1btc']  # Should be ~√5 ≈ 2.24
```

**Why This Is Powerful:**
Price impact directly measures the quality
