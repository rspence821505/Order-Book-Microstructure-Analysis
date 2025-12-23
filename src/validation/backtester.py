"""
Backtesting and Trading Simulation Module

This module provides functions for backtesting trading strategies with realistic constraints:
- Trading simulation with transaction costs
- Performance metrics (Sharpe ratio, max drawdown, win rate, etc.)
- Position sizing strategies
- Confidence-based filtering

Extracted from notebook 70_regime_validation.ipynb
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings


def simulate_trades(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    returns: np.ndarray,
    regimes: Optional[Union[pd.Series, np.ndarray]] = None,
    initial_capital: float = 100000,
    transaction_cost_bps: float = 5,
    position_sizing: str = "fixed",
    position_size_calm: float = 1.0,
    position_size_volatile: float = 0.5,
    confidence_threshold: float = 0.5,
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Simulate trading strategy with realistic constraints.

    Parameters
    ----------
    y_true : np.ndarray
        Actual outcomes (1=up, 0=down)
    y_pred : np.ndarray
        Predicted outcomes (1=up, 0=down)
    y_proba : np.ndarray
        Predicted probabilities for class 1
    returns : np.ndarray
        Actual returns for each period
    regimes : pd.Series or np.ndarray, optional
        Regime labels for each sample (required if position_sizing='regime')
    initial_capital : float, default=100000
        Starting capital in dollars
    transaction_cost_bps : float, default=5
        Transaction cost in basis points per trade
    position_sizing : str, default='fixed'
        Position sizing strategy:
        - 'fixed': Same position size for all trades
        - 'regime': Adjust size based on regime volatility
        - 'confidence': Scale position by prediction confidence
    position_size_calm : float, default=1.0
        Position size multiplier for calm regimes (as fraction of capital)
    position_size_volatile : float, default=0.5
        Position size multiplier for volatile regimes
    confidence_threshold : float, default=0.5
        Minimum confidence to execute trade (distance from 0.5, scaled to [0,1])

    Returns
    -------
    trades_df : pd.DataFrame
        DataFrame with trade-by-trade results containing:
        - step: time step
        - signal: trade direction (1=long, -1=short)
        - prediction: predicted class
        - actual: actual outcome
        - correct: whether prediction was correct
        - confidence: prediction confidence
        - regime: market regime (if provided)
        - position_size: position size used
        - gross_return: return before costs
        - net_return: return after costs
        - trade_pnl: profit/loss for this trade
        - portfolio_value: portfolio value after trade
        - cumulative_pnl: cumulative P&L
    portfolio_values : List[float]
        Portfolio value at each time step

    Examples
    --------
    >>> trades_df, portfolio = simulate_trades(
    ...     y_test, y_pred, y_proba, returns_test,
    ...     initial_capital=100000, transaction_cost_bps=5
    ... )
    >>> print(f"Final portfolio value: ${portfolio[-1]:,.2f}")
    >>> print(f"Total trades: {len(trades_df)}")
    """
    n_samples = len(y_true)
    capital = initial_capital
    portfolio_value = [capital]
    trades = []

    # Convert regimes to array if needed
    if regimes is not None:
        if isinstance(regimes, pd.Series):
            regimes_array = regimes.values
        else:
            regimes_array = regimes
    else:
        regimes_array = None

    for i in range(n_samples):
        # Compute confidence (distance from 0.5, scaled to [0, 1])
        confidence = abs(y_proba[i] - 0.5) * 2

        # Check if we should trade
        if confidence < confidence_threshold:
            # Skip low-confidence predictions
            portfolio_value.append(capital)
            continue

        # Determine position size
        if position_sizing == "fixed":
            position_size = 1.0
        elif position_sizing == "regime":
            if regimes_array is None:
                raise ValueError("position_sizing='regime' requires regimes parameter")
            # Reduce position in volatile regimes
            if regimes_array[i] == "Volatile":
                position_size = position_size_volatile
            else:
                position_size = position_size_calm
        elif position_sizing == "confidence":
            # Scale by confidence
            position_size = confidence
        else:
            position_size = 1.0

        # Trading signal: 1 for buy (go long), -1 for sell (go short)
        if y_pred[i] == 1:
            signal = 1  # Buy
        else:
            signal = -1  # Sell

        # Calculate transaction cost
        transaction_cost = transaction_cost_bps / 10000  # Convert bps to decimal

        # Calculate trade return (adjusted for transaction costs)
        gross_return = signal * returns[i]  # Long or short position
        net_return = gross_return - transaction_cost  # Subtract transaction cost

        # Update capital
        trade_pnl = capital * position_size * net_return
        capital += trade_pnl
        portfolio_value.append(capital)

        # Record trade
        trade_record = {
            "step": i,
            "signal": signal,
            "prediction": int(y_pred[i]),
            "actual": int(y_true[i]),
            "correct": int(y_pred[i] == y_true[i]),
            "confidence": confidence,
            "position_size": position_size,
            "gross_return": gross_return,
            "net_return": net_return,
            "trade_pnl": trade_pnl,
            "portfolio_value": capital,
        }

        if regimes_array is not None:
            trade_record["regime"] = regimes_array[i]

        trades.append(trade_record)

    # Create DataFrame from trades
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        trades_df["cumulative_pnl"] = trades_df["trade_pnl"].cumsum()
    else:
        # Return empty DataFrame with proper columns
        columns = [
            "step", "signal", "prediction", "actual", "correct",
            "confidence", "position_size", "gross_return", "net_return",
            "trade_pnl", "portfolio_value", "cumulative_pnl"
        ]
        if regimes_array is not None:
            columns.insert(6, "regime")
        trades_df = pd.DataFrame(columns=columns)

    return trades_df, portfolio_value


def backtest_strategy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    returns: np.ndarray,
    regimes: Optional[Union[pd.Series, np.ndarray]] = None,
    initial_capital: float = 100000,
    transaction_cost_bps: float = 5,
    position_sizing: str = "fixed",
    confidence_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Run complete backtest and return all results and metrics.

    Convenience function that combines simulate_trades() with
    performance metric calculation.

    Parameters
    ----------
    y_true : np.ndarray
        Actual outcomes
    y_pred : np.ndarray
        Predicted outcomes
    y_proba : np.ndarray
        Predicted probabilities
    returns : np.ndarray
        Actual returns
    regimes : pd.Series or np.ndarray, optional
        Regime labels
    initial_capital : float, default=100000
        Starting capital
    transaction_cost_bps : float, default=5
        Transaction costs in basis points
    position_sizing : str, default='fixed'
        Position sizing strategy
    confidence_threshold : float, default=0.5
        Minimum confidence to trade

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - trades_df: DataFrame with all trades
        - portfolio_values: List of portfolio values
        - metrics: Dict with performance metrics
        - final_capital: Final portfolio value
        - total_return_pct: Total return percentage

    Examples
    --------
    >>> results = backtest_strategy(
    ...     y_test, y_pred, y_proba, returns_test,
    ...     transaction_cost_bps=5, confidence_threshold=0.6
    ... )
    >>> print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
    >>> print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
    """
    # Run simulation
    trades_df, portfolio_values = simulate_trades(
        y_true, y_pred, y_proba, returns, regimes,
        initial_capital, transaction_cost_bps,
        position_sizing, confidence_threshold=confidence_threshold
    )

    # Calculate metrics
    metrics = calculate_performance_metrics(
        trades_df, portfolio_values, initial_capital
    )

    # Compile results
    results = {
        "trades_df": trades_df,
        "portfolio_values": portfolio_values,
        "metrics": metrics,
        "final_capital": portfolio_values[-1],
        "total_return_pct": metrics.get("total_return_pct", 0),
    }

    return results


def calculate_performance_metrics(
    trades_df: pd.DataFrame,
    portfolio_values: List[float],
    initial_capital: float,
) -> Dict[str, float]:
    """
    Calculate comprehensive trading performance metrics.

    Parameters
    ----------
    trades_df : pd.DataFrame
        DataFrame with trade results from simulate_trades()
    portfolio_values : List[float]
        Portfolio value at each time step
    initial_capital : float
        Initial capital amount

    Returns
    -------
    Dict[str, float]
        Dictionary with performance metrics:
        - total_trades: Number of trades executed
        - final_capital: Final portfolio value
        - total_return_pct: Total return percentage
        - sharpe_ratio: Annualized Sharpe ratio
        - max_drawdown: Maximum drawdown percentage
        - win_rate: Percentage of winning trades
        - profit_factor: Ratio of gross profit to gross loss
        - calmar_ratio: Return / max drawdown
        - avg_win: Average winning trade
        - avg_loss: Average losing trade
        - win_loss_ratio: Ratio of avg win to avg loss

    Examples
    --------
    >>> metrics = calculate_performance_metrics(trades_df, portfolio, 100000)
    >>> print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    >>> print(f"Max DD: {metrics['max_drawdown']:.2%}")
    """
    if len(trades_df) == 0:
        return {
            "total_trades": 0,
            "final_capital": initial_capital,
            "total_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "calmar_ratio": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "win_loss_ratio": 0.0,
        }

    final_capital = portfolio_values[-1]
    total_return = (final_capital - initial_capital) / initial_capital

    # Calculate Sharpe ratio
    returns = trades_df["net_return"].values
    if len(returns) > 1 and returns.std() > 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
    else:
        sharpe_ratio = 0.0

    # Calculate maximum drawdown
    max_drawdown = compute_max_drawdown(portfolio_values)

    # Win rate
    win_rate = trades_df["correct"].mean() if len(trades_df) > 0 else 0.0

    # Profit factor (gross profit / gross loss)
    winning_trades = trades_df[trades_df["trade_pnl"] > 0]
    losing_trades = trades_df[trades_df["trade_pnl"] < 0]

    gross_profit = winning_trades["trade_pnl"].sum() if len(winning_trades) > 0 else 0
    gross_loss = abs(losing_trades["trade_pnl"].sum()) if len(losing_trades) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Calmar ratio (return / max drawdown)
    calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0.0

    # Average win/loss
    avg_win = winning_trades["trade_pnl"].mean() if len(winning_trades) > 0 else 0.0
    avg_loss = losing_trades["trade_pnl"].mean() if len(losing_trades) > 0 else 0.0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    return {
        "total_trades": len(trades_df),
        "final_capital": float(final_capital),
        "total_return_pct": float(total_return * 100),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown * 100),
        "win_rate": float(win_rate * 100),
        "profit_factor": float(profit_factor),
        "calmar_ratio": float(calmar_ratio),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "win_loss_ratio": float(win_loss_ratio),
    }


def compute_sharpe(
    returns: Union[np.ndarray, pd.Series],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized Sharpe ratio.

    Parameters
    ----------
    returns : np.ndarray or pd.Series
        Series of returns
    risk_free_rate : float, default=0.0
        Risk-free rate (annualized)
    periods_per_year : int, default=252
        Number of periods per year (252 for daily trading)

    Returns
    -------
    float
        Annualized Sharpe ratio

    Examples
    --------
    >>> sharpe = compute_sharpe(daily_returns, risk_free_rate=0.02)
    >>> print(f"Sharpe Ratio: {sharpe:.2f}")
    """
    if len(returns) == 0:
        return 0.0

    returns_array = np.array(returns)

    if returns_array.std() == 0:
        return 0.0

    # Annualized excess return
    excess_return = returns_array.mean() - (risk_free_rate / periods_per_year)

    # Annualized Sharpe
    sharpe_ratio = (excess_return / returns_array.std()) * np.sqrt(periods_per_year)

    return float(sharpe_ratio)


def compute_max_drawdown(
    portfolio_values: Union[List[float], np.ndarray, pd.Series]
) -> float:
    """
    Compute maximum drawdown from portfolio value series.

    Parameters
    ----------
    portfolio_values : List[float], np.ndarray, or pd.Series
        Portfolio values over time

    Returns
    -------
    float
        Maximum drawdown as a fraction (e.g., 0.15 = 15% drawdown)

    Examples
    --------
    >>> max_dd = compute_max_drawdown(portfolio_values)
    >>> print(f"Maximum Drawdown: {max_dd:.2%}")
    """
    if len(portfolio_values) == 0:
        return 0.0

    portfolio_array = np.array(portfolio_values)

    # Calculate running maximum
    running_max = np.maximum.accumulate(portfolio_array)

    # Calculate drawdown at each point
    drawdown = (portfolio_array - running_max) / running_max

    # Maximum drawdown (most negative value)
    max_drawdown = abs(drawdown.min())

    return float(max_drawdown)


def compute_calmar_ratio(
    total_return: float,
    max_drawdown: float,
) -> float:
    """
    Compute Calmar ratio (return / max drawdown).

    Parameters
    ----------
    total_return : float
        Total return as a fraction (e.g., 0.15 = 15% return)
    max_drawdown : float
        Maximum drawdown as a fraction (e.g., 0.10 = 10% drawdown)

    Returns
    -------
    float
        Calmar ratio

    Examples
    --------
    >>> calmar = compute_calmar_ratio(0.20, 0.10)  # 20% return, 10% drawdown
    >>> print(f"Calmar Ratio: {calmar:.2f}")
    """
    if max_drawdown == 0:
        return 0.0

    return float(total_return / max_drawdown)


def compute_sortino_ratio(
    returns: Union[np.ndarray, pd.Series],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized Sortino ratio (uses downside deviation instead of total volatility).

    Parameters
    ----------
    returns : np.ndarray or pd.Series
        Series of returns
    risk_free_rate : float, default=0.0
        Risk-free rate (annualized)
    periods_per_year : int, default=252
        Number of periods per year

    Returns
    -------
    float
        Annualized Sortino ratio

    Examples
    --------
    >>> sortino = compute_sortino_ratio(daily_returns)
    >>> print(f"Sortino Ratio: {sortino:.2f}")
    """
    if len(returns) == 0:
        return 0.0

    returns_array = np.array(returns)

    # Calculate downside returns (only negative returns)
    downside_returns = returns_array[returns_array < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    # Annualized excess return
    excess_return = returns_array.mean() - (risk_free_rate / periods_per_year)

    # Annualized Sortino
    sortino_ratio = (excess_return / downside_returns.std()) * np.sqrt(periods_per_year)

    return float(sortino_ratio)


def analyze_drawdowns(
    portfolio_values: Union[List[float], np.ndarray],
    min_drawdown_pct: float = 1.0,
) -> pd.DataFrame:
    """
    Analyze all drawdown periods.

    Parameters
    ----------
    portfolio_values : List[float] or np.ndarray
        Portfolio values over time
    min_drawdown_pct : float, default=1.0
        Minimum drawdown percentage to include

    Returns
    -------
    pd.DataFrame
        DataFrame with drawdown periods containing:
        - start_idx: Start index of drawdown
        - end_idx: End index of drawdown
        - recovery_idx: Index where portfolio recovered (or None)
        - depth_pct: Maximum depth of drawdown
        - duration: Number of periods
        - recovery_time: Periods to recover (or None)

    Examples
    --------
    >>> drawdowns = analyze_drawdowns(portfolio_values, min_drawdown_pct=2.0)
    >>> print(f"Found {len(drawdowns)} significant drawdowns")
    """
    portfolio_array = np.array(portfolio_values)
    running_max = np.maximum.accumulate(portfolio_array)
    drawdown = (portfolio_array - running_max) / running_max * 100

    # Find drawdown periods
    in_drawdown = drawdown < -min_drawdown_pct
    drawdown_starts = np.where(np.diff(in_drawdown.astype(int)) == 1)[0] + 1
    drawdown_ends = np.where(np.diff(in_drawdown.astype(int)) == -1)[0] + 1

    # Handle edge cases
    if len(drawdown_starts) == 0:
        return pd.DataFrame(columns=[
            "start_idx", "end_idx", "recovery_idx", "depth_pct", "duration", "recovery_time"
        ])

    if in_drawdown[0]:
        drawdown_starts = np.insert(drawdown_starts, 0, 0)
    if in_drawdown[-1]:
        drawdown_ends = np.append(drawdown_ends, len(portfolio_array) - 1)

    # Analyze each drawdown
    drawdowns = []
    for start, end in zip(drawdown_starts, drawdown_ends):
        dd_slice = drawdown[start:end+1]
        depth = abs(dd_slice.min())

        # Find recovery point
        peak_value = running_max[start]
        recovery_idx = None
        recovery_time = None

        for i in range(end + 1, len(portfolio_array)):
            if portfolio_array[i] >= peak_value:
                recovery_idx = i
                recovery_time = i - start
                break

        drawdowns.append({
            "start_idx": int(start),
            "end_idx": int(end),
            "recovery_idx": int(recovery_idx) if recovery_idx is not None else None,
            "depth_pct": float(depth),
            "duration": int(end - start + 1),
            "recovery_time": int(recovery_time) if recovery_time is not None else None,
        })

    return pd.DataFrame(drawdowns)


def compare_strategies(
    strategies: Dict[str, Dict[str, Any]],
    metric: str = "sharpe_ratio",
) -> pd.DataFrame:
    """
    Compare multiple trading strategies.

    Parameters
    ----------
    strategies : Dict[str, Dict[str, Any]]
        Dictionary mapping strategy names to backtest results
        (output from backtest_strategy())
    metric : str, default='sharpe_ratio'
        Primary metric for ranking

    Returns
    -------
    pd.DataFrame
        Comparison table with all strategies and metrics

    Examples
    --------
    >>> strategies = {
    ...     'Baseline': backtest_strategy(y_test, y_pred1, y_proba1, returns),
    ...     'Regime-Aware': backtest_strategy(y_test, y_pred2, y_proba2, returns),
    ... }
    >>> comparison = compare_strategies(strategies, metric='sharpe_ratio')
    >>> print(comparison)
    """
    comparison_data = []

    for strategy_name, results in strategies.items():
        metrics = results.get("metrics", {})
        comparison_data.append({
            "strategy": strategy_name,
            **metrics
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Sort by metric
    if metric in comparison_df.columns:
        comparison_df = comparison_df.sort_values(metric, ascending=False)

    return comparison_df.reset_index(drop=True)


def bootstrap_sharpe_test(
    returns1: np.ndarray,
    returns2: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Bootstrap test for difference in Sharpe ratios.

    Parameters
    ----------
    returns1 : np.ndarray
        Returns from strategy 1
    returns2 : np.ndarray
        Returns from strategy 2
    n_bootstrap : int, default=1000
        Number of bootstrap iterations
    alpha : float, default=0.05
        Significance level for confidence interval

    Returns
    -------
    Dict[str, float]
        Test results:
        - mean_diff: Mean Sharpe difference (strategy2 - strategy1)
        - ci_lower: Lower bound of CI
        - ci_upper: Upper bound of CI
        - p_value: P-value for one-sided test (strategy2 > strategy1)
        - significant: Whether difference is significant at alpha level

    Examples
    --------
    >>> test_results = bootstrap_sharpe_test(baseline_returns, strategy_returns)
    >>> print(f"P-value: {test_results['p_value']:.4f}")
    >>> print(f"Significant: {test_results['significant']}")
    """
    if len(returns1) == 0 or len(returns2) == 0:
        return {
            "mean_diff": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "p_value": 1.0,
            "significant": False,
        }

    sharpe_diffs = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        idx1 = np.random.choice(len(returns1), len(returns1), replace=True)
        idx2 = np.random.choice(len(returns2), len(returns2), replace=True)

        r1 = returns1[idx1]
        r2 = returns2[idx2]

        sharpe1 = compute_sharpe(r1)
        sharpe2 = compute_sharpe(r2)

        sharpe_diffs.append(sharpe2 - sharpe1)

    sharpe_diffs = np.array(sharpe_diffs)

    # Calculate statistics
    mean_diff = np.mean(sharpe_diffs)
    ci_lower = np.percentile(sharpe_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(sharpe_diffs, 100 * (1 - alpha / 2))
    p_value = np.mean(sharpe_diffs <= 0)  # One-sided test

    return {
        "mean_diff": float(mean_diff),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
    }
