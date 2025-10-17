"""
Hawkes Process Feature Extraction for Trade Arrival Analysis
Uses hawkeslib for fitting, with custom feature engineering
"""

import numpy as np
import pandas as pd
from hawkeslib import UnivariateExpHawkesProcess
from typing import Dict, Optional, Tuple
import warnings


class HawkesFeatureExtractor:
    """
    Wrapper around hawkeslib with feature extraction utilities
    """

    def __init__(self):
        self.model = UnivariateExpHawkesProcess()
        self.is_fitted = False
        self.mu = None
        self.alpha = None
        self.beta = None
        self.branching_ratio = None

    def fit(self, event_times: np.ndarray) -> "HawkesFeatureExtractor":
        """
        Fit Hawkes process to event times

        Args:
            event_times: Array of event timestamps (in seconds, sorted)

        Returns:
            self (fitted model)
        """
        # Ensure times are sorted and converted to numpy array
        event_times = np.asarray(event_times)
        if not np.all(event_times[:-1] <= event_times[1:]):
            warnings.warn("Event times not sorted, sorting automatically")
            event_times = np.sort(event_times)

        # Fit using hawkeslib
        self.model.fit(event_times)

        # Extract parameters - hawkeslib returns (mu, alpha, beta)
        params = self.model.get_params()
        self.mu = params[0]
        self.alpha = params[1]
        self.beta = params[2]

        self.branching_ratio = self.alpha / self.beta if self.beta > 0 else 0
        self.is_fitted = True

        return self

    def intensity(self, eval_times: np.ndarray, event_times: np.ndarray) -> np.ndarray:
        """
        Compute intensity λ(t) at specified times

        Args:
            eval_times: Times at which to evaluate intensity
            event_times: Historical event times used for fitting

        Returns:
            Array of intensity values
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.model.intensity(eval_times)

    def simulate(self, T: float, random_state: Optional[int] = None) -> np.ndarray:
        """
        Simulate Hawkes process

        Args:
            T: Simulation horizon (time duration)
            random_state: Random seed for reproducibility

        Returns:
            Simulated event times
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        if random_state is not None:
            np.random.seed(random_state)

        return self.model.sample(T)

    def get_params(self) -> Dict[str, float]:
        """Return fitted parameters as dictionary"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        return {
            "mu": self.mu,
            "alpha": self.alpha,
            "beta": self.beta,
            "branching_ratio": self.branching_ratio,
        }

    def get_feature_vector(self) -> np.ndarray:
        """
        Return parameters as feature vector for ML models

        Returns:
            Array: [mu, alpha, beta, branching_ratio]
        """
        params = self.get_params()
        return np.array(
            [params["mu"], params["alpha"], params["beta"], params["branching_ratio"]]
        )


def extract_hawkes_features_rolling(
    trade_times: np.ndarray,
    window_size: int = 1000,
    stride: int = 100,
    min_events: int = 50,
) -> pd.DataFrame:
    """
    Extract rolling Hawkes parameters as time-series features

    Args:
        trade_times: Trade arrival timestamps (in seconds)
        window_size: Number of trades per window
        stride: Step size for rolling window
        min_events: Minimum events required to fit model

    Returns:
        DataFrame with columns: [timestamp, mu, alpha, beta, branching_ratio]
    """
    features = []
    n_trades = len(trade_times)

    for start_idx in range(0, n_trades - window_size + 1, stride):
        end_idx = start_idx + window_size
        window_times = trade_times[start_idx:end_idx]

        # Skip if too few events
        if len(window_times) < min_events:
            continue

        try:
            # Normalize times to start at 0 for numerical stability
            normalized_times = window_times - window_times[0]

            # Fit Hawkes model
            extractor = HawkesFeatureExtractor()
            extractor.fit(normalized_times)
            params = extractor.get_params()

            # Record features at the end of the window
            features.append(
                {
                    "timestamp": window_times[-1],
                    "mu": params["mu"],
                    "alpha": params["alpha"],
                    "beta": params["beta"],
                    "branching_ratio": params["branching_ratio"],
                    "window_start": window_times[0],
                    "window_end": window_times[-1],
                    "n_events": len(window_times),
                }
            )

        except Exception as e:
            warnings.warn(f"Failed to fit Hawkes at index {start_idx}: {e}")
            continue

    return pd.DataFrame(features)


def detect_excitation_regimes(
    branching_ratio: np.ndarray, threshold: float = 0.7, method: str = "fixed"
) -> np.ndarray:
    """
    Classify periods as high-excitation vs baseline using branching ratio

    Args:
        branching_ratio: Time series of branching ratios
        threshold: Threshold for high excitation
        method: 'fixed' (use threshold) or 'adaptive' (use mean + std)

    Returns:
        Binary array (1 = excited, 0 = baseline)
    """
    if method == "fixed":
        return (branching_ratio > threshold).astype(int)

    elif method == "adaptive":
        mean_br = np.mean(branching_ratio)
        std_br = np.std(branching_ratio)
        adaptive_threshold = mean_br + std_br
        return (branching_ratio > adaptive_threshold).astype(int)

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_excitation_intensity(
    branching_ratio: np.ndarray, window: int = 10
) -> np.ndarray:
    """
    Compute rolling excitation intensity (smoothed branching ratio)

    Args:
        branching_ratio: Time series of branching ratios
        window: Rolling window size for smoothing

    Returns:
        Smoothed excitation intensity
    """
    return pd.Series(branching_ratio).rolling(window=window, center=True).mean().values


def hawkes_regime_features(
    branching_ratio: np.ndarray, threshold: float = 0.7
) -> Dict[str, np.ndarray]:
    """
    Extract multiple regime-based features from Hawkes branching ratio

    Args:
        branching_ratio: Time series of branching ratios
        threshold: Excitation threshold

    Returns:
        Dictionary of regime features
    """
    regimes = detect_excitation_regimes(branching_ratio, threshold)

    # Regime persistence (how long in current regime)
    regime_duration = np.zeros_like(regimes)
    current_duration = 1
    for i in range(1, len(regimes)):
        if regimes[i] == regimes[i - 1]:
            current_duration += 1
        else:
            current_duration = 1
        regime_duration[i] = current_duration

    # Regime transitions
    regime_changes = np.diff(regimes, prepend=regimes[0])

    return {
        "regime_binary": regimes,
        "regime_duration": regime_duration,
        "regime_change": regime_changes,
        "excitation_intensity": compute_excitation_intensity(branching_ratio),
    }
