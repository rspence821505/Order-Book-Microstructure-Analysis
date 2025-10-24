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
        Compute intensity λ(t) at specified times (vectorized for speed)

        Args:
            eval_times: Times at which to evaluate intensity
            event_times: Historical event times used for fitting

        Returns:
            Array of intensity values
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        eval_times = np.asarray(eval_times)
        event_times = np.asarray(event_times)
        intensities = np.full(len(eval_times), self.mu)

        # Vectorized computation
        for i, t in enumerate(eval_times):
            # Only consider events before time t
            past_events = event_times[event_times < t]
            if len(past_events) > 0:
                # λ(t) = μ + α * Σ exp(-β * (t - t_i))
                time_diffs = t - past_events
                intensities[i] += self.alpha * np.sum(np.exp(-self.beta * time_diffs))

        return intensities

    # def simulate(self, T: float, random_state: Optional[int] = None) -> np.ndarray:
    #     """
    #     Simulate Hawkes process

    #     Args:
    #         T: Simulation horizon (time duration)
    #         random_state: Random seed for reproducibility

    #     Returns:
    #         Simulated event times
    #     """
    #     if not self.is_fitted:
    #         raise ValueError("Model not fitted yet. Call fit() first.")

    #     if random_state is not None:
    #         np.random.seed(random_state)

    #     return self.model.sample(T)

    def simulate(
        self, T: float, random_state: Optional[int] = None, max_events: int = 100000
    ) -> np.ndarray:
        """
        Simulate Hawkes process using Ogata's modified thinning algorithm

        Algorithm:
        1. Use exponential distribution to propose candidate event times
        2. Compute actual intensity at candidate time
        3. Accept/reject based on intensity ratio

        Args:
            T: Simulation horizon (time duration in seconds)
            random_state: Random seed for reproducibility
            max_events: Safety limit on number of events

        Returns:
            Simulated event times as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        if random_state is not None:
            np.random.seed(random_state)

        events = []
        t = 0.0

        # Upper bound on intensity: λ(t) ≤ μ + α/β (when all past events decay slowly)
        # This is the maximum possible intensity
        lambda_bar = self.mu + (self.alpha / self.beta if self.beta > 0 else self.alpha)

        # Safety check
        if lambda_bar <= 0:
            warnings.warn("Invalid intensity bound, returning empty simulation")
            return np.array([])

        iteration = 0
        while t < T and len(events) < max_events:
            iteration += 1

            # Step 1: Generate candidate time from exponential with rate lambda_bar
            # This overestimates the true rate, so we'll thin below
            u = np.random.uniform()
            dt = -np.log(u) / lambda_bar  # Exponential inter-arrival time
            t = t + dt

            if t >= T:
                break

            # Step 2: Compute actual intensity at candidate time t
            # λ(t) = μ + Σ α * exp(-β * (t - t_i)) for all t_i < t
            intensity_t = self.mu
            if len(events) > 0:
                past_events = np.array(events)
                time_diffs = t - past_events
                intensity_t += self.alpha * np.sum(np.exp(-self.beta * time_diffs))

            # Step 3: Accept event with probability λ(t) / λ_bar
            # This is the "thinning" step that corrects for oversampling
            v = np.random.uniform()
            acceptance_prob = intensity_t / lambda_bar

            if v <= acceptance_prob:
                events.append(t)

        if len(events) >= max_events:
            warnings.warn(f"Hit max_events limit ({max_events}), simulation truncated")

        return np.array(events)

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
    branching_ratio: np.ndarray,
    method: str = "binary",
    thresholds: Optional[List[float]] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract multiple regime-based features from Hawkes branching ratio

    Args:
        branching_ratio: Time series of branching ratios
        method: 'binary' (2 regimes) or 'multi' (3+ regimes)
        thresholds: Custom thresholds.
                   For binary: single value (default 0.7)
                   For multi: list of boundaries (default [0.3, 0.5])

    Returns:
        Dictionary of regime features
    """
    if method == "binary":
        threshold = thresholds[0] if thresholds else 0.7
        regimes = detect_excitation_regimes(branching_ratio, threshold)
        regime_labels = ["baseline", "excited"]

    elif method == "multi":
        thresh = thresholds if thresholds else [0.3, 0.5]
        bins = [0] + thresh + [1.0]
        regimes = np.digitize(branching_ratio, bins=thresh)
        regime_labels = ["calm", "normal", "excited"]

    else:
        raise ValueError(f"Unknown method: {method}")

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
        "regime": regimes,
        "regime_labels": regime_labels,
        "regime_duration": regime_duration,
        "regime_change": regime_changes,
        "excitation_intensity": compute_excitation_intensity(branching_ratio),
    }
