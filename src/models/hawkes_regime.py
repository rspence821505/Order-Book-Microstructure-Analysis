"""
Hawkes-Based Regime Detection for Market Microstructure

This module provides regime detection using Hawkes process branching ratios
and intensity bursts for identifying high-excitation periods in trading activity.

Key Features:
- Excitation regime detection via branching ratio thresholds
- Burst period identification based on intensity spikes
- Multi-regime classification (Calm/Normal/Excited)
- Regime statistics and transition analysis
- Production-ready regime monitoring

Extracted from: notebooks/20_hawkes_analysis.ipynb, 30_regime_detection.ipynb
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import json
from pathlib import Path
from src.features.hawkes_features import (
    HawkesFeatureExtractor,
    extract_hawkes_features_rolling,
    detect_excitation_regimes as _detect_excitation_regimes,
    hawkes_regime_features as _hawkes_regime_features
)


class HawkesRegimeDetector:
    """
    Hawkes-based regime detector for market microstructure analysis.

    Uses branching ratio and intensity to identify distinct trading regimes
    based on self-excitation levels and burst activity.

    Attributes:
        hawkes_extractor (HawkesFeatureExtractor): Fitted Hawkes model
        branching_ratio_threshold (float): Threshold for high excitation
        intensity_threshold_std (float): Std multiplier for burst detection
        regimes (np.ndarray): Detected regime labels
        burst_periods (np.ndarray): Binary burst period indicators
    """

    def __init__(
        self,
        branching_ratio_threshold: float = 0.7,
        intensity_threshold_std: float = 2.0,
        window_size: int = 200,
        stride: int = 20
    ):
        """
        Initialize Hawkes regime detector.

        Args:
            branching_ratio_threshold: Threshold for high excitation (default: 0.7)
            intensity_threshold_std: Std multiplier for burst detection (default: 2.0)
            window_size: Number of events per rolling window
            stride: Step size for rolling window
        """
        self.branching_ratio_threshold = branching_ratio_threshold
        self.intensity_threshold_std = intensity_threshold_std
        self.window_size = window_size
        self.stride = stride

        # Model components
        self.hawkes_extractor = None
        self.rolling_features = None
        self.regimes = None
        self.burst_periods = None
        self.regime_labels = None

    def fit(
        self,
        event_times: np.ndarray,
        compute_rolling: bool = True
    ) -> "HawkesRegimeDetector":
        """
        Fit Hawkes regime detector on event times.

        Args:
            event_times: Array of event timestamps (in seconds, sorted)
            compute_rolling: If True, compute rolling window features

        Returns:
            self: Fitted HawkesRegimeDetector instance
        """
        # Fit global Hawkes model
        self.hawkes_extractor = HawkesFeatureExtractor()
        self.hawkes_extractor.fit(event_times)

        # Extract rolling features if requested
        if compute_rolling:
            self.rolling_features = extract_hawkes_features_rolling(
                event_times,
                window_size=self.window_size,
                stride=self.stride
            )

            # Detect regimes from rolling branching ratios
            branching_ratios = self.rolling_features['branching_ratio'].values
            self.regimes = detect_excitation_regimes(
                branching_ratios,
                threshold=self.branching_ratio_threshold
            )

            # Detect burst periods
            self.burst_periods = self._detect_bursts(event_times)

            # Create regime labels
            self.regime_labels = np.array(['Baseline' if r == 0 else 'High Excitation'
                                          for r in self.regimes])

        return self

    def _detect_bursts(self, event_times: np.ndarray) -> np.ndarray:
        """
        Detect burst periods based on intensity exceeding threshold.

        Args:
            event_times: Event timestamps

        Returns:
            Binary array indicating burst periods for rolling windows
        """
        if self.rolling_features is None:
            raise ValueError("Rolling features not computed. Set compute_rolling=True in fit()")

        # Use baseline intensity (mu) as reference
        mu_values = self.rolling_features['mu'].values
        mu_mean = np.mean(mu_values)
        mu_std = np.std(mu_values)

        # Define burst threshold
        burst_threshold = mu_mean + self.intensity_threshold_std * mu_std

        # Flag periods where baseline intensity exceeds threshold
        burst_periods = (mu_values > burst_threshold).astype(int)

        return burst_periods

    def predict(
        self,
        event_times: np.ndarray,
        method: str = "rolling"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict regimes for new event times.

        Args:
            event_times: Event timestamps to classify
            method: "rolling" for rolling window analysis, "global" for single prediction

        Returns:
            regimes: Regime labels (0 = Baseline, 1 = High Excitation)
            branching_ratios: Computed branching ratios
        """
        if method == "rolling":
            # Extract rolling features
            features = extract_hawkes_features_rolling(
                event_times,
                window_size=self.window_size,
                stride=self.stride
            )

            branching_ratios = features['branching_ratio'].values
            regimes = detect_excitation_regimes(
                branching_ratios,
                threshold=self.branching_ratio_threshold
            )

            return regimes, branching_ratios

        elif method == "global":
            # Fit global model
            extractor = HawkesFeatureExtractor()
            extractor.fit(event_times)
            params = extractor.get_params()
            branching_ratio = params['branching_ratio']

            regime = 1 if branching_ratio > self.branching_ratio_threshold else 0

            return np.array([regime]), np.array([branching_ratio])

        else:
            raise ValueError(f"Unknown method: {method}")

    def get_regime_stats(self) -> Dict:
        """
        Compute comprehensive regime statistics.

        Returns:
            Dictionary containing regime statistics and transitions
        """
        if self.regimes is None:
            raise ValueError("Regimes not computed. Call fit() first.")

        # Count regimes
        unique_regimes, counts = np.unique(self.regimes, return_counts=True)
        regime_distribution = {
            self.regime_labels[0]: int(counts[0]) if 0 in unique_regimes else 0,
            'High Excitation': int(counts[1]) if 1 in unique_regimes else 0
        }

        # Regime transitions
        transitions = np.sum(np.diff(self.regimes) != 0)
        avg_duration = len(self.regimes) / (transitions + 1) if transitions > 0 else len(self.regimes)

        # Branching ratio statistics by regime
        br_baseline = self.rolling_features['branching_ratio'][self.regimes == 0]
        br_excited = self.rolling_features['branching_ratio'][self.regimes == 1]

        regime_stats = {
            'Baseline': {
                'mean_br': float(br_baseline.mean()) if len(br_baseline) > 0 else 0.0,
                'std_br': float(br_baseline.std()) if len(br_baseline) > 0 else 0.0,
                'count': int((self.regimes == 0).sum())
            },
            'High Excitation': {
                'mean_br': float(br_excited.mean()) if len(br_excited) > 0 else 0.0,
                'std_br': float(br_excited.std()) if len(br_excited) > 0 else 0.0,
                'count': int((self.regimes == 1).sum())
            }
        }

        # Global Hawkes parameters
        global_params = self.hawkes_extractor.get_params()

        return {
            'regime_distribution': regime_distribution,
            'transitions': int(transitions),
            'avg_duration': float(avg_duration),
            'regime_stats': regime_stats,
            'global_params': global_params,
            'branching_ratio_threshold': self.branching_ratio_threshold,
            'n_windows': len(self.regimes),
            'burst_periods_detected': int(self.burst_periods.sum()) if self.burst_periods is not None else 0
        }


def detect_excitation_regimes(
    branching_ratio: np.ndarray,
    threshold: float = 0.7,
    method: str = "fixed"
) -> np.ndarray:
    """
    Detect high-excitation regimes from branching ratio time series.

    This is a wrapper around the feature extraction function with additional
    validation and options for production use.

    Args:
        branching_ratio: Time series of branching ratios
        threshold: Threshold for high excitation (default: 0.7)
        method: Detection method ("fixed" or "adaptive")

    Returns:
        Binary regime array (0 = Baseline, 1 = High Excitation)

    Example:
        >>> regimes = detect_excitation_regimes(branching_ratios, threshold=0.7)
        >>> print(f"High excitation: {(regimes == 1).sum()} / {len(regimes)} windows")
    """
    return _detect_excitation_regimes(branching_ratio, threshold=threshold, method=method)


def compute_branching_ratio(
    alpha: Union[float, np.ndarray],
    beta: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Compute branching ratio from Hawkes parameters.

    Branching ratio = α/β represents the fraction of events triggered by
    other events vs. baseline arrivals.

    Args:
        alpha: Excitation parameter(s)
        beta: Decay parameter(s)

    Returns:
        Branching ratio(s)

    Example:
        >>> alpha, beta = 0.5, 1.0
        >>> br = compute_branching_ratio(alpha, beta)
        >>> print(f"Branching ratio: {br:.3f}")
    """
    # Handle division by zero
    if isinstance(beta, np.ndarray):
        result = np.zeros_like(alpha, dtype=float)
        nonzero_mask = beta > 0
        result[nonzero_mask] = alpha[nonzero_mask] / beta[nonzero_mask]
        result[~nonzero_mask] = 0.0
        return result
    else:
        return alpha / beta if beta > 0 else 0.0


def flag_burst_periods(
    event_times: np.ndarray,
    intensity: Optional[np.ndarray] = None,
    hawkes_extractor: Optional[HawkesFeatureExtractor] = None,
    threshold_method: str = "mean_std",
    std_multiplier: float = 2.0,
    absolute_threshold: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flag burst periods based on intensity exceeding threshold.

    Bursts are periods where trade arrival intensity significantly exceeds
    normal levels, indicating high activity or market stress.

    Args:
        event_times: Event timestamps
        intensity: Pre-computed intensity values (optional)
        hawkes_extractor: Fitted HawkesFeatureExtractor (optional, computed if not provided)
        threshold_method: "mean_std" (μ + k*σ) or "percentile" or "absolute"
        std_multiplier: Multiplier for std in "mean_std" method (default: 2.0)
        absolute_threshold: Fixed threshold for "absolute" method

    Returns:
        burst_flags: Binary array indicating burst periods
        intensity_values: Computed or provided intensity values

    Example:
        >>> bursts, intensity = flag_burst_periods(trade_times, std_multiplier=2.0)
        >>> print(f"Burst periods: {bursts.sum()} / {len(bursts)} ({100*bursts.mean():.1f}%)")
    """
    # Compute intensity if not provided
    if intensity is None:
        if hawkes_extractor is None:
            hawkes_extractor = HawkesFeatureExtractor()
            hawkes_extractor.fit(event_times)

        intensity = hawkes_extractor.intensity(event_times, event_times)

    # Determine threshold
    if threshold_method == "mean_std":
        mean_intensity = np.mean(intensity)
        std_intensity = np.std(intensity)
        threshold = mean_intensity + std_multiplier * std_intensity

    elif threshold_method == "percentile":
        # Use 95th percentile as threshold
        threshold = np.percentile(intensity, 95)

    elif threshold_method == "absolute":
        if absolute_threshold is None:
            raise ValueError("absolute_threshold required for 'absolute' method")
        threshold = absolute_threshold

    else:
        raise ValueError(f"Unknown threshold_method: {threshold_method}")

    # Flag bursts
    burst_flags = (intensity > threshold).astype(int)

    return burst_flags, intensity


def classify_multi_regime(
    branching_ratio: np.ndarray,
    low_threshold: float = 0.3,
    high_threshold: float = 0.7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify into multiple regimes: Calm/Normal/Excited.

    Args:
        branching_ratio: Time series of branching ratios
        low_threshold: Threshold between Calm and Normal (default: 0.3)
        high_threshold: Threshold between Normal and Excited (default: 0.7)

    Returns:
        regimes: Numeric regime labels (0=Calm, 1=Normal, 2=Excited)
        regime_labels: String regime labels

    Example:
        >>> regimes, labels = classify_multi_regime(branching_ratios)
        >>> print(pd.Series(labels).value_counts())
    """
    regimes = np.zeros(len(branching_ratio), dtype=int)
    regimes[branching_ratio >= low_threshold] = 1  # Normal
    regimes[branching_ratio >= high_threshold] = 2  # Excited

    regime_labels = np.array(['Calm', 'Normal', 'Excited'])[regimes]

    return regimes, regime_labels


def compute_regime_persistence(regimes: np.ndarray) -> np.ndarray:
    """
    Compute how long the system has been in current regime.

    Args:
        regimes: Time series of regime labels

    Returns:
        Duration in current regime for each timestep

    Example:
        >>> persistence = compute_regime_persistence(regimes)
        >>> print(f"Max persistence: {persistence.max()} timesteps")
    """
    persistence = np.zeros(len(regimes), dtype=int)
    current_duration = 1

    for i in range(1, len(regimes)):
        if regimes[i] == regimes[i-1]:
            current_duration += 1
        else:
            current_duration = 1
        persistence[i] = current_duration

    return persistence


def save_hawkes_regime_model(
    detector: HawkesRegimeDetector,
    model_path: Path,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save Hawkes regime detector and metadata.

    Args:
        detector: Fitted HawkesRegimeDetector instance
        model_path: Path to save directory
        metadata: Optional metadata dictionary

    Example:
        >>> save_hawkes_regime_model(detector, Path("models/hawkes"), metadata={"ticker": "AAPL"})
    """
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    # Save rolling features if available
    if detector.rolling_features is not None:
        detector.rolling_features.to_parquet(model_path / "hawkes_rolling_features.parquet")

    # Save metadata
    meta = {
        'branching_ratio_threshold': detector.branching_ratio_threshold,
        'intensity_threshold_std': detector.intensity_threshold_std,
        'window_size': detector.window_size,
        'stride': detector.stride
    }

    if detector.hawkes_extractor is not None and detector.hawkes_extractor.is_fitted:
        meta['global_params'] = detector.hawkes_extractor.get_params()

    if detector.regimes is not None:
        meta['regime_distribution'] = {
            'Baseline': int((detector.regimes == 0).sum()),
            'High Excitation': int((detector.regimes == 1).sum())
        }

    if metadata is not None:
        meta.update(metadata)

    with open(model_path / "hawkes_regime_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_hawkes_regime_model(model_path: Path) -> Tuple[HawkesRegimeDetector, Dict]:
    """
    Load Hawkes regime detector and metadata.

    Args:
        model_path: Path to model directory

    Returns:
        detector: Loaded HawkesRegimeDetector instance
        metadata: Metadata dictionary

    Example:
        >>> detector, metadata = load_hawkes_regime_model(Path("models/hawkes"))
    """
    model_path = Path(model_path)

    # Load metadata
    with open(model_path / "hawkes_regime_metadata.json", "r") as f:
        metadata = json.load(f)

    # Create detector
    detector = HawkesRegimeDetector(
        branching_ratio_threshold=metadata['branching_ratio_threshold'],
        intensity_threshold_std=metadata['intensity_threshold_std'],
        window_size=metadata['window_size'],
        stride=metadata['stride']
    )

    # Load rolling features if available
    features_path = model_path / "hawkes_rolling_features.parquet"
    if features_path.exists():
        detector.rolling_features = pd.read_parquet(features_path)

        # Reconstruct regimes
        if 'branching_ratio' in detector.rolling_features.columns:
            branching_ratios = detector.rolling_features['branching_ratio'].values
            detector.regimes = detect_excitation_regimes(
                branching_ratios,
                threshold=detector.branching_ratio_threshold
            )

    return detector, metadata
