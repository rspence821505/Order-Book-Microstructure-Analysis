"""
Hidden Markov Model Regime Detection for Market Microstructure

This module provides HMM-based regime detection for identifying distinct market
states (Calm, Volatile, Trending) using Gaussian emissions and Viterbi decoding.

Key Features:
- Multi-state HMM with Gaussian emissions
- Viterbi algorithm for optimal state sequence
- Regime transition analysis and duration estimation
- Feature-based regime characterization
- Production-ready state persistence

Extracted from: notebooks/30_regime_detection.ipynb
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import json
import pickle
from pathlib import Path


class HMMRegimeDetector:
    """
    Hidden Markov Model for market regime detection.

    Uses Gaussian HMM to identify distinct market states based on microstructure
    features like volatility, spread, trade intensity, and order flow.

    Attributes:
        n_states (int): Number of hidden states (regimes)
        model (GaussianHMM): Fitted HMM model
        scaler (StandardScaler): Feature standardization scaler
        state_labels (Dict[int, str]): Mapping from state index to regime name
        feature_names (List[str]): Names of observation features
    """

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42
    ):
        """
        Initialize HMM regime detector.

        Args:
            n_states: Number of hidden states (default: 3 for Calm/Trending/Volatile)
            covariance_type: Type of covariance matrix ("full", "diag", "spherical")
            n_iter: Maximum number of EM iterations
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        # Model components
        self.model = None
        self.scaler = None
        self.state_labels = None
        self.feature_names = None
        self.log_likelihood = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        auto_label: bool = True
    ) -> "HMMRegimeDetector":
        """
        Fit HMM on microstructure features.

        Args:
            X: Observation features (n_samples, n_features)
            feature_names: Names of features (required if X is ndarray)
            auto_label: If True, automatically label states based on volatility

        Returns:
            self: Fitted HMMRegimeDetector instance
        """
        # Handle DataFrame vs ndarray
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            if feature_names is None:
                raise ValueError("feature_names required when X is ndarray")
            self.feature_names = feature_names
            X_array = X

        # Standardize features (HMM with Gaussian emissions expects normalized data)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_array)

        # Initialize and fit Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False
        )

        # Fit using Baum-Welch (EM algorithm)
        self.model.fit(X_scaled)

        # Store log-likelihood
        self.log_likelihood = self.model.score(X_scaled)

        # Auto-label states if requested
        if auto_label:
            self.state_labels = self._auto_label_states(X, X_scaled)
        else:
            # Default numeric labels
            self.state_labels = {i: f"State_{i}" for i in range(self.n_states)}

        return self

    def _auto_label_states(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        X_scaled: np.ndarray
    ) -> Dict[int, str]:
        """
        Automatically label states based on feature characteristics.

        Labels states as Calm/Trending/Volatile based on volatility levels.

        Args:
            X: Original (unscaled) features
            X_scaled: Scaled features

        Returns:
            Dictionary mapping state index to human-readable label
        """
        # Decode states to get state assignments
        _, states = self.model.decode(X_scaled, algorithm="viterbi")

        # Convert X to DataFrame if needed
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X

        # Find a volatility-related column
        vol_candidates = [
            "realized_vol_1min",
            "realized_vol_5min",
            "agg_realized_vol_5min",
            "agg_parkinson_vol",
            "agg_gk_vol",
            "volatility"
        ]
        vol_col = next((col for col in vol_candidates if col in X_df.columns), None)

        if vol_col is None:
            # Fallback: use first column
            vol_col = X_df.columns[0]

        # Compute mean volatility for each state
        state_vol = {}
        for state in range(self.n_states):
            state_mask = states == state
            state_vol[state] = X_df.loc[state_mask, vol_col].mean()

        # Sort states by volatility
        sorted_states = sorted(state_vol.items(), key=lambda x: x[1])

        # Assign labels based on volatility ranking
        if self.n_states == 3:
            labels = {
                sorted_states[0][0]: "Calm",      # Lowest volatility
                sorted_states[1][0]: "Trending",  # Medium volatility
                sorted_states[2][0]: "Volatile"   # Highest volatility
            }
        elif self.n_states == 2:
            labels = {
                sorted_states[0][0]: "Calm",
                sorted_states[1][0]: "Volatile"
            }
        else:
            # For other n_states, use generic labels
            labels = {state: f"State_{i}" for i, (state, _) in enumerate(sorted_states)}

        return labels

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict most likely state sequence using Viterbi algorithm.

        Args:
            X: Observation features (n_samples, n_features)

        Returns:
            State sequence (n_samples,) with numeric state indices
        """
        if self.model is None:
            raise ValueError("HMM not fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        X_scaled = self.scaler.transform(X_array)
        _, states = self.model.decode(X_scaled, algorithm="viterbi")

        return states

    def predict_labels(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict regime labels (e.g., "Calm", "Volatile").

        Args:
            X: Observation features (n_samples, n_features)

        Returns:
            Array of regime labels (n_samples,)
        """
        states = self.predict(X)
        return np.array([self.state_labels[s] for s in states])

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get transition probability matrix as DataFrame.

        Returns:
            DataFrame with transition probabilities (from_state x to_state)
        """
        if self.model is None:
            raise ValueError("HMM not fitted. Call fit() first.")

        labels = [self.state_labels[i] for i in range(self.n_states)]
        return pd.DataFrame(
            self.model.transmat_,
            index=labels,
            columns=labels
        )

    def get_regime_durations(self) -> Dict[str, float]:
        """
        Calculate expected regime duration for each state.

        Expected duration = 1 / (1 - P(stay in same state))

        Returns:
            Dictionary mapping regime name to expected duration in timesteps
        """
        if self.model is None:
            raise ValueError("HMM not fitted. Call fit() first.")

        durations = {}
        for i in range(self.n_states):
            stay_prob = self.model.transmat_[i, i]
            if stay_prob < 1.0:
                duration = 1.0 / (1.0 - stay_prob)
            else:
                duration = np.inf

            durations[self.state_labels[i]] = duration

        return durations


def fit_hmm(
    X: Union[pd.DataFrame, np.ndarray],
    n_states: int = 3,
    feature_names: Optional[List[str]] = None,
    covariance_type: str = "full",
    n_iter: int = 100,
    random_state: int = 42
) -> Tuple[HMMRegimeDetector, np.ndarray, np.ndarray]:
    """
    Fit HMM and decode regime sequence.

    Convenience function for quick HMM fitting and decoding.

    Args:
        X: Observation features (n_samples, n_features)
        n_states: Number of hidden states (default: 3)
        feature_names: Feature names (required if X is ndarray)
        covariance_type: Covariance matrix type
        n_iter: Maximum EM iterations
        random_state: Random seed

    Returns:
        hmm_detector: Fitted HMMRegimeDetector instance
        states: Numeric state sequence (n_samples,)
        regime_labels: String regime labels (n_samples,)

    Example:
        >>> hmm_detector, states, labels = fit_hmm(features_df, n_states=3)
        >>> print(f"Regime distribution: {pd.Series(labels).value_counts()}")
    """
    detector = HMMRegimeDetector(
        n_states=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state
    )

    detector.fit(X, feature_names=feature_names, auto_label=True)
    states = detector.predict(X)
    regime_labels = detector.predict_labels(X)

    return detector, states, regime_labels


def viterbi_decode(
    hmm_detector: HMMRegimeDetector,
    X: Union[pd.DataFrame, np.ndarray]
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Decode most likely state sequence using Viterbi algorithm.

    Args:
        hmm_detector: Fitted HMMRegimeDetector instance
        X: Observation features (n_samples, n_features)

    Returns:
        log_prob: Log probability of most likely sequence
        states: Numeric state sequence (n_samples,)
        regime_labels: String regime labels (n_samples,)

    Example:
        >>> log_prob, states, labels = viterbi_decode(hmm_detector, test_features)
        >>> print(f"Sequence log-probability: {log_prob:.2f}")
    """
    if hmm_detector.model is None:
        raise ValueError("HMM not fitted.")

    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X

    X_scaled = hmm_detector.scaler.transform(X_array)
    log_prob, states = hmm_detector.model.decode(X_scaled, algorithm="viterbi")

    regime_labels = np.array([hmm_detector.state_labels[s] for s in states])

    return log_prob, states, regime_labels


def compute_regime_stats(
    hmm_detector: HMMRegimeDetector,
    X: Union[pd.DataFrame, np.ndarray],
    states: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute comprehensive regime statistics.

    Args:
        hmm_detector: Fitted HMMRegimeDetector instance
        X: Observation features (n_samples, n_features)
        states: Pre-computed state sequence (optional, will decode if not provided)

    Returns:
        Dictionary containing:
            - transition_matrix: Transition probabilities
            - regime_durations: Expected duration for each regime
            - state_distribution: Count and percentage of each state
            - feature_means: Mean feature values by regime
            - feature_stds: Feature standard deviations by regime
            - log_likelihood: Model log-likelihood
            - converged: Whether EM converged

    Example:
        >>> stats = compute_regime_stats(hmm_detector, features_df)
        >>> print(f"Calm duration: {stats['regime_durations']['Calm']:.2f} timesteps")
    """
    if hmm_detector.model is None:
        raise ValueError("HMM not fitted.")

    # Decode states if not provided
    if states is None:
        states = hmm_detector.predict(X)

    # Convert X to DataFrame if needed
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=hmm_detector.feature_names)
    else:
        X_df = X.copy()

    X_df["hmm_state"] = states
    X_df["regime_label"] = [hmm_detector.state_labels[s] for s in states]

    # Transition matrix
    transition_matrix = hmm_detector.get_transition_matrix()

    # Regime durations
    regime_durations = hmm_detector.get_regime_durations()

    # State distribution
    state_counts = pd.Series(states).value_counts().sort_index()
    state_distribution = {
        hmm_detector.state_labels[state]: {
            "count": int(count),
            "percentage": float(100 * count / len(states))
        }
        for state, count in state_counts.items()
    }

    # Feature statistics by regime
    feature_cols = hmm_detector.feature_names
    feature_means = X_df.groupby("regime_label")[feature_cols].mean().to_dict()
    feature_stds = X_df.groupby("regime_label")[feature_cols].std().to_dict()

    return {
        "transition_matrix": transition_matrix.to_dict(),
        "regime_durations": regime_durations,
        "state_distribution": state_distribution,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "log_likelihood": float(hmm_detector.log_likelihood),
        "converged": bool(hmm_detector.model.monitor_.converged),
        "n_states": hmm_detector.n_states,
        "n_samples": len(states)
    }


def save_hmm_model(
    hmm_detector: HMMRegimeDetector,
    model_path: Path,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save HMM model and metadata to disk.

    Args:
        hmm_detector: Fitted HMMRegimeDetector instance
        model_path: Path to save directory
        metadata: Optional metadata dictionary

    Example:
        >>> save_hmm_model(hmm_detector, Path("models/hmm"), metadata={"ticker": "AAPL"})
    """
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    # Save HMM model
    with open(model_path / "hmm_model.pkl", "wb") as f:
        pickle.dump(hmm_detector.model, f)

    # Save scaler
    with open(model_path / "hmm_scaler.pkl", "wb") as f:
        pickle.dump(hmm_detector.scaler, f)

    # Save metadata
    meta = {
        "n_states": hmm_detector.n_states,
        "state_labels": hmm_detector.state_labels,
        "feature_names": hmm_detector.feature_names,
        "covariance_type": hmm_detector.covariance_type,
        "n_iter": hmm_detector.n_iter,
        "random_state": hmm_detector.random_state,
        "log_likelihood": hmm_detector.log_likelihood,
        "converged": bool(hmm_detector.model.monitor_.converged) if hmm_detector.model else None
    }

    if metadata is not None:
        meta.update(metadata)

    with open(model_path / "hmm_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_hmm_model(model_path: Path) -> Tuple[HMMRegimeDetector, Dict]:
    """
    Load HMM model and metadata from disk.

    Args:
        model_path: Path to model directory

    Returns:
        hmm_detector: Loaded HMMRegimeDetector instance
        metadata: Metadata dictionary

    Example:
        >>> hmm_detector, metadata = load_hmm_model(Path("models/hmm"))
    """
    model_path = Path(model_path)

    # Load metadata
    with open(model_path / "hmm_metadata.json", "r") as f:
        metadata = json.load(f)

    # Create HMMRegimeDetector instance
    detector = HMMRegimeDetector(
        n_states=metadata["n_states"],
        covariance_type=metadata["covariance_type"],
        n_iter=metadata["n_iter"],
        random_state=metadata["random_state"]
    )

    # Load HMM model
    with open(model_path / "hmm_model.pkl", "rb") as f:
        detector.model = pickle.load(f)

    # Load scaler
    with open(model_path / "hmm_scaler.pkl", "rb") as f:
        detector.scaler = pickle.load(f)

    # Set attributes
    detector.state_labels = metadata["state_labels"]
    # Convert string keys back to int for state_labels
    detector.state_labels = {int(k): v for k, v in detector.state_labels.items()}
    detector.feature_names = metadata["feature_names"]
    detector.log_likelihood = metadata.get("log_likelihood")

    return detector, metadata
