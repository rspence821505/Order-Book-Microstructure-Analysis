"""
PCA with Stability Monitoring for Microstructure Features

This module provides production-ready PCA functionality with time-series stability
monitoring for high-frequency trading applications.

Key Features:
- Automatic component selection based on variance threshold
- Stability analysis across time windows
- Drift monitoring for production deployment
- Feature importance extraction from loadings

Extracted from: notebooks/25_dimensionality_reduction.ipynb
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union
import json
import pickle
from pathlib import Path


class PCAStable:
    """
    PCA with stability monitoring for time-series microstructure features.

    This class extends standard PCA with:
    - Automatic variance-based component selection
    - Stability monitoring across time windows
    - Drift detection for production environments
    - Feature importance extraction

    Attributes:
        pca (PCA): Fitted sklearn PCA model
        scaler (StandardScaler): Fitted sklearn StandardScaler
        n_components (int): Number of selected components
        variance_explained (float): Total variance explained by selected components
        feature_names (List[str]): Names of input features
        stability_score (float): Average loading similarity across windows
    """

    def __init__(
        self,
        target_variance: float = 0.90,
        n_components: Optional[int] = None,
        random_state: int = 42
    ):
        """
        Initialize PCA with stability monitoring.

        Args:
            target_variance: Target cumulative variance to explain (e.g., 0.90 for 90%)
            n_components: Fixed number of components (overrides target_variance if set)
            random_state: Random seed for reproducibility
        """
        self.target_variance = target_variance
        self.n_components_fixed = n_components
        self.random_state = random_state

        # Model components
        self.pca = None
        self.scaler = None
        self.n_components = None
        self.variance_explained = None
        self.feature_names = None
        self.stability_score = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> "PCAStable":
        """
        Fit PCA model with automatic component selection.

        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: Names of features (required if X is ndarray)

        Returns:
            self: Fitted PCAStable instance
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

        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_array)

        # Fit PCA with all components first to determine optimal n
        if self.n_components_fixed is None:
            pca_full = PCA(random_state=self.random_state)
            pca_full.fit(X_scaled)

            # Find number of components for target variance
            cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
            self.n_components = int(np.argmax(cumulative_variance >= self.target_variance) + 1)
        else:
            self.n_components = self.n_components_fixed

        # Fit final PCA with selected components
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca.fit(X_scaled)

        self.variance_explained = self.pca.explained_variance_ratio_.sum()

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform features to principal component space.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            X_pca: Transformed features (n_samples, n_components)
        """
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        X_scaled = self.scaler.transform(X_array)
        return self.pca.transform(X_scaled)

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Fit PCA and transform features in one step.

        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: Names of features (required if X is ndarray)

        Returns:
            X_pca: Transformed features (n_samples, n_components)
        """
        self.fit(X, feature_names)
        return self.transform(X)

    def get_loadings_df(self) -> pd.DataFrame:
        """
        Get feature loadings as DataFrame.

        Returns:
            DataFrame with features as rows and PCs as columns
        """
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        loadings = self.pca.components_.T  # Shape: (n_features, n_components)
        pc_names = [f"PC{i+1}" for i in range(self.n_components)]

        return pd.DataFrame(
            loadings,
            columns=pc_names,
            index=self.feature_names
        )

    def get_top_features(
        self,
        pc_idx: int,
        n_top: int = 10,
        abs_values: bool = True
    ) -> pd.DataFrame:
        """
        Get top features for a principal component.

        Args:
            pc_idx: Index of principal component (0-indexed)
            n_top: Number of top features to return
            abs_values: If True, sort by absolute loading values

        Returns:
            DataFrame with feature names, loadings, and absolute loadings
        """
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        loadings_df = self.get_loadings_df()
        pc_name = f"PC{pc_idx+1}"

        if abs_values:
            abs_loadings = loadings_df[pc_name].abs().sort_values(ascending=False)
            top_features = abs_loadings.head(n_top)
            top_with_sign = loadings_df.loc[top_features.index, pc_name]

            return pd.DataFrame({
                "Feature": top_features.index,
                "Loading": top_with_sign.values,
                "Abs_Loading": top_features.values
            })
        else:
            sorted_loadings = loadings_df[pc_name].sort_values(ascending=False)
            top_features = sorted_loadings.head(n_top)

            return pd.DataFrame({
                "Feature": top_features.index,
                "Loading": top_features.values
            })


def fit_pca(
    X: Union[pd.DataFrame, np.ndarray],
    target_variance: float = 0.90,
    n_components: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    random_state: int = 42
) -> Tuple[PCAStable, np.ndarray]:
    """
    Fit PCA with automatic component selection.

    This is a convenience function for quick PCA fitting without creating
    a PCAStable instance manually.

    Args:
        X: Feature matrix (n_samples, n_features)
        target_variance: Target cumulative variance (e.g., 0.90 for 90%)
        n_components: Fixed number of components (overrides target_variance)
        feature_names: Feature names (required if X is ndarray)
        random_state: Random seed for reproducibility

    Returns:
        pca_model: Fitted PCAStable instance
        X_transformed: Transformed features (n_samples, n_components)

    Example:
        >>> pca_model, X_pca = fit_pca(features_df, target_variance=0.85)
        >>> print(f"Reduced from {features_df.shape[1]} to {X_pca.shape[1]} features")
    """
    pca_model = PCAStable(
        target_variance=target_variance,
        n_components=n_components,
        random_state=random_state
    )

    X_transformed = pca_model.fit_transform(X, feature_names)

    return pca_model, X_transformed


def compute_stability(
    X: Union[pd.DataFrame, np.ndarray],
    n_windows: int = 5,
    target_variance: float = 0.90,
    n_components: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    random_state: int = 42
) -> Dict[str, Union[float, List[Dict]]]:
    """
    Compute PCA stability across time windows.

    Splits data into time windows and measures how stable the PCA structure
    is across them. Higher stability means PCA can be used longer without
    retraining in production.

    Args:
        X: Feature matrix (n_samples, n_features) - must be time-ordered
        n_windows: Number of time windows to split data into
        target_variance: Target cumulative variance for PCA
        n_components: Fixed number of components (overrides target_variance)
        feature_names: Feature names (required if X is ndarray)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - avg_similarity: Average loading similarity (0-1)
            - std_similarity: Standard deviation of similarity
            - window_results: List of per-window results
            - stability_category: "STABLE", "MODERATE", or "UNSTABLE"

    Example:
        >>> stability = compute_stability(features_df, n_windows=5)
        >>> print(f"Stability: {stability['stability_category']}")
        >>> print(f"Average similarity: {stability['avg_similarity']:.4f}")
    """
    # Convert to array if DataFrame
    if isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = X.columns.tolist()
        X_array = X.values
    else:
        X_array = X

    # Standardize once for all windows
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_array)

    # Fit overall PCA for comparison
    pca_overall = PCAStable(
        target_variance=target_variance,
        n_components=n_components,
        random_state=random_state
    )
    pca_overall.fit(X, feature_names)

    # Determine n_components if not fixed
    n_comp = n_components if n_components is not None else pca_overall.n_components

    # Split into windows
    window_size = len(X_scaled) // n_windows
    window_results = []

    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size if i < n_windows - 1 else len(X_scaled)

        X_window = X_scaled[start_idx:end_idx]

        # Fit PCA on window
        pca_window = PCA(n_components=n_comp, random_state=random_state)
        pca_window.fit(X_window)

        # Calculate similarity with overall PCA
        similarity = _loading_similarity(pca_window, pca_overall.pca, n_components=min(5, n_comp))

        window_results.append({
            "window": i + 1,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "n_samples": end_idx - start_idx,
            "variance_explained": float(pca_window.explained_variance_ratio_.sum()),
            "similarity": float(similarity)
        })

    # Compute statistics
    similarities = [r["similarity"] for r in window_results]
    avg_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)

    # Categorize stability
    if avg_similarity > 0.80:
        stability_category = "STABLE"
    elif avg_similarity > 0.60:
        stability_category = "MODERATE"
    else:
        stability_category = "UNSTABLE"

    return {
        "avg_similarity": float(avg_similarity),
        "std_similarity": float(std_similarity),
        "window_results": window_results,
        "stability_category": stability_category,
        "n_windows": n_windows,
        "n_components": n_comp
    }


def monitor_drift(
    pca_baseline: PCAStable,
    X_new: Union[pd.DataFrame, np.ndarray],
    similarity_threshold: float = 0.70
) -> Dict[str, Union[float, bool, str]]:
    """
    Monitor PCA drift in production by comparing new data to baseline.

    This function is used in production to detect when the PCA model needs
    retraining due to changing market microstructure.

    Args:
        pca_baseline: Baseline PCA model (fitted on training data)
        X_new: New data to check for drift (n_samples, n_features)
        similarity_threshold: Threshold for drift detection (default: 0.70)

    Returns:
        Dictionary containing:
            - similarity: Loading similarity between baseline and new PCA
            - drift_detected: True if similarity < threshold
            - recommendation: Action recommendation
            - variance_explained_baseline: Variance explained by baseline
            - variance_explained_new: Variance explained on new data

    Example:
        >>> drift_result = monitor_drift(pca_baseline, new_features)
        >>> if drift_result['drift_detected']:
        >>>     print(f"Drift detected! {drift_result['recommendation']}")
    """
    if pca_baseline.pca is None:
        raise ValueError("Baseline PCA not fitted.")

    # Fit PCA on new data with same number of components
    pca_new = PCAStable(
        n_components=pca_baseline.n_components,
        random_state=pca_baseline.random_state
    )
    pca_new.fit(X_new, feature_names=pca_baseline.feature_names)

    # Calculate similarity
    similarity = _loading_similarity(
        pca_baseline.pca,
        pca_new.pca,
        n_components=min(5, pca_baseline.n_components)
    )

    # Detect drift
    drift_detected = similarity < similarity_threshold

    # Generate recommendation
    if not drift_detected:
        recommendation = "No drift detected. Continue using baseline PCA."
    elif similarity >= 0.60:
        recommendation = "Moderate drift detected. Consider recalibration within 1-2 days."
    else:
        recommendation = "Significant drift detected. Recalibrate PCA immediately."

    return {
        "similarity": float(similarity),
        "drift_detected": bool(drift_detected),
        "recommendation": recommendation,
        "variance_explained_baseline": float(pca_baseline.variance_explained),
        "variance_explained_new": float(pca_new.variance_explained),
        "similarity_threshold": similarity_threshold
    }


def _loading_similarity(
    pca1: PCA,
    pca2: PCA,
    n_components: int = 5
) -> float:
    """
    Calculate loading similarity between two PCA models.

    Computes average absolute correlation between corresponding principal
    components. Higher values indicate more similar PCA structures.

    Args:
        pca1: First PCA model
        pca2: Second PCA model
        n_components: Number of components to compare

    Returns:
        Average absolute correlation (0-1)
    """
    correlations = []
    n_comp = min(n_components, pca1.n_components_, pca2.n_components_)

    for i in range(n_comp):
        # Compute correlation between loadings
        corr = np.corrcoef(pca1.components_[i], pca2.components_[i])[0, 1]
        correlations.append(abs(corr))

    return np.mean(correlations)


def save_pca_model(
    pca_model: PCAStable,
    model_path: Path,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save PCA model and metadata to disk.

    Args:
        pca_model: Fitted PCAStable instance
        model_path: Path to save directory (will create if needed)
        metadata: Optional metadata dictionary to save alongside model

    Example:
        >>> save_pca_model(pca_model, Path("models/pca"), metadata={"ticker": "AAPL"})
    """
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    # Save PCA and scaler
    with open(model_path / "pca_model.pkl", "wb") as f:
        pickle.dump(pca_model.pca, f)

    with open(model_path / "scaler.pkl", "wb") as f:
        pickle.dump(pca_model.scaler, f)

    # Save metadata
    meta = {
        "n_components": pca_model.n_components,
        "variance_explained": pca_model.variance_explained,
        "target_variance": pca_model.target_variance,
        "feature_names": pca_model.feature_names,
        "stability_score": pca_model.stability_score,
        "random_state": pca_model.random_state
    }

    if metadata is not None:
        meta.update(metadata)

    with open(model_path / "pca_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_pca_model(model_path: Path) -> Tuple[PCAStable, Dict]:
    """
    Load PCA model and metadata from disk.

    Args:
        model_path: Path to model directory

    Returns:
        pca_model: Loaded PCAStable instance
        metadata: Metadata dictionary

    Example:
        >>> pca_model, metadata = load_pca_model(Path("models/pca"))
    """
    model_path = Path(model_path)

    # Load metadata
    with open(model_path / "pca_metadata.json", "r") as f:
        metadata = json.load(f)

    # Create PCAStable instance
    pca_model = PCAStable(
        target_variance=metadata["target_variance"],
        n_components=metadata["n_components"],
        random_state=metadata["random_state"]
    )

    # Load PCA and scaler
    with open(model_path / "pca_model.pkl", "rb") as f:
        pca_model.pca = pickle.load(f)

    with open(model_path / "scaler.pkl", "rb") as f:
        pca_model.scaler = pickle.load(f)

    # Set attributes
    pca_model.n_components = metadata["n_components"]
    pca_model.variance_explained = metadata["variance_explained"]
    pca_model.feature_names = metadata["feature_names"]
    pca_model.stability_score = metadata.get("stability_score")

    return pca_model, metadata
