"""
Tree-Based Model Wrappers for Microstructure Prediction

This module provides production-ready wrappers for tree-based classifiers:
- Decision Trees
- Random Forests (bagging)
- Gradient Boosting (sequential boosting)

Key Features:
- Automatic hyperparameter tuning with time-series CV
- Feature importance extraction
- Model persistence and loading
- Performance monitoring and comparison
- Production-ready inference

Extracted from: notebooks/40_decision_trees.ipynb, 45_random_forest.ipynb, 50_gradient_boosting.ipynb
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import pickle
import json
from pathlib import Path
import time


class DecisionTreeWrapper:
    """
    Production wrapper for Decision Tree classifier.

    Provides automatic hyperparameter tuning, feature importance,
    and model persistence for decision tree models.

    Attributes:
        model (DecisionTreeClassifier): Fitted sklearn decision tree
        best_params (Dict): Best hyperparameters from tuning
        feature_names (List[str]): Names of input features
        training_time (float): Time taken to train model
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 100,
        min_samples_leaf: int = 10,
        criterion: str = "entropy",
        random_state: int = 42
    ):
        """
        Initialize Decision Tree wrapper.

        Args:
            max_depth: Maximum tree depth (None for unlimited)
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples in leaf node
            criterion: Split criterion ("gini" or "entropy")
            random_state: Random seed for reproducibility
        """
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=random_state,
            class_weight="balanced"
        )
        self.best_params = None
        self.feature_names = None
        self.training_time = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> "DecisionTreeWrapper":
        """
        Fit decision tree model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: Names of features

        Returns:
            self: Fitted wrapper instance
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            X_array = X

        start_time = time.time()
        self.model.fit(X_array, y)
        self.training_time = time.time() - start_time

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels."""
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_array)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities."""
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict_proba(X_array)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance as DataFrame."""
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

    def get_tree_depth(self) -> int:
        """Get actual tree depth."""
        return self.model.get_depth()

    def get_n_leaves(self) -> int:
        """Get number of leaf nodes."""
        return self.model.get_n_leaves()


class RandomForestWrapper:
    """
    Production wrapper for Random Forest classifier.

    Provides ensemble of decision trees with bagging, OOB scoring,
    and automatic hyperparameter tuning.

    Attributes:
        model (RandomForestClassifier): Fitted sklearn random forest
        best_params (Dict): Best hyperparameters from tuning
        feature_names (List[str]): Names of input features
        training_time (float): Time taken to train model
        oob_score (float): Out-of-bag score (if enabled)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 20,
        min_samples_leaf: int = 20,
        max_features: Union[str, float] = "sqrt",
        oob_score: bool = True,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize Random Forest wrapper.

        Args:
            n_estimators: Number of trees in forest
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples in leaf node
            max_features: Number of features for best split
            oob_score: Whether to use out-of-bag samples for scoring
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            oob_score=oob_score,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=n_jobs
        )
        self.best_params = None
        self.feature_names = None
        self.training_time = None
        self.oob_score_ = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> "RandomForestWrapper":
        """
        Fit random forest model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: Names of features

        Returns:
            self: Fitted wrapper instance
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            X_array = X

        start_time = time.time()
        self.model.fit(X_array, y)
        self.training_time = time.time() - start_time

        if self.model.oob_score:
            self.oob_score_ = self.model.oob_score_

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels."""
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_array)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities."""
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict_proba(X_array)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance as DataFrame."""
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)


class GradientBoostingWrapper:
    """
    Production wrapper for Gradient Boosting classifier.

    Provides sequential boosting with learning rate control,
    early stopping, and staged predictions.

    Attributes:
        model (GradientBoostingClassifier): Fitted sklearn gradient boosting
        best_params (Dict): Best hyperparameters from tuning
        feature_names (List[str]): Names of input features
        training_time (float): Time taken to train model
    """

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        min_samples_split: int = 100,
        min_samples_leaf: int = 50,
        subsample: float = 0.8,
        random_state: int = 42
    ):
        """
        Initialize Gradient Boosting wrapper.

        Args:
            n_estimators: Number of boosting stages
            learning_rate: Shrinkage parameter
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples in leaf node
            subsample: Fraction of samples for fitting base learners
            random_state: Random seed
        """
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=random_state
        )
        self.best_params = None
        self.feature_names = None
        self.training_time = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> "GradientBoostingWrapper":
        """
        Fit gradient boosting model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: Names of features

        Returns:
            self: Fitted wrapper instance
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            X_array = X

        start_time = time.time()
        self.model.fit(X_array, y)
        self.training_time = time.time() - start_time

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels."""
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_array)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities."""
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict_proba(X_array)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance as DataFrame."""
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

    def staged_predict(self, X: Union[pd.DataFrame, np.ndarray]):
        """
        Generator for predictions at each boosting iteration.

        Useful for analyzing learning curves and early stopping.

        Args:
            X: Feature matrix

        Yields:
            Predictions at each stage
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.staged_predict(X_array)


def train_with_cv(
    model_type: str,
    X: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    n_splits: int = 5,
    scoring: str = "f1",
    **model_params
) -> Tuple[Any, Dict]:
    """
    Train model with time-series cross-validation.

    Args:
        model_type: Type of model ("decision_tree", "random_forest", "gradient_boosting")
        X: Feature matrix
        y: Target vector
        n_splits: Number of CV splits
        scoring: Scoring metric for CV
        **model_params: Model-specific parameters

    Returns:
        model: Fitted model wrapper
        cv_results: Cross-validation results

    Example:
        >>> model, cv_results = train_with_cv("random_forest", X_train, y_train, n_estimators=100)
        >>> print(f"CV F1 score: {cv_results['mean_score']:.4f}")
    """
    # Create model based on type
    if model_type == "decision_tree":
        model = DecisionTreeWrapper(**model_params)
    elif model_type == "random_forest":
        model = RandomForestWrapper(**model_params)
    elif model_type == "gradient_boosting":
        model = GradientBoostingWrapper(**model_params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []

    for train_idx, val_idx in tscv.split(X):
        # Split data
        if isinstance(X, pd.DataFrame):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        else:
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]

        y_train_cv, y_val_cv = y[train_idx], y[val_idx]

        # Fit and evaluate
        model_cv = model.__class__(**model_params) if hasattr(model, '__class__') else model
        model_cv.fit(X_train_cv, y_train_cv)
        y_pred = model_cv.predict(X_val_cv)

        # Compute score
        if scoring == "f1":
            score = f1_score(y_val_cv, y_pred)
        elif scoring == "accuracy":
            score = accuracy_score(y_val_cv, y_pred)
        elif scoring == "roc_auc":
            y_proba = model_cv.predict_proba(X_val_cv)[:, 1]
            score = roc_auc_score(y_val_cv, y_proba)
        else:
            raise ValueError(f"Unknown scoring: {scoring}")

        cv_scores.append(score)

    # Train final model on full data
    model.fit(X, y)

    cv_results = {
        'mean_score': float(np.mean(cv_scores)),
        'std_score': float(np.std(cv_scores)),
        'scores': [float(s) for s in cv_scores],
        'n_splits': n_splits,
        'scoring': scoring
    }

    return model, cv_results


def optimize_hyperparameters(
    model_type: str,
    X: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    param_grid: Optional[Dict] = None,
    n_iter: int = 50,
    n_splits: int = 5,
    scoring: str = "f1",
    search_type: str = "randomized",
    n_jobs: int = -1,
    random_state: int = 42
) -> Tuple[Any, Dict]:
    """
    Optimize hyperparameters using grid or randomized search.

    Args:
        model_type: Type of model ("decision_tree", "random_forest", "gradient_boosting")
        X: Feature matrix
        y: Target vector
        param_grid: Parameter grid/distributions for search
        n_iter: Number of iterations for randomized search
        n_splits: Number of CV splits
        scoring: Scoring metric
        search_type: "grid" or "randomized"
        n_jobs: Number of parallel jobs
        random_state: Random seed

    Returns:
        best_model: Model with best parameters
        search_results: Search results including best params and scores

    Example:
        >>> param_grid = {'n_estimators': [100, 200], 'max_depth': [5, 10]}
        >>> model, results = optimize_hyperparameters("random_forest", X, y, param_grid)
        >>> print(f"Best params: {results['best_params']}")
    """
    # Default parameter grids
    if param_grid is None:
        if model_type == "decision_tree":
            param_grid = {
                'max_depth': [5, 6, 7, 8, 10],
                'min_samples_split': [20, 50, 100, 200],
                'min_samples_leaf': [10, 20, 50],
                'criterion': ['gini', 'entropy']
            }
        elif model_type == "random_forest":
            param_grid = {
                'n_estimators': [100, 150, 200, 300],
                'max_depth': [6, 8, 10, 12, 15],
                'min_samples_split': [20, 50, 100],
                'min_samples_leaf': [10, 20, 50],
                'max_features': ['sqrt', 'log2', 0.3, 0.5]
            }
        elif model_type == "gradient_boosting":
            param_grid = {
                'n_estimators': [100, 150, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'max_depth': [3, 4, 5, 6, 7, 8],
                'min_samples_split': [20, 50, 100],
                'min_samples_leaf': [10, 20, 50],
                'subsample': [0.8, 0.9, 1.0]
            }

    # Create base model
    if model_type == "decision_tree":
        base_model = DecisionTreeClassifier(random_state=random_state, class_weight="balanced")
    elif model_type == "random_forest":
        base_model = RandomForestClassifier(
            random_state=random_state,
            class_weight="balanced",
            oob_score=True,
            n_jobs=n_jobs
        )
    elif model_type == "gradient_boosting":
        base_model = GradientBoostingClassifier(random_state=random_state)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Perform search
    if search_type == "grid":
        search = GridSearchCV(
            base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=0
        )
    else:  # randomized
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=tscv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=0
        )

    # Convert X to array if needed
    X_array = X.values if isinstance(X, pd.DataFrame) else X

    # Fit search
    search.fit(X_array, y)

    # Wrap best model
    if model_type == "decision_tree":
        best_model = DecisionTreeWrapper(**search.best_params_)
    elif model_type == "random_forest":
        best_model = RandomForestWrapper(**search.best_params_)
    else:  # gradient_boosting
        best_model = GradientBoostingWrapper(**search.best_params_)

    # Fit on full data
    best_model.fit(X, y)
    best_model.best_params = search.best_params_

    search_results = {
        'best_params': search.best_params_,
        'best_score': float(search.best_score_),
        'cv_results': {
            'mean_test_score': search.cv_results_['mean_test_score'].tolist(),
            'std_test_score': search.cv_results_['std_test_score'].tolist(),
            'params': search.cv_results_['params']
        },
        'n_splits': n_splits,
        'scoring': scoring
    }

    return best_model, search_results


def save_model(
    model: Union[DecisionTreeWrapper, RandomForestWrapper, GradientBoostingWrapper],
    model_path: Path,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save tree model and metadata.

    Args:
        model: Fitted model wrapper
        model_path: Path to save directory
        metadata: Optional metadata dictionary
    """
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    # Save model
    with open(model_path / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save metadata
    meta = {
        'model_type': model.__class__.__name__,
        'feature_names': model.feature_names,
        'training_time': model.training_time,
        'best_params': model.best_params
    }

    if isinstance(model, RandomForestWrapper) and model.oob_score_ is not None:
        meta['oob_score'] = float(model.oob_score_)

    if metadata is not None:
        meta.update(metadata)

    with open(model_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_model(model_path: Path) -> Tuple[Any, Dict]:
    """
    Load tree model and metadata.

    Args:
        model_path: Path to model directory

    Returns:
        model: Loaded model wrapper
        metadata: Metadata dictionary
    """
    model_path = Path(model_path)

    # Load model
    with open(model_path / "model.pkl", "rb") as f:
        model = pickle.load(f)

    # Load metadata
    with open(model_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    return model, metadata
