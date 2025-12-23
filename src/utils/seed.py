"""
Random Seed Management for Reproducibility

This module provides utilities for setting random seeds across multiple libraries
to ensure reproducible results in machine learning experiments.

Supports:
- Python's built-in random module
- NumPy
- Scikit-learn (via numpy seed)
- Environment variables for additional control

Usage:
    >>> from src.utils.seed import seed_everything
    >>> seed_everything(42)
    >>> # All random operations are now reproducible
"""

import os
import random
import numpy as np
from typing import Optional, Any


def set_seed(seed: int) -> None:
    """
    Set random seed for Python's random module and NumPy.

    Parameters
    ----------
    seed : int
        Random seed value

    Examples
    --------
    >>> set_seed(42)
    >>> np.random.rand(3)
    array([0.37454012, 0.95071431, 0.73199394])
    """
    random.seed(seed)
    np.random.seed(seed)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """
    Set random seed for all random number generators.

    Comprehensive seed setting for reproducibility across:
    - Python's random module
    - NumPy
    - Environment variables (PYTHONHASHSEED)

    Parameters
    ----------
    seed : int
        Random seed value
    deterministic : bool, default=False
        If True, sets additional environment variables for deterministic behavior.
        Note: May impact performance.

    Examples
    --------
    >>> seed_everything(42)
    >>> # All random operations are now reproducible
    >>> np.random.rand(2)
    array([0.37454012, 0.95071431])
    >>> random.random()
    0.6394267984578837

    Notes
    -----
    For complete reproducibility in multi-threaded environments,
    set deterministic=True and limit parallelism:

    >>> seed_everything(42, deterministic=True)
    >>> os.environ['OMP_NUM_THREADS'] = '1'
    >>> os.environ['MKL_NUM_THREADS'] = '1'
    """
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PYTHONHASHSEED for hash reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # Additional environment variables for deterministic behavior
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Get a NumPy random number generator instance.

    Provides modern NumPy random number generation using np.random.Generator.
    This is the recommended approach for new code.

    Parameters
    ----------
    seed : int, optional
        Random seed value. If None, uses non-deterministic seeding.

    Returns
    -------
    rng : np.random.Generator
        Random number generator instance

    Examples
    --------
    >>> rng = get_rng(42)
    >>> rng.random(3)
    array([0.77395605, 0.43887844, 0.85859792])

    >>> # Use in train_test_split
    >>> from sklearn.model_selection import train_test_split
    >>> rng = get_rng(42)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.2, random_state=rng.integers(0, 2**31)
    ... )

    Notes
    -----
    NumPy's new random API (Generator) is preferred over legacy RandomState.
    See: https://numpy.org/doc/stable/reference/random/generator.html
    """
    if seed is not None:
        return np.random.default_rng(seed)
    else:
        return np.random.default_rng()


def create_random_state(seed: int) -> np.random.RandomState:
    """
    Create a NumPy RandomState instance (legacy API).

    Use this for compatibility with older code that requires RandomState.
    For new code, prefer get_rng() which returns a Generator.

    Parameters
    ----------
    seed : int
        Random seed value

    Returns
    -------
    random_state : np.random.RandomState
        Random state instance

    Examples
    --------
    >>> rs = create_random_state(42)
    >>> rs.rand(3)
    array([0.37454012, 0.95071431, 0.73199394])

    >>> # Use with scikit-learn
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> rs = create_random_state(42)
    >>> model = RandomForestClassifier(random_state=rs)
    """
    return np.random.RandomState(seed)


def get_random_int(low: int = 0, high: int = 2**31, seed: Optional[int] = None) -> int:
    """
    Generate a random integer suitable for use as a random seed.

    Useful for generating random seeds programmatically while maintaining
    reproducibility via the master seed.

    Parameters
    ----------
    low : int, default=0
        Lower bound (inclusive)
    high : int, default=2**31
        Upper bound (exclusive)
    seed : int, optional
        Random seed for the generator. If None, uses non-deterministic seeding.

    Returns
    -------
    random_int : int
        Random integer in range [low, high)

    Examples
    --------
    >>> seed = get_random_int(seed=42)
    >>> print(seed)
    1608637542

    >>> # Use to create child seeds
    >>> master_seed = 42
    >>> cv_seed = get_random_int(seed=master_seed)
    >>> model_seed = get_random_int(seed=master_seed + 1)
    """
    rng = get_rng(seed)
    return int(rng.integers(low, high))


def reset_random_state() -> None:
    """
    Reset random state to non-deterministic behavior.

    This function clears any previously set seeds and restores
    random number generators to non-deterministic operation.

    Examples
    --------
    >>> seed_everything(42)
    >>> np.random.rand()  # Deterministic
    0.3745401188473625
    >>> reset_random_state()
    >>> np.random.rand()  # Non-deterministic
    0.8327429174285727  # Different each time
    """
    # Reset Python random
    random.seed()

    # Reset NumPy random (uses current time or OS entropy)
    np.random.seed(None)

    # Clear environment variable
    if "PYTHONHASHSEED" in os.environ:
        del os.environ["PYTHONHASHSEED"]


def check_reproducibility(seed: int, n_trials: int = 3) -> bool:
    """
    Check if random operations are reproducible with given seed.

    Runs multiple trials with the same seed and verifies that
    results are identical.

    Parameters
    ----------
    seed : int
        Random seed to test
    n_trials : int, default=3
        Number of trials to run

    Returns
    -------
    is_reproducible : bool
        True if all trials produce identical results

    Examples
    --------
    >>> is_reproducible = check_reproducibility(42)
    >>> print(f"Reproducible: {is_reproducible}")
    Reproducible: True
    """
    results = []

    for _ in range(n_trials):
        seed_everything(seed)

        # Generate some random numbers
        trial_result = {
            "random": random.random(),
            "numpy": np.random.rand(5).tolist(),
            "choice": random.choice([1, 2, 3, 4, 5]),
        }
        results.append(trial_result)

    # Check if all trials are identical
    first_result = results[0]
    return all(result == first_result for result in results)


def generate_seeds(master_seed: int, n_seeds: int) -> list:
    """
    Generate a list of random seeds from a master seed.

    Useful for creating reproducible but different seeds for multiple
    experiments, cross-validation folds, or model ensembles.

    Parameters
    ----------
    master_seed : int
        Master seed for generating child seeds
    n_seeds : int
        Number of seeds to generate

    Returns
    -------
    seeds : list of int
        List of random seeds

    Examples
    --------
    >>> seeds = generate_seeds(master_seed=42, n_seeds=5)
    >>> print(seeds)
    [1608637542, 1156443393, 1276821039, 1510869654, 615428889]

    >>> # Use for cross-validation
    >>> master_seed = 42
    >>> cv_seeds = generate_seeds(master_seed, n_seeds=5)
    >>> for fold, seed in enumerate(cv_seeds):
    ...     print(f"Fold {fold}: seed={seed}")
    Fold 0: seed=1608637542
    Fold 1: seed=1156443393
    Fold 2: seed=1276821039
    Fold 3: seed=1510869654
    Fold 4: seed=615428889
    """
    rng = get_rng(master_seed)
    return [int(rng.integers(0, 2**31)) for _ in range(n_seeds)]


def make_reproducible_split(
    data_size: int, test_size: float = 0.2, seed: int = 42
) -> tuple:
    """
    Generate reproducible train/test split indices.

    Parameters
    ----------
    data_size : int
        Total number of samples
    test_size : float, default=0.2
        Proportion of data to use for testing
    seed : int, default=42
        Random seed

    Returns
    -------
    train_indices : np.ndarray
        Training set indices
    test_indices : np.ndarray
        Test set indices

    Examples
    --------
    >>> train_idx, test_idx = make_reproducible_split(100, test_size=0.2, seed=42)
    >>> len(train_idx), len(test_idx)
    (80, 20)
    """
    rng = get_rng(seed)
    indices = np.arange(data_size)
    rng.shuffle(indices)

    split_point = int(data_size * (1 - test_size))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    return train_indices, test_indices


class ReproducibleContext:
    """
    Context manager for temporary reproducible execution.

    Ensures that code within the context uses a specific seed,
    then restores the previous random state afterward.

    Examples
    --------
    >>> with ReproducibleContext(42):
    ...     print(np.random.rand())
    0.3745401188473625

    >>> with ReproducibleContext(42):
    ...     print(np.random.rand())
    0.3745401188473625  # Same as before

    >>> # Outside context, random state is independent
    >>> print(np.random.rand())
    0.8327429174285727  # Different value
    """

    def __init__(self, seed: int):
        """
        Initialize reproducible context.

        Parameters
        ----------
        seed : int
            Random seed to use within context
        """
        self.seed = seed
        self.random_state = None
        self.numpy_state = None

    def __enter__(self):
        """Save current state and set new seed."""
        # Save current states
        self.random_state = random.getstate()
        self.numpy_state = np.random.get_state()

        # Set new seed
        seed_everything(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous random state."""
        random.setstate(self.random_state)
        np.random.set_state(self.numpy_state)


def save_random_state() -> dict:
    """
    Save current random state for later restoration.

    Returns
    -------
    state : dict
        Dictionary containing random states for Python random and NumPy

    Examples
    --------
    >>> state = save_random_state()
    >>> # Do some random operations
    >>> restore_random_state(state)  # Restore to previous state
    """
    return {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
    }


def restore_random_state(state: dict) -> None:
    """
    Restore random state from previously saved state.

    Parameters
    ----------
    state : dict
        Dictionary containing random states (from save_random_state())

    Examples
    --------
    >>> seed_everything(42)
    >>> val1 = np.random.rand()
    >>> state = save_random_state()
    >>> val2 = np.random.rand()
    >>> restore_random_state(state)
    >>> val3 = np.random.rand()
    >>> val2 == val3  # True - same random value
    True
    """
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])


if __name__ == "__main__":
    print("Random Seed Management Module")
    print("=" * 70)
    print("\nKey functions:")
    print("  - set_seed(): Set seed for random and numpy")
    print("  - seed_everything(): Comprehensive seed setting")
    print("  - get_rng(): Get NumPy random number generator")
    print("  - generate_seeds(): Generate multiple reproducible seeds")
    print("  - check_reproducibility(): Verify reproducibility")
    print("  - ReproducibleContext: Context manager for temporary seeding")
    print("\nExample usage:")
    print("  >>> from src.utils.seed import seed_everything")
    print("  >>> seed_everything(42)")
    print("  >>> # All random operations are now reproducible")
    print("\nSee function docstrings for detailed usage examples.")
