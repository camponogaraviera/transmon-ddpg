import numpy as np


def min_max_norm_env(
    state: np.ndarray, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    """
    Normalize state based on environment bounds to [0, 1].

    Args:
        state: Input array to normalize.
        low: Lower bounds of the environment.
        high: Upper bounds of the environment.

    Returns:
        The normalized array with values scaled to [0, 1].
    """

    if np.isinf(low).any():
        return state

    return np.clip((state - low) / (high - low), 0.0, 1.0)
