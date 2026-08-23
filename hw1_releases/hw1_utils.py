"""Starter implementations for CS 577 Homework 1.

Replace each NotImplementedError with your own implementation. Do not change
function names, arguments, or return-value order because the public and hidden
tests use this interface.
"""

import random
from typing import Tuple

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # Allows NumPy-only public tests to import this file.
    torch = None
    nn = None


def set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and all CUDA devices when available."""
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def affine_numpy(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return X @ W + b without a Python loop."""
    return X @ W + b

def standardize_columns(
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Columnwise population standardization with safe constant columns."""
    x_mean=np.mean(X, axis=0)
    x_std=np.std(X, axis=0, ddof=0)
    Z=(X-x_mean)/x_std
    return Z, x_mean, x_std


def pairwise_squared_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return all pairwise squared Euclidean distances, shape (M, N)."""
    D2=(A.reshape(A.shape[0],1,A.shape[1])-B.reshape(1, B.shape[0], B.shape[1]))**2
    D2=D2.sum(-1)
    return D2


if nn is not None:

    class LinearRegressor(nn.Module):
        """One-input, one-output linear regression model."""

        def __init__(self) -> None:
            super().__init__()
            raise NotImplementedError

        def forward(self, x):
            raise NotImplementedError

else:

    class LinearRegressor:  # pragma: no cover - used only when torch is absent.
        def __init__(self) -> None:
            raise ImportError("PyTorch is required for LinearRegressor")


def train_tiny_regressor(model, x_train, y_train, steps: int = 100, lr: float = 0.1):
    """Train with full-batch SGD and return a list of scalar losses."""
    raise NotImplementedError


def mse_l2_objective_numpy(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float
) -> float:
    """Mean squared error plus 0.5 * lam * ||w||_2^2."""
    raise NotImplementedError


def mse_l2_grad_numpy(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float
) -> np.ndarray:
    """Analytic gradient of mse_l2_objective_numpy."""
    raise NotImplementedError


def finite_difference_gradient(
    objective, w: np.ndarray, epsilon: float = 1e-5
) -> np.ndarray:
    """Centered finite-difference gradient of a scalar objective(objective(w))."""
    raise NotImplementedError

