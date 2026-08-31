"""Starter implementations for CS 577 Homework 1.

Replace each NotImplementedError with your own implementation. Do not change
function names, arguments, or return-value order because the public and hidden
tests use this interface.
"""

import random
from typing import Tuple

import numpy as np
import torch

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
    safe_std = np.where(x_std == 0, 1.0, x_std)
    Z = (X - x_mean) / safe_std
    Z[:, x_std == 0] = 0.0
    return Z, x_mean, x_std


def pairwise_squared_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return all pairwise squared Euclidean distances, shape (M, N)."""    
    return ((A[:,None,:]-B[None,:, :])**2).sum(-1)


if nn is not None:

    class LinearRegressor(nn.Module):
        """One-input, one-output linear regression model."""

        def __init__(self) -> None:
            super().__init__()
            self.linear=nn.Linear(1,1)

        def forward(self, x):
            return self.linear(x)

else:

    class LinearRegressor:  # pragma: no cover - used only when torch is absent.
        def __init__(self) -> None:
            raise ImportError("PyTorch is required for LinearRegressor")


def train_tiny_regressor(model, x_train, y_train, steps: int = 100, lr: float = 0.1):
    """Train with full-batch SGD and return a list of scalar losses."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    model.train()
    for _i in range(steps):
        optimizer.zero_grad()
        predictions=model(x_train)
        loss=criterion(predictions,y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

def mse_l2_objective_numpy(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float
) -> float:
    """Mean squared error plus 0.5 * lam * ||w||_2^2."""
    residual = X @ w - y
    mse = np.mean(residual ** 2)
    l2_reg = 0.5 * lam * np.sum(w ** 2)
    return float(mse + l2_reg)


def mse_l2_grad_numpy(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float
) -> np.ndarray:
    """Analytic gradient of mse_l2_objective_numpy."""
    N = X.shape[0]
    residual = X @ w - y
    grad_mse = (2.0 / N) * (X.T @ residual)
    grad_l2 = lam * w
    return grad_mse + grad_l2


def finite_difference_gradient(
    objective, w: np.ndarray, epsilon: float = 1e-5
) -> np.ndarray:
    """Centered finite-difference gradient of a scalar objective(objective(w))."""
    grad = np.zeros_like(w, dtype=np.float64)
    
    w_base = w.astype(np.float64, copy=True)

    for i in range(len(w_base)):
        w_plus = w_base.copy()
        w_minus = w_base.copy()

        w_plus[i] += epsilon
        w_minus[i] -= epsilon

        grad[i] = (objective(w_plus) - objective(w_minus)) / (2.0 * epsilon)

    return grad

