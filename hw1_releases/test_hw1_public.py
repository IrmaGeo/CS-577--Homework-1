"""Public smoke tests for CS 577 Homework 1.

Run from the HW1 directory with either command:

    python test_hw1_public.py
    python -m unittest -v test_hw1_public.py

These tests check function interfaces and representative correctness cases.
Passing them is useful evidence, but it does not guarantee full credit: the
grader may use additional inputs and will also assess derivations, explanations,
plots/tables, code quality, and the executed notebook.
"""

import random
import unittest

import numpy as np

import hw1_utils as h


ATOL = 1e-8
RTOL = 1e-7


class InterfaceAndSeedTests(unittest.TestCase):
    """Basic API and reproducibility checks."""

    def test_01_required_names_exist(self):
        required = [
            "set_all_seeds",
            "affine_numpy",
            "standardize_columns",
            "pairwise_squared_distances",
            "LinearRegressor",
            "train_tiny_regressor",
            "mse_l2_objective_numpy",
            "mse_l2_grad_numpy",
            "finite_difference_gradient",
        ]
        missing = [name for name in required if not hasattr(h, name)]
        self.assertEqual(
            missing,
            [],
            msg=f"Missing required name(s) in hw1_utils.py: {missing}",
        )

    def test_02_seed_reproducibility(self):
        h.set_all_seeds(577)
        python_1 = [random.random() for _ in range(4)]
        numpy_1 = np.random.standard_normal(4)
        torch_1 = h.torch.randn(4) if h.torch is not None else None

        h.set_all_seeds(577)
        python_2 = [random.random() for _ in range(4)]
        numpy_2 = np.random.standard_normal(4)
        torch_2 = h.torch.randn(4) if h.torch is not None else None

        self.assertEqual(
            python_1,
            python_2,
            msg="Python's random stream was not reproduced after resetting the seed.",
        )
        np.testing.assert_array_equal(
            numpy_1,
            numpy_2,
            err_msg="NumPy's random stream was not reproduced after resetting the seed.",
        )
        if h.torch is not None:
            self.assertTrue(
                h.torch.equal(torch_1, torch_2),
                msg="PyTorch's random stream was not reproduced after resetting the seed.",
            )


class NumPyTests(unittest.TestCase):
    """Representative NumPy, objective, and gradient checks."""

    def test_03_affine_values_shape_and_no_mutation(self):
        X = np.array([[1.0, -2.0], [0.5, 3.0], [-1.0, 0.0]])
        W = np.array([[2.0, -1.0, 0.5], [0.25, 2.0, -3.0]])
        b = np.array([0.5, -0.5, 1.0])
        X_before, W_before, b_before = X.copy(), W.copy(), b.copy()

        result = h.affine_numpy(X, W, b)
        expected = X @ W + b

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(
            result.shape,
            (3, 3),
            msg="affine_numpy should return shape (B, K).",
        )
        np.testing.assert_allclose(
            result,
            expected,
            atol=ATOL,
            rtol=RTOL,
            err_msg="affine_numpy returned incorrect values.",
        )
        np.testing.assert_array_equal(X, X_before, err_msg="Do not modify X in place.")
        np.testing.assert_array_equal(W, W_before, err_msg="Do not modify W in place.")
        np.testing.assert_array_equal(b, b_before, err_msg="Do not modify b in place.")

    def test_04_standardization_including_constant_column(self):
        X = np.array(
            [
                [1.0, 7.0, -2.0],
                [2.0, 7.0, 0.0],
                [4.0, 7.0, 2.0],
                [5.0, 7.0, 4.0],
            ]
        )
        X_before = X.copy()

        result = h.standardize_columns(X)
        self.assertIsInstance(result, tuple, msg="Return exactly (Z, mean, std).")
        self.assertEqual(len(result), 3, msg="Return exactly (Z, mean, std).")
        Z, mean, std = result

        self.assertEqual(Z.shape, X.shape)
        self.assertEqual(mean.shape, (X.shape[1],))
        self.assertEqual(std.shape, (X.shape[1],))
        np.testing.assert_allclose(mean, X.mean(axis=0), atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(std, X.std(axis=0, ddof=0), atol=ATOL, rtol=RTOL)
        self.assertTrue(np.all(np.isfinite(Z)), msg="Z must not contain inf or NaN.")

        nonconstant = std > 0
        np.testing.assert_allclose(Z[:, nonconstant].mean(axis=0), 0.0, atol=ATOL)
        np.testing.assert_allclose(
            Z[:, nonconstant].std(axis=0, ddof=0), 1.0, atol=ATOL
        )
        np.testing.assert_allclose(
            Z[:, ~nonconstant],
            0.0,
            atol=ATOL,
            err_msg="A constant column should standardize to zeros.",
        )
        np.testing.assert_array_equal(X, X_before, err_msg="Do not modify X in place.")

    def test_05_pairwise_distances_shape_values_and_no_mutation(self):
        A = np.array([[0.5, -1.0, 2.0], [2.0, 0.0, -0.5]])
        B = np.array(
            [[1.0, 1.0, 1.0], [-2.0, 0.5, 3.0], [0.0, -1.0, 0.0]]
        )
        A_before, B_before = A.copy(), B.copy()

        result = h.pairwise_squared_distances(A, B)
        expected = np.array(
            [[np.sum((a - b) ** 2) for b in B] for a in A], dtype=float
        )

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(
            result.shape,
            (2, 3),
            msg="pairwise_squared_distances should return shape (M, N).",
        )
        self.assertTrue(np.all(result >= -ATOL), msg="Squared distances must be nonnegative.")
        np.testing.assert_allclose(
            result,
            expected,
            atol=ATOL,
            rtol=RTOL,
            err_msg="Pairwise squared distances are incorrect.",
        )
        np.testing.assert_array_equal(A, A_before, err_msg="Do not modify A in place.")
        np.testing.assert_array_equal(B, B_before, err_msg="Do not modify B in place.")

    def test_06_objective_and_analytic_gradient(self):
        X = np.array([[1.0, 2.0], [-1.0, 0.5], [0.25, -2.0], [2.0, 1.0]])
        y = np.array([0.5, -1.0, 2.0, 1.5])
        w = np.array([0.3, -0.2])
        lam = 0.07
        X_before, y_before, w_before = X.copy(), y.copy(), w.copy()

        objective = h.mse_l2_objective_numpy(X, y, w, lam)
        gradient = h.mse_l2_grad_numpy(X, y, w, lam)

        residual = X @ w - y
        expected_objective = np.mean(residual**2) + 0.5 * lam * np.sum(w**2)
        expected_gradient = (2.0 / X.shape[0]) * X.T @ residual + lam * w

        self.assertTrue(
            np.isscalar(objective),
            msg="mse_l2_objective_numpy should return one scalar value.",
        )
        self.assertEqual(gradient.shape, w.shape)
        self.assertAlmostEqual(float(objective), float(expected_objective), places=10)
        np.testing.assert_allclose(
            gradient,
            expected_gradient,
            atol=ATOL,
            rtol=RTOL,
            err_msg="The analytic gradient is incorrect.",
        )
        np.testing.assert_array_equal(X, X_before, err_msg="Do not modify X in place.")
        np.testing.assert_array_equal(y, y_before, err_msg="Do not modify y in place.")
        np.testing.assert_array_equal(w, w_before, err_msg="Do not modify w in place.")

    def test_07_centered_finite_difference_and_no_mutation(self):
        Q = np.array([[3.0, 0.5], [0.5, 2.0]])
        c = np.array([-1.0, 0.25])
        objective = lambda v: 0.5 * v @ Q @ v + c @ v
        w = np.array([0.4, -0.7])
        w_before = w.copy()

        result = h.finite_difference_gradient(objective, w, epsilon=1e-5)
        expected = Q @ w + c

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, w.shape)
        np.testing.assert_allclose(
            result,
            expected,
            atol=1e-7,
            rtol=1e-7,
            err_msg="Centered finite-difference gradient is incorrect.",
        )
        np.testing.assert_array_equal(
            w,
            w_before,
            err_msg="finite_difference_gradient must leave w unchanged.",
        )


@unittest.skipIf(h.torch is None, "PyTorch is not installed in this environment")
class PyTorchTests(unittest.TestCase):
    """Lightweight model and training-loop checks."""

    def test_08_model_architecture_and_forward_shape(self):
        model = h.LinearRegressor()
        self.assertIsInstance(model, h.nn.Module)

        linear_layers = [m for m in model.modules() if isinstance(m, h.nn.Linear)]
        self.assertEqual(
            len(linear_layers),
            1,
            msg="LinearRegressor should contain exactly one nn.Linear layer.",
        )
        self.assertEqual(linear_layers[0].in_features, 1)
        self.assertEqual(linear_layers[0].out_features, 1)

        x = h.torch.zeros(7, 1)
        output = model(x)
        self.assertEqual(
            tuple(output.shape),
            (7, 1),
            msg="LinearRegressor.forward should map (B, 1) to (B, 1).",
        )

    def test_09_training_loop_returns_losses_and_updates_model(self):
        h.set_all_seeds(577)
        model = h.LinearRegressor()
        with h.torch.no_grad():
            model.linear.weight.zero_()
            model.linear.bias.zero_()

        x = h.torch.linspace(-1.0, 1.0, 9).reshape(-1, 1)
        y = 1.5 * x + 0.25
        weight_before = model.linear.weight.detach().clone()

        losses = h.train_tiny_regressor(model, x, y, steps=8, lr=0.1)

        self.assertEqual(
            len(losses),
            8,
            msg="Return one scalar loss for each requested training step.",
        )
        self.assertTrue(
            all(np.isscalar(value) for value in losses),
            msg="Each returned loss should be a Python/NumPy scalar, not a tensor.",
        )
        self.assertTrue(
            np.all(np.isfinite(np.asarray(losses, dtype=float))),
            msg="Training losses must be finite.",
        )
        self.assertLess(
            losses[-1],
            losses[0],
            msg="Loss should decrease on this small deterministic problem.",
        )
        self.assertFalse(
            h.torch.equal(weight_before, model.linear.weight.detach()),
            msg="Training should update the model parameters.",
        )


if __name__ == "__main__":
    print("CS 577 HW1 public tests")
    print("Passing these tests does not guarantee full credit.\n")
    unittest.main(verbosity=2)
