import unittest

import numpy as np
import torch
from parameterized import parameterized

from orthogonal_dfa.module.monotonic import Monotonic1DFixedRange, Monotonic2DFixedRange

eps = 1e-3


class TestTorchProbabilityFns(unittest.TestCase):
    def checkMeetsReqs(self, monotonic: Monotonic1DFixedRange, rng):
        self.assertAlmostEqual(
            monotonic(
                torch.tensor(-monotonic.input_range),
            ).item(),
            -monotonic.input_range,
            delta=eps,
        )
        self.assertAlmostEqual(
            monotonic(
                torch.tensor(monotonic.input_range),
            ).item(),
            +monotonic.input_range,
            delta=eps,
        )
        for _ in range(100):
            x1, x2 = rng.standard_normal(2) * monotonic.input_range / 2
            x1, x2 = sorted([x1, x2])
            if x2 - x1 < eps:
                continue
            y1 = monotonic(torch.tensor(x1)).item()
            y2 = monotonic(torch.tensor(x2)).item()
            self.assertGreater(y2, y1, "Monotonicity requirement violated")

        # check breaks work correctly
        breaks = torch.linspace(
            -monotonic.input_range,
            monotonic.input_range,
            steps=monotonic.num_input_breaks,
        )
        slopes = monotonic.slopes
        values_at_breaks = monotonic.values_at_breaks
        computed_values = monotonic(breaks)
        # pylint: disable=consider-using-enumerate
        for i in range(len(breaks)):
            self.assertAlmostEqual(
                computed_values[i].item(),
                values_at_breaks[i].item(),
                msg=f"Value at break {i} incorrect",
                delta=eps,
            )
            derivative_left = (
                monotonic(breaks[i] - eps) - monotonic(breaks[i] - 2 * eps)
            ).item() / eps
            derivative_right = (
                monotonic(breaks[i] + 2 * eps) - monotonic(breaks[i] + eps)
            ).item() / eps
            self.assertAlmostEqual(
                slopes[i].item(),
                derivative_left,
                msg=f"Slope at break {i} incorrect (left derivative)",
                delta=eps,
            )
            self.assertAlmostEqual(
                slopes[i + 1].item(),
                derivative_right,
                msg=f"Slope at break {i} incorrect (right derivative)",
                delta=eps,
            )

    @parameterized.expand([(i,) for i in range(100)])
    def test_monotonic_basic(self, i):
        torch.manual_seed(i)
        rng = np.random.default_rng(i)
        monotonic = Monotonic1DFixedRange(
            input_range=2.0, num_input_breaks=rng.choice([5, 10, 15, 20])
        )
        self.checkMeetsReqs(monotonic, rng)


class TestMonotonic2D(unittest.TestCase):
    def checkMeetsReqs(self, monotonic: Monotonic2DFixedRange, rng):
        # Check boundary values at corners
        self.assertAlmostEqual(
            monotonic(
                torch.tensor(-monotonic.input_range),
                torch.tensor(-monotonic.input_range),
            ).item(),
            -monotonic.input_range,
            delta=eps,
            msg="Bottom-left corner value incorrect",
        )
        self.assertAlmostEqual(
            monotonic(
                torch.tensor(monotonic.input_range),
                torch.tensor(monotonic.input_range),
            ).item(),
            monotonic.input_range,
            delta=eps,
            msg="Top-right corner value incorrect",
        )
        # For the other corners, values depend on the step function, but should be monotonic
        # Check that bottom-right is >= bottom-left (monotonic in x)
        bottom_left = monotonic(
            torch.tensor(-monotonic.input_range),
            torch.tensor(-monotonic.input_range),
        ).item()
        bottom_right = monotonic(
            torch.tensor(monotonic.input_range),
            torch.tensor(-monotonic.input_range),
        ).item()
        self.assertGreaterEqual(
            bottom_right,
            bottom_left,
            msg="Bottom-right should be >= bottom-left (monotonic in x)",
        )

        # Check that top-left is >= bottom-left (monotonic in y)
        top_left = monotonic(
            torch.tensor(-monotonic.input_range),
            torch.tensor(monotonic.input_range),
        ).item()
        self.assertGreaterEqual(
            top_left,
            bottom_left,
            msg="Top-left should be >= bottom-left (monotonic in y)",
        )

        # Check monotonicity in x direction (fix y, vary x)
        for _ in range(50):
            y_fixed = rng.standard_normal() * monotonic.input_range / 2
            # Clamp to valid range to avoid boundary issues
            y_fixed = np.clip(y_fixed, -monotonic.input_range, monotonic.input_range)
            x1, x2 = rng.standard_normal(2) * monotonic.input_range / 2
            x1, x2 = sorted([x1, x2])
            # Clamp to valid range
            x1 = np.clip(x1, -monotonic.input_range, monotonic.input_range)
            x2 = np.clip(x2, -monotonic.input_range, monotonic.input_range)
            if abs(x2 - x1) < eps:
                continue
            y1 = monotonic(torch.tensor(x1), torch.tensor(y_fixed)).item()
            y2 = monotonic(torch.tensor(x2), torch.tensor(y_fixed)).item()
            # Allow small numerical errors
            self.assertGreaterEqual(
                y2,
                y1 - eps,
                f"Monotonicity in x violated: F({x1}, {y_fixed}) = {y1} > F({x2}, {y_fixed}) = {y2}",
            )

        # Check monotonicity in y direction (fix x, vary y)
        for _ in range(50):
            x_fixed = rng.standard_normal() * monotonic.input_range / 2
            # Clamp to valid range
            x_fixed = np.clip(x_fixed, -monotonic.input_range, monotonic.input_range)
            y1, y2 = rng.standard_normal(2) * monotonic.input_range / 2
            y1, y2 = sorted([y1, y2])
            # Clamp to valid range
            y1 = np.clip(y1, -monotonic.input_range, monotonic.input_range)
            y2 = np.clip(y2, -monotonic.input_range, monotonic.input_range)
            if abs(y2 - y1) < eps:
                continue
            z1 = monotonic(torch.tensor(x_fixed), torch.tensor(y1)).item()
            z2 = monotonic(torch.tensor(x_fixed), torch.tensor(y2)).item()
            # Allow small numerical errors
            self.assertGreaterEqual(
                z2,
                z1 - eps,
                f"Monotonicity in y violated: F({x_fixed}, {y1}) = {z1} > F({x_fixed}, {y2}) = {z2}",
            )

        # Check values at grid break points
        breaks = torch.linspace(
            -monotonic.input_range,
            monotonic.input_range,
            steps=monotonic.num_input_breaks,
        )
        cum_int = monotonic.cumulative_integral
        # Check that computed values match cumulative integral at grid points
        for i in range(len(breaks)):
            for j in range(len(breaks)):
                computed = monotonic(breaks[i], breaks[j]).item()
                expected = cum_int[j, i].item()
                self.assertAlmostEqual(
                    computed,
                    expected,
                    msg=f"Value at grid point ({i}, {j}) incorrect",
                    delta=eps,
                )
                eps_tiny = 1e-6
                # check continuity
                dx, dy = rng.standard_normal(2) * eps_tiny * monotonic.input_range / 2
                perturbed = monotonic(breaks[i] + dx, breaks[j] + dy).item()
                self.assertAlmostEqual(
                    perturbed,
                    computed,
                    msg=f"Continuity at grid point ({i}, {j}) violated",
                    delta=(dx**2 + dy**2) ** 0.5 * 10,
                )

        # Note: Partial derivative continuity is not guaranteed for piecewise bilinear functions
        # The double integral of a step function gives a piecewise bilinear function where
        # partial derivatives may have discontinuities at cell boundaries.
        # We skip this test as it's not a requirement for the monotonic function.

    @parameterized.expand([(i,) for i in range(50)])
    def test_monotonic_2d_basic(self, i):
        torch.manual_seed(i)
        rng = np.random.default_rng(i)
        monotonic = Monotonic2DFixedRange(
            input_range=2.0, num_input_breaks=rng.choice([5, 10, 15, 20])
        )
        self.checkMeetsReqs(monotonic, rng)
