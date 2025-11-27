import unittest

import numpy as np
import torch
from parameterized import parameterized

from orthogonal_dfa.module.monotonic import Monotonic1DFixedRange

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
