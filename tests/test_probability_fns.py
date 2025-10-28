import unittest

import numpy as np
import torch

from orthogonal_dfa.psams.psams import (
    conditional_cascade_log_probs,
    flip_log_probs,
    union_log_probs,
)


class TestTorchProbabilityFns(unittest.TestCase):
    def assertProbabilityOperation(self, fn, inp, expected_output, **kwargs):
        inp_tensor = torch.tensor(inp)
        output_tensor = fn(inp_tensor, **kwargs)
        expected_tensor = torch.tensor(expected_output)
        if torch.allclose(output_tensor, expected_tensor, atol=1e-5):
            return
        differences = torch.abs(output_tensor - expected_tensor)
        max_diff = torch.max(differences).item()
        self.fail(
            f"Max difference {max_diff} exceeds tolerance between output and expected output.\n"
            f"Output: {output_tensor}\nExpected: {expected_tensor}"
        )

    def test_flip_log_probs(self):
        # 1e-7 is the clipping threshold, so we use it here
        self.assertProbabilityOperation(
            flip_log_probs,
            np.log([[0.9, 0.5], [0.2, 0.8], [1e-7, 1.0]]),
            expected_output=np.log([[0.1, 0.5], [0.8, 0.2], [1.0, 1e-7]]),
        )

    def test_union_log_probs(self):
        self.assertProbabilityOperation(
            union_log_probs,
            np.log([[0.9, 0.5], [0.2, 0.8], [1e-7, 1.0]]),
            expected_output=np.log(
                [
                    1 - (1 - 0.9) * (1 - 0.2) * (1 - 1e-7),
                    1 - (1 - 0.5) * (1 - 0.8) * 1e-7,
                ]
            ),
            axis=0,
        )

    def test_conditional_cascade_log_probs(self):
        self.assertProbabilityOperation(
            conditional_cascade_log_probs,
            np.log(
                [
                    [0.9, 0.5],
                    [0.2, 0.8],
                    [1e-7, 1.0],
                ]
            ),
            expected_output=np.log(
                [
                    [0.9, 0.5],
                    [0.1 * 0.2, 0.5 * 0.8],
                    [0.1 * 0.8 * 1e-7, 0.5 * 0.2 * 1.0],
                    [0.1 * 0.8 * (1 - 1e-7), 0.5 * 0.2 * 1e-7],
                ]
            ),
            axis=0,
        )
