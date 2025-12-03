import unittest

import numpy as np
import torch
from torch import nn
from parameterized import parameterized

from orthogonal_dfa.utils.pdfa import pdfa_forward_fast, pdfa_forward_slow


class TestPDFAFast(unittest.TestCase):
    def run_pdfa(self, methods, seed):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        S = rng.choice([2, 3, 4, 5])
        C = rng.choice([2, 3, 4, 5])
        N = rng.choice([1, 5, 10, 20])
        L = rng.choice([10, 50, 100, 200])
        initial_state_probs = torch.randn(S, dtype=torch.float64).softmax(dim=0)
        transition_probs = torch.randn(S, C, S, dtype=torch.float64).softmax(dim=2)
        accepting_state_logprobs = torch.sigmoid(
            torch.randn(S, dtype=torch.float64)
        ).log()
        input_probs = torch.randn(N, L, C, dtype=torch.float64).softmax(dim=2)
        params = [
            initial_state_probs,
            transition_probs,
            accepting_state_logprobs,
            input_probs,
        ]
        params = [nn.Parameter(tensor) for tensor in params]
        rand_output = torch.randn(N)
        results = []
        for method in methods:
            for tensor in params:
                tensor.grad = None
            result = method(*params)
            loss = ((result - rand_output) ** 2).mean()
            loss.backward()
            results.append(
                {
                    "result": result.detach().cpu().numpy(),
                    "grads": [p.grad.detach().cpu().numpy() for p in params],
                }
            )
        return results

    def assertAllSame(self, methods, seed=0, tol=1e-5):
        results = self.run_pdfa(methods, seed)
        first_result = results[0]
        for other_result in results[1:]:
            np.testing.assert_allclose(
                first_result["result"],
                other_result["result"],
                atol=tol,
                err_msg="Results do not match",
            )
            for g1, g2 in zip(first_result["grads"], other_result["grads"]):
                np.testing.assert_allclose(
                    g1,
                    g2,
                    atol=tol,
                    err_msg="Gradients do not match",
                )

    def test_basic_sanity_check(self):
        self.assertAllSame([pdfa_forward_slow, pdfa_forward_slow], seed=0)

    @parameterized.expand([(i,) for i in range(100)])
    def test_slow_vs_fast(self, seed):

        self.assertAllSame([pdfa_forward_slow, pdfa_forward_fast], seed=seed)

    def test_speed(self):
        multiplier = 5
        import time

        timings = {}

        seeds = list(range(100))
        methods = [pdfa_forward_slow, pdfa_forward_fast]
        for method in methods:
            start_time = time.time()
            for seed in seeds:
                self.run_pdfa([method], seed)
            end_time = time.time()
            timings[method.__name__] = end_time - start_time
        print("Timings:", timings)
        speedup = timings["pdfa_forward_slow"] / timings["pdfa_forward_fast"]
        self.assertGreater(
            speedup,
            multiplier,
            f"Fast method is not at least {multiplier}x faster than slow method",
        )
