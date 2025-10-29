import unittest

import numpy as np
import torch

from orthogonal_dfa.oracle.evaluate import (
    multidimensional_confusion_from_proabilistic_results,
)


class TestConfusionProbabilistic(unittest.TestCase):
    def run_multidimensional_confusion(self, prob_model, prob_dfa_test, prob_dfa_prev):
        result = multidimensional_confusion_from_proabilistic_results(
            torch.tensor(np.log(prob_model)),
            torch.tensor(np.log(prob_dfa_test)),
            [torch.tensor(np.log(p)) for p in prob_dfa_prev],
        )
        return torch.exp(result).numpy()

    def assertExample(self, prob_model, prob_dfa_test, prob_dfa_prev, expected):
        result = self.run_multidimensional_confusion(
            prob_model, prob_dfa_test, prob_dfa_prev
        )
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)

    def test_basic_example(self):
        self.assertExample(
            # model
            [0.75, 0.5, 0.2],
            # dfa to test
            [[0.7, 0.4, 0.1], [0.3, 0.6, 0.9]],
            [],
            np.array(
                [
                    [
                        [
                            # model rejects, dfa rejects
                            0.25 * 0.3 + 0.5 * 0.6 + 0.8 * 0.9,
                            # model accepts, dfa rejects
                            0.75 * 0.3 + 0.5 * 0.6 + 0.2 * 0.9,
                        ],
                        [
                            # model rejects, dfa accepts
                            0.25 * 0.7 + 0.5 * 0.4 + 0.8 * 0.1,
                            # model accepts, dfa accepts
                            0.75 * 0.7 + 0.5 * 0.4 + 0.2 * 0.1,
                        ],
                    ],
                    [
                        [
                            # model rejects, dfa rejects
                            0.25 * 0.7 + 0.5 * 0.4 + 0.8 * 0.1,
                            # model accepts, dfa rejects
                            0.75 * 0.7 + 0.5 * 0.4 + 0.2 * 0.1,
                        ],
                        [
                            # model rejects, dfa accepts
                            0.25 * 0.3 + 0.5 * 0.6 + 0.9 * 0.8,
                            # model accepts, dfa accepts
                            0.75 * 0.3 + 0.5 * 0.6 + 0.2 * 0.9,
                        ],
                    ],
                ]
            )
            / 3,
        )

    def sample_logprobs(self, shape):
        return -torch.abs(torch.randn(shape))

    def test_with_previous_dfa(self):
        torch.random.manual_seed(0)
        num_samples = 3
        log_prob_model = self.sample_logprobs((num_samples,))
        log_prob_dfa_test = self.sample_logprobs((5, num_samples))
        log_prob_dfa_prev = [self.sample_logprobs((num_samples,)) for _ in range(4)]
        result = multidimensional_confusion_from_proabilistic_results(
            log_prob_model, log_prob_dfa_test, log_prob_dfa_prev
        )
        prob_model = torch.exp(log_prob_model)
        prob_dfa_test = torch.exp(log_prob_dfa_test)
        prob_dfa_prev = [torch.exp(p) for p in log_prob_dfa_prev]
        # 6 total, 1 for dfa to test, 4 for previous dfas, 1 for model
        self.assertEqual(tuple(result.shape), (5, 2, 2, 2, 2, 2, 2))
        pull_specific_prob = torch.exp(result[3, 0, 1, 0, 1, 0, 1]).item()
        self.assertAlmostEqual(
            pull_specific_prob,
            (
                (1 - prob_dfa_test[3])
                * prob_dfa_prev[0]
                * (1 - prob_dfa_prev[1])
                * prob_dfa_prev[2]
                * (1 - prob_dfa_prev[3])
                * prob_model
            )
            .mean()
            .item(),
        )
