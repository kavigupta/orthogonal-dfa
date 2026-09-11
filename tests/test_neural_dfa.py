import unittest

import numpy as np
import torch

from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliParityOracle
from orthogonal_dfa.l_star.neural.extract import accept_threshold
from orthogonal_dfa.l_star.neural.model import NeuralDFA
from orthogonal_dfa.l_star.neural.objective import (
    TransitionStatistics,
    batch_transition_statistics,
    external_objective,
    internal_information,
    internal_objective,
)
from orthogonal_dfa.l_star.neural.train import NeuralConfig, train_neural_dfa
from orthogonal_dfa.l_star.structures import SymmetricBernoulli
from tests.test_lstar import assertion_allowed_error, evaluate_accuracy


def one_hot(idx, num_classes):
    return (idx.unsqueeze(-1) == torch.arange(num_classes)).float()


def prefix_parity(x):
    """``(B, L + 1)`` parity of each prefix of ``x`` -- a genuine 2-state congruence."""
    out = torch.zeros(x.shape[0], x.shape[1] + 1, dtype=torch.long)
    out[:, 1:] = torch.cumsum(x, dim=1) % 2
    return out


class TestInternalObjective(unittest.TestCase):
    """J_internal must be 1 exactly for a hard deterministic assignment and strictly
    less for anything that hedges, since that is what makes it a tight relaxation."""

    def _score(self, log_m, x, alphabet_size):
        stats = TransitionStatistics(alphabet_size, log_m.shape[-1])
        t_batch, n_batch = batch_transition_statistics(log_m, x, alphabet_size)
        stats.update(t_batch, n_batch)
        return float(
            internal_objective(t_batch, stats, temperature=0.0, min_support=0.0)
        )

    def test_deterministic_assignment_scores_one(self):
        x = torch.tensor([[0, 1, 1, 0, 1], [1, 1, 0, 1, 0]])
        log_m = one_hot(prefix_parity(x), 2).clamp_min(1e-9).log()
        self.assertAlmostEqual(self._score(log_m, x, 2), 1.0, places=5)

    def test_hedging_between_duplicate_states_is_penalised(self):
        # Same partition, but mass split 50/50 over two copies of each state.
        x = torch.tensor([[0, 1, 1, 0, 1], [1, 1, 0, 1, 0]])
        parity = prefix_parity(x)
        m = torch.zeros(*parity.shape, 4)
        m.scatter_(2, (2 * parity).unsqueeze(-1), 0.5)
        m.scatter_(2, (2 * parity + 1).unsqueeze(-1), 0.5)
        self.assertAlmostEqual(
            self._score(m.clamp_min(1e-9).log(), x, 2), 0.5, places=5
        )

    def test_total_collapse_also_scores_one(self):
        # The reason J_external is required: collapse is a global maximum of J_internal.
        x = torch.tensor([[0, 1, 1, 0, 1]])
        log_m = torch.log(torch.tensor([[[1.0, 1e-9]] * 6]))
        self.assertAlmostEqual(self._score(log_m, x, 2), 1.0, places=5)


class TestInternalInformation(unittest.TestCase):
    """The point of the information form: collapse scores 0 rather than 1, so merging is
    not a descent direction, and a k-state deterministic partition scores log k, so each
    extra distinction is rewarded on its own."""

    def _score(self, log_m, x, alphabet_size):
        t_batch, _ = batch_transition_statistics(log_m, x, alphabet_size)
        return float(internal_information(t_batch))

    def test_collapse_scores_zero(self):
        x = torch.tensor([[0, 1, 1, 0, 1]])
        log_m = torch.log(torch.tensor([[[1.0, 1e-9]] * 6]))
        self.assertAlmostEqual(self._score(log_m, x, 2), 0.0, places=4)

    def test_balanced_two_state_congruence_scores_log_two(self):
        x = torch.tensor([[0, 1, 1, 0, 1], [1, 1, 0, 1, 0], [1, 0, 1, 0, 1]])
        log_m = one_hot(prefix_parity(x), 2).clamp_min(1e-9).log()
        self.assertAlmostEqual(
            self._score(log_m, x, 2), float(torch.tensor(2.0).log()), places=2
        )

    def test_refinement_scores_higher_than_collapse(self):
        # The gradient signal the max form lacks: a finer deterministic partition wins.
        x = torch.tensor([[0, 1, 1, 0, 1], [1, 1, 0, 1, 0], [1, 0, 1, 0, 1]])
        coarse = one_hot(prefix_parity(x), 4).clamp_min(1e-9).log()
        positions = torch.arange(x.shape[1] + 1).expand(x.shape[0], -1)
        # parity refined by length mod 2 -> still deterministic, but 4 states.
        fine = one_hot(2 * prefix_parity(x) + positions % 2, 4).clamp_min(1e-9).log()
        self.assertGreater(self._score(fine, x, 2), self._score(coarse, x, 2))


class TestExternalObjective(unittest.TestCase):
    def test_matches_plain_weighted_bce(self):
        torch.manual_seed(0)
        probs = torch.rand(3, 6, 1) * 0.8 + 0.1
        labels = (torch.rand(3, 6) > 0.5).float()
        mask = torch.rand(3, 6)
        p = probs[:, :, 0]
        expected = (
            (labels * p.log() + (1 - labels) * (1 - p).log()) * mask
        ).sum() / mask.sum()
        self.assertAlmostEqual(
            float(external_objective(probs, labels, mask)), float(expected), places=5
        )

    def test_lag_k_reads_position_t_plus_k(self):
        # Constant probs, so only target alignment can matter. Lag 1 must see label[t + 1],
        # and positions running past the end must carry no weight: 4 lag-0 targets plus 3
        # in-range lag-1 targets = 7 terms, each log(0.5).
        probs = torch.full((1, 4, 2), 0.5)
        labels = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        mask = torch.ones(1, 4)
        self.assertAlmostEqual(
            float(external_objective(probs, labels, mask)),
            float(torch.tensor(0.5).log()),
            places=5,
        )
        weights_used = torch.cat([mask, torch.zeros(1, 1)], 1).unfold(1, 2, 1).sum()
        self.assertEqual(int(weights_used), 7)

    def test_splitting_a_state_can_change_the_objective(self):
        """The whole point: a per-state scalar accept rate is invariant under refinement,
        so it gives no gradient toward it. Conditioning on the continuation must not be.
        """
        torch.manual_seed(0)
        model = NeuralDFA(2, 4, hidden_size=16, num_lags=3, suffix_dim=8)
        x = torch.tensor([[0, 1, 1, 0, 1], [1, 0, 1, 0, 0]])
        merged = one_hot(torch.zeros(2, 6, dtype=torch.long), 4).clamp_min(1e-9).log()
        positions = torch.arange(6).expand(2, -1)
        split = one_hot((positions % 2), 4).clamp_min(1e-9).log()
        labels = torch.tensor([[1.0, 0, 1, 0, 1, 0], [0.0, 1, 0, 1, 0, 1]])
        mask = torch.ones(2, 6)
        a = external_objective(model.continuation_accept_probs(x, merged), labels, mask)
        b = external_objective(model.continuation_accept_probs(x, split), labels, mask)
        self.assertNotAlmostEqual(float(a), float(b), places=4)


class TestAcceptThreshold(unittest.TestCase):
    def test_finds_asymmetric_boundary(self):
        # Accept rates cluster at the noise rates, not at 0/1, and need not straddle 0.5.
        probs = np.array([0.10, 0.11, 0.40, 0.39, 0.41])
        weights = np.ones(5)
        self.assertAlmostEqual(accept_threshold(probs, weights), 0.2525, places=3)

    def test_single_cluster_falls_back(self):
        probs = np.array([0.4, 0.4, 0.4])
        self.assertEqual(accept_threshold(probs, np.ones(3)), 0.5)

    def test_zero_weight_states_ignored(self):
        # Unused states carry junk accept probabilities; they must not move the split.
        probs = np.array([0.10, 0.40, 0.99])
        self.assertAlmostEqual(
            accept_threshold(probs, np.array([1.0, 1.0, 0.0])), 0.25, places=3
        )


class TestModel(unittest.TestCase):
    def test_empty_prefix_matches_position_zero(self):
        torch.manual_seed(0)
        model = NeuralDFA(2, 6, hidden_size=16)
        x = torch.tensor([[0, 1, 1], [1, 0, 1]])
        self.assertTrue(
            torch.allclose(
                model.state_log_probs(x)[:, 0],
                model.empty_prefix_log_probs().expand(2, -1),
                atol=1e-6,
            )
        )

    def test_accept_rates_not_initialised_tied(self):
        # Tied accept rates make p constant in m and zero out m's gradient entirely.
        torch.manual_seed(0)
        model = NeuralDFA(2, 8, hidden_size=16)
        self.assertGreater(float(model.accept_probs().std()), 0.02)

    def test_continuation_encoding_distinguishes_suffixes(self):
        # phi must separate specific continuations, or conditioning on them buys nothing.
        torch.manual_seed(0)
        model = NeuralDFA(2, 4, hidden_size=16, num_lags=3, suffix_dim=8)
        x = torch.tensor([[1, 0, 1, 0], [1, 1, 1, 0]])
        enc = model.continuation_encodings(x)
        # K columns, one per continuation length 1..K; the empty one is not in here.
        self.assertEqual(enc.shape, (2, 5, 3, 8))
        # Length-2 continuations "10" vs "11" from position 0 must encode differently.
        self.assertGreater(float((enc[0, 0, 1] - enc[1, 0, 1]).abs().max()), 1e-3)


class TestNeuralLStar(unittest.TestCase):
    def test_parity(self):
        oracle_creator = BernoulliParityOracle
        oracle = oracle_creator(SymmetricBernoulli(p_correct=0.8), 0)
        dfa, info = train_neural_dfa(
            oracle,
            NeuralConfig(
                num_states=8,
                num_strings=3000,
                rounds=8,
                lambda_external=1.0,
                sift_strings=2048,
            ),
            log=lambda *_: None,
        )
        accuracy = evaluate_accuracy(dfa, oracle_creator)
        self.assertGreaterEqual(accuracy, 1 - assertion_allowed_error)
        # Mining is unsupervised, so the only queries are labelled prefixes plus the
        # accept-label denoising pass.
        self.assertLess(info["distinct_queries"], 100_000)


if __name__ == "__main__":
    unittest.main()
