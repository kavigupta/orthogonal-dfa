import unittest

import numpy as np
import torch

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.data.sample_text import sample_text
from orthogonal_dfa.l_star.examples.spliceai_oracle import (
    SpliceModelOracle,
    flanks,
    median_threshold,
    run_over_middles,
    wrap_with_flanks,
)
from orthogonal_dfa.oracle.run_model import compute_exon_scores
from orthogonal_dfa.spliceai.exon_score import (
    FLANK_MARGIN,
    SpliceAIExonScore,
    forward_batch,
    full_lengths,
    spliceai_exon_scores,
)
from orthogonal_dfa.spliceai.module import SpliceAIModule

# cl=4 -> trim = cl//2+2 = 4, so the flanks are the first/last 4 bases of the text.
EXON = RawExon.of(4, "ACGTAAAAGGGG")
FLANK_L = [0, 1, 2, 3]
FLANK_R = [2, 2, 2, 2]

# The smallest window SpliceAI is defined for; a forward pass takes a few ms.
SMALL_CL = 80
FULL_LENGTH = 30


def wrapped_row(middle, width):
    row = FLANK_L + list(middle) + FLANK_R
    return row + [0] * (width - len(row))


def small_model_and_exon(cl=SMALL_CL):
    """A randomly initialised SpliceAI of context ``cl``, and an exon to match."""
    torch.manual_seed(0)
    model = SpliceAIModule(window=SMALL_CL).eval()
    rng = np.random.default_rng(0)
    exon = RawExon(
        cl, rng.integers(0, 4, size=cl + 2 * FLANK_MARGIN + FULL_LENGTH).tolist()
    )
    return model, exon


class RecordingScore(torch.nn.Module):
    """Fake score model recording its argmax-decoded input; scores a row by length."""

    def __init__(self):
        super().__init__()
        self.seen = []
        self.seen_lengths = []

    def forward(self, x, lengths):
        self.seen.append(x.argmax(-1).cpu().numpy())
        self.seen_lengths.append(lengths.cpu().numpy())
        return lengths.float()


class TestWrapWithFlanks(unittest.TestCase):
    def test_wraps_pads_and_reports_lengths(self):
        strings = [[1, 2], [3], []]
        wrapped, lengths = wrap_with_flanks(
            np.array(FLANK_L), np.array(FLANK_R), strings
        )
        self.assertEqual(wrapped.shape, (3, 10))  # flank 8 + longest middle 2
        np.testing.assert_array_equal(lengths, [2, 1, 0])
        np.testing.assert_array_equal(wrapped[0], wrapped_row([1, 2], 10))
        np.testing.assert_array_equal(wrapped[1], wrapped_row([3], 10))
        np.testing.assert_array_equal(wrapped[2], wrapped_row([], 10))


class TestSpliceModelOracle(unittest.TestCase):
    def setUp(self):
        self.model = RecordingScore().eval()

    def _oracle(self, threshold=1.5, **kw):
        return SpliceModelOracle(EXON, self.model, threshold, device="cpu", **kw)

    def test_alphabet_and_length(self):
        oracle = self._oracle()
        self.assertEqual(oracle.alphabet_size, 4)
        self.assertEqual(oracle.string_length, EXON.random_text_length)

    def test_membership_batching_and_wrapping(self):
        oracle = self._oracle(threshold=1.5, chunk=2)
        result = oracle.membership_queries([[1, 2], [3], [0, 1, 2, 3, 0], []])

        # the score is the middle length, so length > 1.5 -> [T, F, T, F]
        np.testing.assert_array_equal(result, [True, False, True, False])
        self.assertEqual(result.dtype, bool)

        self.assertEqual(len(self.model.seen), 2)
        np.testing.assert_array_equal(self.model.seen_lengths[0], [2, 1])
        np.testing.assert_array_equal(self.model.seen_lengths[1], [5, 0])
        w0, w1 = self.model.seen[0], self.model.seen[1]
        self.assertEqual(w0.shape, (2, 10))
        self.assertEqual(w1.shape, (2, 13))
        np.testing.assert_array_equal(w0[0], wrapped_row([1, 2], 10))
        np.testing.assert_array_equal(w1[0], wrapped_row([0, 1, 2, 3, 0], 13))
        np.testing.assert_array_equal(w1[1], wrapped_row([], 13))

    def test_membership_query_singular(self):
        oracle = self._oracle(threshold=1.5)
        self.assertTrue(oracle.membership_query([0, 0, 0]))  # length 3 > 1.5
        self.assertFalse(oracle.membership_query([0]))  # length 1 < 1.5

    def test_empty_batch(self):
        result = self._oracle().membership_queries([])
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, bool)

    def test_rejects_a_model_in_train_mode(self):
        """Padding invariance has to hold at every query, so the oracle refuses a
        train-mode model rather than quietly switching one it does not own."""
        oracle = self._oracle()
        self.model.train()
        with self.assertRaises(AssertionError):
            oracle.membership_queries([[1, 2]])


class TestSpliceaiExonScore(unittest.TestCase):
    """Everything here runs a real SpliceAI, so the score's output indexing is
    checked against the model's actual cl trimming rather than a fake."""

    def setUp(self):
        self.model, self.exon = small_model_and_exon()
        self.score_model = SpliceAIExonScore(self.model).eval()
        self.flank_l, self.flank_r = flanks(self.exon)
        self.rng = np.random.default_rng(1)

    def _scores(self, middles, chunk=64):
        return run_over_middles(
            self.score_model,
            self.flank_l,
            self.flank_r,
            middles,
            device="cpu",
            chunk=chunk,
        )

    def test_matches_first_and_last_output_positions(self):
        """On full-length rows the score is the acceptor at output position 0 and
        the donor at the last, which is what oracle.run_model has always read."""
        middles = self.rng.integers(0, 4, size=(4, FULL_LENGTH)).tolist()
        wrapped, _ = wrap_with_flanks(self.flank_l, self.flank_r, middles)
        logits = forward_batch(self.model, wrapped, device="cpu")
        self.assertEqual(logits.shape[1], FULL_LENGTH + 2 * FLANK_MARGIN)

        reference = logits.log_softmax(-1)[:, [0, -1], [1, 2]].mean(-1)
        np.testing.assert_allclose(
            spliceai_exon_scores(logits, full_lengths(logits)).numpy(),
            reference.numpy(),
            rtol=1e-6,
        )
        np.testing.assert_allclose(self._scores(middles), reference.numpy(), rtol=1e-6)

    def test_padding_does_not_change_scores(self):
        """A short middle scores the same whether or not it is padded up to a
        longer row's width."""
        middles = [
            self.rng.integers(0, 4, size=n).tolist() for n in [FULL_LENGTH, 17, 5, 1, 0]
        ]
        ragged = self._scores(middles)
        one_at_a_time = np.concatenate([self._scores([m]) for m in middles])
        np.testing.assert_allclose(ragged, one_at_a_time, rtol=1e-6)

    def test_rejects_a_model_whose_cl_disagrees(self):
        """Flanks cut for a cl=400 exon put the donor 320 positions off for an
        80-context model; that has to fail loudly rather than score noise."""
        _, wide_exon = small_model_and_exon(cl=400)
        flank_l, flank_r = flanks(wide_exon)
        with self.assertRaises(AssertionError):
            run_over_middles(
                self.score_model,
                flank_l,
                flank_r,
                [[0] * FULL_LENGTH],
                device="cpu",
                chunk=64,
            )


class TestCalibration(unittest.TestCase):
    def setUp(self):
        self.model, self.exon = small_model_and_exon()
        self.score_model = SpliceAIExonScore(self.model).eval()

    def test_threshold_splits_random_middles_in_half(self):
        length, count = 12, 256
        # .function skips permacache: nothing here is worth caching to disk.
        threshold = median_threshold.function(
            self.score_model, self.exon, length, count=count, chunk=64, device="cpu"
        )
        oracle = SpliceModelOracle(
            self.exon, self.score_model, threshold, device="cpu", chunk=64
        )
        fresh = np.random.default_rng(7).integers(0, 4, size=(count, length)).tolist()
        self.assertAlmostEqual(oracle.membership_queries(fresh).mean(), 0.5, delta=0.15)

    def test_oracle_accepts_exactly_the_scores_above_the_threshold(self):
        flank_l, flank_r = flanks(self.exon)
        middles = [
            np.random.default_rng(3).integers(0, 4, size=n).tolist()
            for n in [FULL_LENGTH, 9, 2]
        ]
        kw = dict(device="cpu", chunk=64)
        scores = run_over_middles(self.score_model, flank_l, flank_r, middles, **kw)
        threshold = float(np.median(scores))
        oracle = SpliceModelOracle(self.exon, self.score_model, threshold, **kw)
        np.testing.assert_array_equal(
            oracle.membership_queries(middles), scores > threshold
        )


class TestComputeExonScores(unittest.TestCase):
    """oracle.run_model's path, where the output width is derived from itself and
    so the cl check has to come from the input width instead."""

    def setUp(self):
        self.model, self.exon = small_model_and_exon()

    def test_agrees_with_the_oracle_path(self):
        middles, arr = sample_text(self.exon, 0, 4)
        flank_l, flank_r = flanks(self.exon)
        via_oracle = run_over_middles(
            SpliceAIExonScore(self.model).eval(),
            flank_l,
            flank_r,
            middles.tolist(),
            device="cpu",
            chunk=64,
        )
        np.testing.assert_allclose(
            compute_exon_scores(self.model, arr, cl=self.exon.cl).numpy(),
            via_oracle,
            rtol=1e-6,
        )

    def test_rejects_a_model_whose_cl_disagrees(self):
        _, wide_exon = small_model_and_exon(cl=400)
        _, arr = sample_text(wide_exon, 0, 2)
        with self.assertRaises(AssertionError):
            compute_exon_scores(self.model, arr, cl=wide_exon.cl)


if __name__ == "__main__":
    unittest.main()
