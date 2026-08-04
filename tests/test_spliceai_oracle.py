import unittest
import warnings

import numpy as np
import torch

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.data.sample_text import sample_text
from orthogonal_dfa.l_star.examples.spliceai_oracle import (
    SpliceModelOracle,
    calibrated_spliceai_readout,
    flanks,
    median_threshold,
    run_over_middles,
    wrap_with_flanks,
)
from orthogonal_dfa.oracle.run_model import compute_exon_scores
from orthogonal_dfa.spliceai.exon_score import (
    FLANK_MARGIN,
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


class RecordingModel:
    """Fake model recording the argmax-decoded one-hot input it is given."""

    def __init__(self):
        self.seen = []
        self.training = False

    def __call__(self, x):
        self.seen.append(x.argmax(-1).cpu().numpy())
        return x


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
        self.model = RecordingModel()
        self.seen_lengths = []

        def readout(logits, lengths):
            del logits  # unused
            self.seen_lengths.append(lengths.cpu().numpy())
            return lengths >= 2

        self.readout = readout

    def _oracle(self, **kw):
        return SpliceModelOracle(EXON, self.model, self.readout, device="cpu", **kw)

    def test_alphabet_and_length(self):
        oracle = self._oracle()
        self.assertEqual(oracle.alphabet_size, 4)
        self.assertEqual(oracle.string_length, EXON.random_text_length)

    def test_membership_batching_and_wrapping(self):
        oracle = self._oracle(chunk=2)
        result = oracle.membership_queries([[1, 2], [3], [0, 1, 2, 3, 0], []])

        np.testing.assert_array_equal(result, [True, False, True, False])
        self.assertEqual(result.dtype, bool)

        self.assertEqual(len(self.model.seen), 2)
        np.testing.assert_array_equal(self.seen_lengths[0], [2, 1])
        np.testing.assert_array_equal(self.seen_lengths[1], [5, 0])
        w0, w1 = self.model.seen[0], self.model.seen[1]
        self.assertEqual(w0.shape, (2, 10))
        self.assertEqual(w1.shape, (2, 13))
        np.testing.assert_array_equal(w0[0], wrapped_row([1, 2], 10))
        np.testing.assert_array_equal(w1[0], wrapped_row([0, 1, 2, 3, 0], 13))
        np.testing.assert_array_equal(w1[1], wrapped_row([], 13))

    def test_membership_query_singular(self):
        oracle = self._oracle()
        self.assertTrue(oracle.membership_query([0, 0, 0]))
        self.assertFalse(oracle.membership_query([0]))

    def test_empty_batch(self):
        result = self._oracle().membership_queries([])
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, bool)

    def test_rejects_a_model_in_train_mode(self):
        """Padding invariance has to hold at every query, so the oracle refuses a
        train-mode model rather than quietly switching one it does not own."""
        oracle = self._oracle()
        self.model.training = True
        with self.assertRaises(AssertionError):
            oracle.membership_queries([[1, 2]])


class TestSpliceaiExonScores(unittest.TestCase):
    """Everything here runs a real SpliceAI, so the readout's output indexing is
    checked against the model's actual cl trimming rather than a fake."""

    def setUp(self):
        self.model, self.exon = small_model_and_exon()
        self.flank_l, self.flank_r = flanks(self.exon)
        self.rng = np.random.default_rng(1)

    def _scores(self, middles, chunk=64):
        return run_over_middles(
            self.model,
            self.flank_l,
            self.flank_r,
            middles,
            spliceai_exon_scores,
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
                self.model,
                flank_l,
                flank_r,
                [[0] * FULL_LENGTH],
                spliceai_exon_scores,
                device="cpu",
                chunk=64,
            )


class TestCalibration(unittest.TestCase):
    def setUp(self):
        self.model, self.exon = small_model_and_exon()

    def test_threshold_splits_random_middles_in_half(self):
        length, count = 12, 256
        # .function skips permacache: nothing here is worth caching to disk.
        threshold = median_threshold.function(
            self.model, self.exon, length, count=count, chunk=64, device="cpu"
        )
        oracle = SpliceModelOracle(
            self.exon,
            self.model,
            calibrated_spliceai_readout(threshold, length),
            device="cpu",
            chunk=64,
        )
        fresh = np.random.default_rng(7).integers(0, 4, size=(count, length)).tolist()
        self.assertAlmostEqual(oracle.membership_queries(fresh).mean(), 0.5, delta=0.15)

    def test_warns_once_per_off_calibration_length(self):
        oracle = SpliceModelOracle(
            self.exon,
            self.model,
            calibrated_spliceai_readout(0.0, FULL_LENGTH),
            device="cpu",
            chunk=64,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            oracle.membership_queries([[0] * n for n in [FULL_LENGTH, 5, 5, 7]])
            oracle.membership_queries([[0] * 5])
        self.assertEqual(len(caught), 2)  # lengths 5 and 7, once each
        self.assertIn(
            f"calibrated at middle length {FULL_LENGTH}", str(caught[0].message)
        )

    def test_readout_accepts_exactly_the_scores_above_the_threshold(self):
        flank_l, flank_r = flanks(self.exon)
        middles = [
            np.random.default_rng(3).integers(0, 4, size=n).tolist()
            for n in [FULL_LENGTH, 9, 2]
        ]
        kw = dict(device="cpu", chunk=64)
        scores = run_over_middles(
            self.model, flank_l, flank_r, middles, spliceai_exon_scores, **kw
        )
        threshold = float(np.median(scores))
        oracle = SpliceModelOracle(
            self.exon,
            self.model,
            calibrated_spliceai_readout(threshold, FULL_LENGTH),
            **kw,
        )
        with warnings.catch_warnings():  # the two short middles are off-calibration
            warnings.simplefilter("ignore")
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
            self.model,
            flank_l,
            flank_r,
            middles.tolist(),
            spliceai_exon_scores,
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
