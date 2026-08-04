import unittest

import numpy as np

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.l_star.examples.spliceai_oracle import (
    SpliceModelOracle,
    wrap_with_flanks,
)

# cl=4 -> trim = cl//2+2 = 4, so the flanks are the first/last 4 bases of the text.
EXON = RawExon.of(4, "ACGTAAAAGGGG")
FLANK_L = [0, 1, 2, 3]
FLANK_R = [2, 2, 2, 2]


def wrapped_row(middle, width):
    row = FLANK_L + list(middle) + FLANK_R
    return row + [0] * (width - len(row))


class RecordingModel:
    """Fake model recording the argmax-decoded one-hot input it is given."""

    def __init__(self):
        self.seen = []

    def eval(self):
        return self

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


if __name__ == "__main__":
    unittest.main()
