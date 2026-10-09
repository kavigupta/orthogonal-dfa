import unittest
from types import SimpleNamespace

import numpy as np

from orthogonal_dfa.l_star.suffix_family import SuffixFamily


class TestSuffixFamily(unittest.TestCase):
    def test_a_later_boundary_does_not_move_an_earlier_family(self):
        pst = SimpleNamespace(accept_thresh=0.6, reject_thresh=0.4)
        family = SuffixFamily(pst, [0, 1, 2, 3])
        votes = [1, 1, 0, 1]
        self.assertIsNone(family.train_side(votes))
        pst.accept_thresh, pst.reject_thresh = 0.5, 0.3
        self.assertIsNone(family.train_side(votes))

    def test_the_middle_of_the_band_sides_by_the_mean_and_a_tie_rejects(self):
        family = SuffixFamily(
            SimpleNamespace(accept_thresh=0.6, reject_thresh=0.4), [0, 1, 2, 3]
        )
        # pylint: disable=protected-access
        family._means.update({b"a": 0.55, b"b": 0.45, b"c": 0.5})
        self.assertTrue(family.middle_side(b"a", b""))
        self.assertFalse(family.middle_side(b"b", b""))
        self.assertFalse(family.middle_side(b"c", b""))


class _Table:
    """Suffix rows that are themselves, read from fixed random bits, counting
    the reads."""

    def __init__(self, rng):
        self.bit = {}
        self.rng = rng
        self.memo = self
        self.reads = 0

    def suffix(self, v):
        return bytes([v])

    def membership_queries(self, strings):
        self.reads += len(strings)
        return [self.bit.setdefault(s, int(self.rng.integers(2))) for s in strings]


class TestReadingOnlyAsFarAsTheVerdict(unittest.TestCase):
    def test_it_reads_the_verdicts_the_whole_family_would(self):
        rng = np.random.default_rng(0)
        for accept, reject in ((0.6, 0.4), (0.7, 0.3), (0.55, 0.55)):
            table = _Table(rng)
            pst = SimpleNamespace(
                accept_thresh=accept, reject_thresh=reject, table=table
            )
            early, full = SuffixFamily(pst, range(62)), SuffixFamily(pst, range(62))
            for _ in range(200):
                # Strings skewed to either side, so verdicts settle early too.
                seq = rng.integers(0, 2, size=6, dtype=np.uint8).tobytes()
                lean = rng.choice([0.2, 0.5, 0.8])
                for v in range(62):
                    table.bit[seq + bytes([v])] = int(rng.random() < lean)
                full.mean(seq, b"")
                self.assertEqual(full.is_accept(seq, b""), early.is_accept(seq, b""))
                self.assertEqual(
                    full.middle_side(seq, b""), early.middle_side(seq, b"")
                )

    def test_a_clear_string_is_not_read_through(self):
        table = _Table(np.random.default_rng(0))
        family = SuffixFamily(
            SimpleNamespace(accept_thresh=0.6, reject_thresh=0.4, table=table),
            range(62),
        )
        table.bit.update({b"s" + bytes([v]): 1 for v in range(62)})

        self.assertTrue(family.is_accept(b"s", b""))
        self.assertLess(table.reads, 62)


if __name__ == "__main__":
    unittest.main()
