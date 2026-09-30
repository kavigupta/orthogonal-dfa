import unittest
from types import SimpleNamespace

from orthogonal_dfa.l_star.memoized_oracle import MemoizedOracle
from orthogonal_dfa.l_star.suffix_family import SuffixFamily


class _ByLastSymbol:
    """Accepts a string ending in symbol ``v`` iff ``v`` is in ``accepting``."""

    alphabet_size = 2

    def __init__(self, accepting):
        self.accepting = accepting
        self.asked = []

    def membership_queries(self, strings):
        self.asked += strings
        return [int(s[-1] in self.accepting) for s in strings]


def _family(accepting, vs, reserve):
    oracle = _ByLastSymbol(accepting)
    table = SimpleNamespace(memo=MemoizedOracle(oracle), suffix=lambda v: bytes([v]))
    pst = SimpleNamespace(accept_thresh=0.7, reject_thresh=0.3, table=table)
    return SuffixFamily(pst, vs, reserve), oracle


class TestReread(unittest.TestCase):
    def test_a_mean_in_the_band_is_settled_by_the_reserve(self):
        # Half the family accepts, which is in the band; with the first of the
        # reserve, three of four do.
        family, _ = _family({0, 2, 3, 4, 5}, [0, 1], [2, 3, 4, 5])
        self.assertTrue(family.is_accept(b"", b""))

    def test_a_mean_inside_the_band_stays_undecided(self):
        family, _ = _family({0, 2}, [0, 1], [2, 3, 4, 5])
        self.assertIsNone(family.is_accept(b"", b""))

    def test_a_decided_read_never_touches_the_reserve(self):
        family, oracle = _family({0, 1}, [0, 1], [2, 3])
        self.assertTrue(family.is_accept(b"", b""))
        self.assertEqual(sorted(oracle.asked), [bytes([0]), bytes([1])])

    def test_the_reserve_is_read_a_family_at_a_time(self):
        # The first family's worth of reserve settles it, so the rest is not asked.
        family, oracle = _family({0, 2, 3}, [0, 1], [2, 3, 4, 5])
        self.assertTrue(family.is_accept(b"", b""))
        self.assertNotIn(bytes([4]), oracle.asked)

    def test_knows_a_read_the_reserve_has_not_settled_only_once_it_has(self):
        family, _ = _family({0, 2, 3}, [0, 1], [2, 3])
        family.mean(b"", b"")
        self.assertFalse(family.knows(b"", b""))
        family.is_accept(b"", b"")
        self.assertTrue(family.knows(b"", b""))


if __name__ == "__main__":
    unittest.main()
