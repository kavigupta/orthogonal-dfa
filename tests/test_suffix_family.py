import unittest
from types import SimpleNamespace

from orthogonal_dfa.l_star.memoized_oracle import MemoizedOracle
from orthogonal_dfa.l_star.suffix_family import _SIFT_BLOCK, SuffixFamily


class _CountingBase:
    """Deterministic membership from a per-string rule, counting real queries."""

    def __init__(self, accept):
        self._accept = accept
        self.queries = 0

    def membership_queries(self, strings):
        self.queries += len(strings)
        return [int(self._accept(tuple(s))) for s in strings]


def _family(accept, *, n=200, accept_thresh=0.55, reject_thresh=0.45):
    """A family of ``n`` single-symbol suffixes over a counting oracle.

    ``accept(suffix_index)`` fixes the membership of ``base + [suffix_index]`` --
    the base is irrelevant, so a whole family's accept-rate is set by one rule.
    """
    base = _CountingBase(lambda s: accept(s[-1]))
    pst = SimpleNamespace(
        table=SimpleNamespace(suffix=lambda v: [v]),
        sift_cache=MemoizedOracle(base),
        accept_thresh=accept_thresh,
        reject_thresh=reject_thresh,
    )
    return SuffixFamily(pst, list(range(n))), base


class TestSequentialIsAccept(unittest.TestCase):
    def test_clear_accept_stops_after_one_block(self):
        family, base = _family(lambda v: True)
        self.assertIs(family.is_accept([1], []), True)
        self.assertEqual(base.queries, _SIFT_BLOCK)

    def test_clear_reject_stops_after_one_block(self):
        family, base = _family(lambda v: False)
        self.assertIs(family.is_accept([1], []), False)
        self.assertEqual(base.queries, _SIFT_BLOCK)

    def test_boundary_is_indecisive_and_spends_whole_family(self):
        family, base = _family(lambda v: v % 2 == 0, n=200)
        self.assertIsNone(family.is_accept([1], []))
        self.assertEqual(base.queries, 200)

    def test_verdict_is_memoized(self):
        family, base = _family(lambda v: True)
        family.is_accept([1], [])
        family.is_accept([1], [])
        self.assertEqual(base.queries, _SIFT_BLOCK)

    def test_matches_exact_full_family_decision(self):
        # Rates far from the thresholds: the early stop must agree with the plain
        # full-family mean it approximates.
        for rate, expected in [(0.9, True), (0.1, False), (0.5, None)]:
            with self.subTest(rate=rate):
                cut = round(rate * 200)
                family, _ = _family(lambda v, cut=cut: v < cut, n=200)
                self.assertEqual(family.is_accept([7], [2]), expected)

    def test_warm_sift_touches_only_the_first_block(self):
        family, base = _family(lambda v: True, n=200)
        family.warm_sift([[1], [2], [3]])
        self.assertEqual(base.queries, 3 * _SIFT_BLOCK)


if __name__ == "__main__":
    unittest.main()
