"""The tripartite search keeps every promise ``TripartiteSearchCorrect`` makes."""

import itertools
import unittest

import numpy as np

from orthogonal_dfa.l_star.tripartite_search import (
    ALL_AGREE,
    DISAGREE_LO,
    EDGE,
    PAIR,
    TRIPLE,
    UNDECIDED_HI,
    UNDECIDED_LO,
    UNDECIDED_MID,
    tripartite_search,
)


def clog2(n):
    """``Nat.clog 2 n``: the least ``k`` with ``n ≤ 2^k``, and ``0`` at ``n = 0``."""
    return 0 if n <= 1 else (n - 1).bit_length()


def search_logged(table, lo, hi, recover):
    calls = []

    def agrees(i):
        calls.append(i)
        return table[i]

    return tripartite_search(agrees, lo, hi, recover=recover), calls


def check(test, table, lo, hi, recover):
    """Asserts every conjunct of ``TripartiteSearchCorrect`` at one input."""
    found, calls = search_logged(table, lo, hi, recover)
    kind, j = found.kind, found.position
    test.assertEqual(list(found.queried), calls)
    test.assertTrue(all(lo <= p <= hi for p in calls))
    test.assertLessEqual(len(calls), 2 + 3 * clog2(hi - lo))
    if kind == UNDECIDED_LO:
        test.assertIsNone(table[lo])
    elif kind == DISAGREE_LO:
        test.assertIs(table[lo], False)
    elif kind == UNDECIDED_HI:
        test.assertEqual((table[lo], table[hi]), (True, None))
    elif kind == ALL_AGREE:
        test.assertEqual((table[lo], table[hi]), (True, True))
    elif kind == EDGE:
        test.assertTrue(lo < j <= hi)
        test.assertEqual((table[j - 1], table[j]), (True, False))
    elif kind == PAIR:
        test.assertTrue(lo <= j and j + 1 <= hi)
        test.assertEqual((table[j], table[j + 1]), (None, None))
    elif kind == TRIPLE:
        test.assertTrue(lo < j < hi)
        test.assertEqual((table[j - 1], table[j], table[j + 1]), (True, None, False))
    else:
        test.assertEqual(kind, UNDECIDED_MID)
        test.assertFalse(recover)
        test.assertTrue(lo < j < hi)
        test.assertIsNone(table[j])
    return found


class TestEveryTable(unittest.TestCase):
    def test_every_table_on_ranges_up_to_seven(self):
        for n in range(1, 8):
            for table in itertools.product([None, True, False], repeat=n):
                for lo in range(n):
                    for hi in range(lo, n):
                        for recover in (False, True):
                            check(self, table, lo, hi, recover)

    def test_every_turn_with_nothing_undecided_is_found(self):
        for n in range(2, 12):
            for lo in range(n):
                for hi in range(lo + 1, n):
                    for t in range(lo + 1, hi + 1):
                        table = [i < t for i in range(n)]
                        for recover in (False, True):
                            found = check(self, table, lo, hi, recover)
                            self.assertEqual((EDGE, t), (found.kind, found.position))


class TestLargeRanges(unittest.TestCase):
    def test_random_tables(self):
        rng = np.random.default_rng(0)
        for _ in range(3000):
            n = int(rng.integers(2, 2000))
            undecided = rng.random()
            table = [
                None if rng.random() < undecided else bool(rng.random() < 0.5)
                for _ in range(n)
            ]
            lo = int(rng.integers(n))
            hi = int(rng.integers(lo, n))
            table[lo] = rng.choice([True, None]) if rng.random() < 0.1 else True
            table[hi] = rng.choice([False, None]) if rng.random() < 0.1 else False
            for recover in (False, True):
                check(self, table, lo, hi, recover)

    def test_random_turns(self):
        rng = np.random.default_rng(1)
        for _ in range(3000):
            n = int(rng.integers(2, 100000))
            lo = int(rng.integers(n - 1))
            hi = int(rng.integers(lo + 1, n))
            t = int(rng.integers(lo + 1, hi + 1))

            def agrees(i, t=t):
                return i < t

            found = tripartite_search(agrees, lo, hi, recover=True)
            self.assertEqual((EDGE, t), (found.kind, found.position))
            self.assertLessEqual(len(found.queried), 2 + clog2(hi - lo))


class TestExamples(unittest.TestCase):
    def test_an_undecided_midpoint_is_stepped_past(self):
        table = [True, True, True, None, True, False, False]
        found, _ = search_logged(table, 0, 6, recover=True)
        self.assertEqual((EDGE, 5, (0, 6, 3, 2, 4, 5)), astuple(found))

    def test_without_recovery_an_undecided_midpoint_ends_the_search(self):
        table = [True, True, True, None, True, False, False]
        found, _ = search_logged(table, 0, 6, recover=False)
        self.assertEqual((UNDECIDED_MID, 3, (0, 6, 3)), astuple(found))

    def test_an_undecided_run_between_the_sides_is_a_triple(self):
        table = [True, True, None, False, False]
        found, _ = search_logged(table, 0, 4, recover=True)
        self.assertEqual((TRIPLE, 2, (0, 4, 2, 1, 3)), astuple(found))

    def test_two_undecided_neighbours_are_a_pair(self):
        table = [True, None, None, False, False]
        found, _ = search_logged(table, 0, 4, recover=True)
        self.assertEqual((PAIR, 1, (0, 4, 2, 1)), astuple(found))


def astuple(found):
    return found.kind, found.position, found.queried


if __name__ == "__main__":
    unittest.main()
