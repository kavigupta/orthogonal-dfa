"""The batched oracle paths must agree, cell for cell, with the per-string ones."""

import unittest

import numpy as np

from orthogonal_dfa.l_star.lstar import _batch_before_possible_stop
from orthogonal_dfa.l_star.mask_table import UNOBSERVED, MaskTable
from orthogonal_dfa.l_star.memoized_oracle import MemoizedOracle
from orthogonal_dfa.l_star.midfix_tree import MidfixTree, oracle_decider
from orthogonal_dfa.l_star.statistics import binomial_side_of_boundary
from orthogonal_dfa.l_star.structures import Oracle


class HashOracle(Oracle):
    """Deterministic, order-independent, and records the batch sizes it saw."""

    alphabet_size = 2

    def __init__(self):
        self.calls = []

    def membership_query(self, string):
        return sum(3**i * (s + 1) for i, s in enumerate(string)) % 7 < 3

    def membership_queries(self, strings):
        self.calls.append(len(strings))
        return super().membership_queries(strings)


def _word(rng, length) -> bytes:
    return rng.integers(0, 2, size=length, dtype=np.uint8).tobytes()


def random_midfix_tree(rng, num_splits):
    base = [_word(rng, int(rng.integers(1, 4))) for _ in range(5)]
    tree = MidfixTree(base)
    for _ in range(num_splits):
        state = int(rng.choice(list(tree.leaves())))
        midfix = _word(rng, int(rng.integers(0, 3)))
        tree.split(state, midfix)
    return tree


class TestClassifyMany(unittest.TestCase):
    def test_matches_per_string_classify(self):
        rng = np.random.default_rng(0)
        saw_none = False
        for _ in range(25):
            tree = random_midfix_tree(rng, 6)
            # Half decisive, half tri-state, so the None ("could not classify")
            # path is hit.
            accept, reject = (0.5, 0.5) if rng.random() < 0.5 else (0.7, 0.3)
            strings = [_word(rng, int(rng.integers(0, 8))) for _ in range(40)]
            per_string, batched = HashOracle(), HashOracle()
            decide, _ = oracle_decider(per_string, tree.base_family, accept, reject)
            _, decide_level = oracle_decider(batched, tree.base_family, accept, reject)
            expected = [tree.classify(s, decide) for s in strings]
            self.assertEqual(expected, tree.classify_many(strings, decide_level))
            # Same work, far fewer calls: one per tree level rather than per string.
            self.assertEqual(sum(per_string.calls), sum(batched.calls))
            self.assertLessEqual(len(batched.calls), tree.depth)
            saw_none |= None in expected
        self.assertTrue(saw_none, "no undecided classification exercised")

    def test_empty(self):
        tree = random_midfix_tree(np.random.default_rng(0), 3)
        oracle = HashOracle()
        _, decide_level = oracle_decider(oracle, tree.base_family, 0.5, 0.5)
        self.assertEqual([], tree.classify_many([], decide_level))
        self.assertEqual([], oracle.calls)


class TestMaskTableBatching(unittest.TestCase):
    # These tests deliberately inspect MaskTable's lazy-observation internals
    # (_masks / _ensure); there is no public API for which cells are UNOBSERVED.
    # pylint: disable=protected-access
    def _table(self):
        oracle = HashOracle()
        table = MaskTable(
            oracle, [bytes([0]), bytes([1]), bytes([0, 1])], [True, True, False]
        )
        return oracle, table

    def _assert_cells_correct(self, oracle, table):
        for row, mask in enumerate(table._masks):
            suffix = table.suffix(row)
            for col, prefix in enumerate(table.prefixes):
                observed = mask[col]
                if observed != UNOBSERVED:
                    self.assertEqual(
                        bool(observed),
                        oracle.membership_query(prefix + suffix),
                        (prefix, suffix),
                    )

    def test_add_prefixes_fills_family_columns(self):
        # add_prefixes flattens (suffix, prefix) pairs into one call and reshapes the
        # answers back, so the fixture has to be able to see both ways that can go
        # wrong: 2 full columns x 3 new prefixes is non-square (a reshape with the dims
        # swapped no longer fits), and the block is checked to be order-sensitive (a
        # swapped comprehension order changes the values, not just the layout).
        oracle, table = self._table()
        full = [
            table.intern_suffix(bytes([1, 1])),
            table.intern_suffix(bytes([0, 1, 0])),
        ]
        partial = table.intern_suffix(bytes([0]))
        for row in full:
            table.column(row)
        table._ensure([partial], np.array([True, False, True]))
        new_prefixes = [bytes([1, 1]), bytes([0, 0, 1]), bytes([1, 0, 0])]
        block = [
            [oracle.membership_query(p + table.suffix(r)) for p in new_prefixes]
            for r in full
        ]
        self.assertNotEqual(len(full), len(new_prefixes), "fixture is reshape-blind")
        self.assertNotEqual(
            [c for row in block for c in row],
            [row[j] for j in range(len(new_prefixes)) for row in block],
            "fixture is order-blind",
        )

        table.add_prefixes(new_prefixes)
        self._assert_cells_correct(oracle, table)
        # The fully-observed columns stay fully observed; the partial one does not
        # acquire cells it was never asked for.
        for row in full:
            self.assertFalse((table._masks[row] == UNOBSERVED).any())
        self.assertEqual(
            1 + len(new_prefixes), int((table._masks[partial] == UNOBSERVED).sum())
        )

    def test_every_prefix_gates_a_columns_candidacy(self):
        # Regression: scoping "fully observed" to the *representative* rows made a
        # column a clustering candidate while its core cells -- non-representative,
        # but still prefixes -- were unobserved.  That silently changed which
        # suffixes clustering could pick, and with them the whole search path,
        # surfacing only as a distant end-to-end timeout.
        oracle, table = self._table()  # prefix 2 is core: non-representative
        row = table.intern_suffix(bytes([1, 1]))

        # Observing just the representative cells leaves a prefix unobserved, so
        # the column is not a candidate yet.
        table._ensure([row], table.representative)
        self.assertTrue((table._masks[row] == UNOBSERVED).any())
        self.assertNotIn(row, list(table.fully_observed()))

        # Observing every prefix makes it one.
        oracle.calls.clear()
        table.column(row)
        self.assertEqual([int((~table.representative).sum())], oracle.calls)
        self.assertIn(row, list(table.fully_observed()))

    def test_ensure_queries_only_missing_cells_in_one_call(self):
        oracle, table = self._table()
        rows = [table.intern_suffix(bytes([1])), table.intern_suffix(bytes([0, 0]))]
        narrow = np.array([True, False, True])
        wide = np.ones(table.num_prefixes, dtype=bool)
        table._ensure(rows, narrow)
        self.assertEqual([len(rows) * int(narrow.sum())], oracle.calls)
        self._assert_cells_correct(oracle, table)
        # Widening asks only for the cells the narrow mask left out, still in one call.
        oracle.calls.clear()
        table._ensure(rows, wide)
        self.assertEqual([len(rows) * int((~narrow).sum())], oracle.calls)
        self._assert_cells_correct(oracle, table)
        # Everything is observed now, so a repeat asks the oracle nothing.
        oracle.calls.clear()
        table._ensure(rows, wide)
        self.assertEqual([], oracle.calls)

    def test_ensure_scatters_answers_to_the_right_cells(self):
        # The scatter-back is a zip over a flat result list; a misordered zip is only
        # visible if the cells being filled disagree with each other.
        oracle, table = self._table()
        rows = [table.intern_suffix(bytes([1])), table.intern_suffix(bytes([0, 0]))]
        table._ensure(rows, np.ones(table.num_prefixes, dtype=bool))
        filled = np.array([table._masks[r] for r in rows])
        self.assertNotEqual(filled.min(), filled.max(), "fixture is order-blind")
        self._assert_cells_correct(oracle, table)


class TestBatchBeforePossibleStop(unittest.TestCase):
    """The look-ahead chunk must never span the sequential early-stop: below the
    returned size the binomial test provably cannot fire, so batching that many is
    identical to drawing them one at a time."""

    boundary = 0.98
    min_valid = 30
    remaining = 2000
    states = [(0, 0), (30, 30), (29, 30), (98, 100), (490, 500), (1900, 1950)]

    def _fires(self, agreements, valid, k):
        return (
            binomial_side_of_boundary(agreements + k, valid + k, self.boundary) is True
            or binomial_side_of_boundary(agreements, valid + k, self.boundary) is False
        )

    def test_never_spans_the_stop(self):
        # No k strictly inside the chunk (at or above the min_valid floor) can fire.
        for a, n in self.states:
            k = _batch_before_possible_stop(
                a, n, self.boundary, self.min_valid, self.remaining
            )
            self.assertGreaterEqual(n + k, self.min_valid)  # respects the floor
            floor = max(self.min_valid - n, 1)
            for kp in range(floor, k):
                self.assertFalse(self._fires(a, n, kp), (a, n, kp))

    def test_chunk_is_maximal(self):
        # It is the *largest* safe chunk: firing is possible exactly at k (unless we
        # ran out of budget), so stopping one sooner would have left batching on table.
        for a, n in self.states:
            k = _batch_before_possible_stop(
                a, n, self.boundary, self.min_valid, self.remaining
            )
            if k < self.remaining:
                self.assertTrue(self._fires(a, n, k), (a, n, k))

    def test_capped_by_remaining(self):
        self.assertEqual(
            5, _batch_before_possible_stop(0, 0, self.boundary, self.min_valid, 5)
        )


class TestMemoizedOracle(unittest.TestCase):
    def test_caches_batches_and_dedupes(self):
        oracle = HashOracle()
        memo = MemoizedOracle(oracle)
        strings = [bytes([1, 0, 1]), bytes([0, 0]), bytes([1, 0, 1])]  # note the repeat
        bits = memo.membership_queries(strings)
        self.assertEqual([oracle.membership_query(s) for s in strings], bits)
        self.assertEqual([2], oracle.calls, "one batched call, deduped")
        oracle.calls.clear()
        self.assertEqual(bits, memo.membership_queries(strings))
        self.assertEqual([], oracle.calls)
        # the single-string query rides the same cache
        oracle.calls.clear()
        self.assertEqual(
            oracle.membership_query(bytes([1, 0, 1])),
            memo.membership_query(bytes([1, 0, 1])),
        )
        self.assertEqual([], oracle.calls, "cached, no new call")

    def test_distinct_strings_do_not_share_an_answer(self):
        # The cache keys on a digest, so a collision would be invisible: it
        # would hand one string another's bit rather than fail.
        oracle = HashOracle()
        memo = MemoizedOracle(oracle)
        rng = np.random.default_rng(0)
        strings = [_word(rng, int(rng.integers(1, 40))) for _ in range(2000)]
        self.assertEqual(
            [oracle.membership_query(s) for s in strings],
            memo.membership_queries(strings),
        )

    def test_prefixes_and_permutations_are_distinct_keys(self):
        oracle = HashOracle()
        memo = MemoizedOracle(oracle)
        strings = [
            bytes([1]),
            bytes([1, 0]),
            bytes([0, 1]),
            bytes([1, 0, 0]),
            bytes([0, 0, 1]),
            bytes([1, 0, 0, 0]),
        ]
        self.assertEqual(
            [oracle.membership_query(s) for s in strings],
            memo.membership_queries(strings),
        )


if __name__ == "__main__":
    unittest.main()
