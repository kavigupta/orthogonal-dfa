"""The batched oracle paths must agree, cell for cell, with the per-string ones."""

import unittest

import numpy as np

from orthogonal_dfa.l_star.mask_table import UNOBSERVED, MaskTable
from orthogonal_dfa.l_star.memoized_oracle import MemoizedOracle
from orthogonal_dfa.l_star.midfix_tree import MidfixTree
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


def _decider(oracle, base, accept, reject):
    """Reads one string, or a level of them in one call, by its mean over
    ``base``: above ``accept`` accepts, below ``reject`` rejects."""

    def decide_level(pairs):
        bits = oracle.membership_queries([s + m + v for s, m in pairs for v in base])
        means = [
            sum(bits[i : i + len(base)]) / len(base)
            for i in range(0, len(bits), len(base))
        ]
        return [True if m > accept else False if m < reject else None for m in means]

    return (lambda s, m: decide_level([(s, m)])[0]), decide_level


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
            decide, _ = _decider(per_string, tree.base_family, accept, reject)
            _, decide_level = _decider(batched, tree.base_family, accept, reject)
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
        _, decide_level = _decider(oracle, tree.base_family, 0.5, 0.5)
        self.assertEqual([], tree.classify_many([], decide_level))
        self.assertEqual([], oracle.calls)


class TestMaskTableBatching(unittest.TestCase):
    # These tests deliberately inspect MaskTable's lazy-observation internals
    # (_masks / _ensure); there is no public API for which cells are UNOBSERVED.
    # pylint: disable=protected-access
    def _table(self):
        oracle = HashOracle()
        prefixes = [bytes([0]), bytes([1]), bytes([0, 1])]
        table = MaskTable(oracle, prefixes[:2], population="uniform")
        # Added and then retired: a prefix is named as it enters, so being in
        # the table and in no population is what outliving a population means.
        table.add_prefixes(prefixes[2:], population="scratch")
        table.drop_population("scratch")
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

    def test_add_prefixes_queries_nothing(self):
        oracle, table = self._table()
        rows = [table.intern_suffix(bytes([1, 1])), table.intern_suffix(bytes([0]))]
        table.column(rows[0])
        oracle.calls.clear()
        new_prefixes = [bytes([1, 1]), bytes([0, 0, 1]), bytes([1, 0, 0])]
        table.add_prefixes(new_prefixes, population="uniform")
        self.assertEqual([], oracle.calls)
        table.observed_masks(rows, table.representative)
        self._assert_cells_correct(oracle, table)

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
