"""The batched MaskTable fill paths must agree, cell for cell, with per-string queries."""

import unittest

import numpy as np

from orthogonal_dfa.l_star.mask_table import UNOBSERVED, MaskTable
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


class TestMaskTableBatching(unittest.TestCase):
    # These tests deliberately inspect MaskTable's lazy-observation internals
    # (_masks / _ensure); there is no public API for which cells are UNOBSERVED.
    # pylint: disable=protected-access
    def _table(self):
        oracle = HashOracle()
        table = MaskTable(oracle, [[0], [1], [0, 1]], [True, True, False])
        return oracle, table

    def _assert_cells_correct(self, oracle, table):
        for row, mask in enumerate(table._masks):
            suffix = table.suffix(row)
            for col, prefix in enumerate(table.prefixes):
                observed = mask[col]
                if observed != UNOBSERVED:
                    self.assertEqual(
                        bool(observed),
                        oracle.membership_query(list(prefix) + suffix),
                        (prefix, suffix),
                    )

    def test_add_prefixes_fills_family_columns(self):
        # add_prefixes flattens (suffix, prefix) pairs into one call and reshapes the
        # answers back, so the fixture has to be able to see both ways that can go
        # wrong: 2 full columns x 3 new prefixes is non-square (a reshape with the dims
        # swapped no longer fits), and the block is checked to be order-sensitive (a
        # swapped comprehension order changes the values, not just the layout).
        oracle, table = self._table()
        full = [table.intern_suffix([1, 1]), table.intern_suffix([0, 1, 0])]
        partial = table.intern_suffix([0])
        for row in full:
            table.column(row)
        table._ensure([partial], np.array([True, False, True]))
        new_prefixes = [[1, 1], [0, 0, 1], [1, 0, 0]]
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

    def test_ensure_queries_only_missing_cells_in_one_call(self):
        oracle, table = self._table()
        rows = [table.intern_suffix([1]), table.intern_suffix([0, 0])]
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
        rows = [table.intern_suffix([1]), table.intern_suffix([0, 0])]
        table._ensure(rows, np.ones(table.num_prefixes, dtype=bool))
        filled = np.array([table._masks[r] for r in rows])
        self.assertNotEqual(filled.min(), filled.max(), "fixture is order-blind")
        self._assert_cells_correct(oracle, table)


if __name__ == "__main__":
    unittest.main()
