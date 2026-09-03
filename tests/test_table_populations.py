"""The prefix populations a table can be asked to hold.

Prefixes are drawn for a reason -- uniformly, or to reach a state, or because
some family could not place them -- and a caller that means to read a rate over
each reason separately needs the table to remember which is which.
"""

import unittest

from orthogonal_dfa.l_star.mask_table import MaskTable

#: Anything: the table below is never asked a membership question.
_NO_ORACLE = None


def _table(prefixes):
    return MaskTable(_NO_ORACLE, prefixes)


def _words(n, offset=0):
    return [bytes([(i + offset) // 256, (i + offset) % 256]) for i in range(n)]


class TestPopulations(unittest.TestCase):
    def test_everything_is_one_population_until_told_otherwise(self):
        table = _table(_words(6))
        self.assertEqual(list(table.strata_masks()), ["baseline"])
        self.assertEqual(int(table.strata_masks()["baseline"].sum()), 6)

    def test_one_population_scopes_the_table_to_it(self):
        words = _words(6)
        table = _table(words)
        table.set_representative(words[:4], ["baseline"] * 4)

        masks = table.strata_masks()
        self.assertEqual(list(masks), ["baseline"])
        self.assertEqual(int(masks["baseline"].sum()), 4)
        self.assertEqual(int(table.representative.sum()), 4)

    def test_strata_name_the_population_each_prefix_was_drawn_for(self):
        baseline, state = _words(4), _words(3, offset=4)
        table = _table(baseline + state)
        table.set_representative(
            baseline + state, ["baseline"] * 4 + [("state", 2)] * 3
        )

        masks = table.strata_masks()
        self.assertEqual(int(masks["baseline"].sum()), 4)
        self.assertEqual(int(masks[("state", 2)].sum()), 3)

    def test_a_prefix_belongs_to_every_population_that_holds_it(self):
        # Drawn uniformly *and* found to reach state 0: evidence about both.
        shared, only_state = _words(5), _words(3, offset=5)
        table = _table(shared + only_state)
        table.set_representative(
            shared + shared + only_state,
            ["baseline"] * 5 + [("state", 0)] * 5 + [("state", 0)] * 3,
        )

        masks = table.strata_masks()
        self.assertEqual(int(masks["baseline"].sum()), 5)
        self.assertEqual(int(masks[("state", 0)].sum()), 8)
        # Which is more than the table holds, because five are in both.
        self.assertEqual(table.num_prefixes, 8)

    def test_the_representative_set_is_the_union_of_the_populations(self):
        baseline, state = _words(4), _words(3, offset=4)
        table = _table(baseline + state + _words(2, offset=7))
        table.set_representative(
            baseline + state, ["baseline"] * 4 + [("state", 1)] * 3
        )

        self.assertEqual(int(table.representative.sum()), 7)
        masks = table.strata_masks()
        union = masks["baseline"] | masks[("state", 1)]
        self.assertEqual(int(union.sum()), int(table.representative.sum()))

    def test_re_scoping_narrows_the_representative_set(self):
        words = _words(6)
        table = _table(words)
        self.assertEqual(int(table.representative.sum()), 6)

        table.set_representative(words[:2], ["baseline"] * 2)
        self.assertEqual(int(table.representative.sum()), 2)

    def test_a_population_of_prefixes_the_table_lacks_is_not_reported(self):
        words = _words(4)
        table = _table(words)
        # The state label names a prefix nobody added, so there is no column to
        # read a rate over and nothing to report.
        table.set_representative(
            words[:2] + _words(1, offset=9), ["baseline"] * 2 + [("state", 0)]
        )

        self.assertEqual(list(table.strata_masks()), ["baseline"])

    def test_freshly_sampled_prefixes_join_the_baseline(self):
        table = _table(_words(3))
        table.add_prefixes(_words(2, offset=3))

        self.assertEqual(int(table.strata_masks()["baseline"].sum()), 5)


if __name__ == "__main__":
    unittest.main()
