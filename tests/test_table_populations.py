"""The prefix populations a table holds.

Prefixes are drawn for a reason -- uniformly, or to reach a state, or because
some family could not place them -- and a caller that means to read a rate over
each reason separately needs the table to remember which is which.  A prefix is
named as it enters, so the table is never between a population and its members.
"""

import unittest

import numpy as np

from orthogonal_dfa.l_star.mask_table import MaskTable

#: Anything: the table below is never asked a membership question.
_NO_ORACLE = None


def _table(prefixes):
    return MaskTable(_NO_ORACLE, prefixes, population="uniform")


def _words(n, offset=0):
    return [bytes([(i + offset) // 256, (i + offset) % 256]) for i in range(n)]


class TestPopulations(unittest.TestCase):
    def test_the_prefixes_it_is_built_with_are_the_population_it_is_named_for(self):
        table = _table(_words(6))
        self.assertEqual(list(table.population_masks()), ["uniform"])
        self.assertEqual(int(table.population_masks()["uniform"].sum()), 6)

    def test_each_population_gets_its_own_mask(self):
        uniform, state = _words(4), _words(3, offset=4)
        table = _table(uniform)
        table.add_prefixes(state, population=("state", 2))

        masks = table.population_masks()
        self.assertEqual(int(masks["uniform"].sum()), 4)
        self.assertEqual(int(masks[("state", 2)].sum()), 3)

    def test_a_prefix_belongs_to_every_population_that_holds_it(self):
        # Drawn uniformly *and* found to reach state 0: evidence about both.
        shared, only_state = _words(5), _words(3, offset=5)
        table = _table(shared)
        table.add_prefixes(shared + only_state, population=("state", 0))

        masks = table.population_masks()
        self.assertEqual(int(masks["uniform"].sum()), 5)
        self.assertEqual(int(masks[("state", 0)].sum()), 8)
        # Which is more than the table holds, because five are in both.
        self.assertEqual(table.num_prefixes, 8)

    def test_a_prefix_the_table_lacks_is_added_with_the_population(self):
        table = _table(_words(4))
        table.add_prefixes(_words(2, offset=9), population=("state", 0))

        self.assertEqual(table.num_prefixes, 6)
        self.assertEqual(int(table.population_masks()[("state", 0)].sum()), 2)

    def test_the_representative_set_is_the_union_of_the_populations(self):
        uniform, state = _words(4), _words(3, offset=4)
        table = _table(uniform)
        table.add_prefixes(state, population=("state", 1))

        self.assertEqual(int(table.representative.sum()), 7)
        masks = table.population_masks()
        union = masks["uniform"] | masks[("state", 1)]
        self.assertEqual(int(union.sum()), int(table.representative.sum()))

    def test_dropping_a_population_narrows_the_representative_set(self):
        uniform, state = _words(4), _words(3, offset=4)
        table = _table(uniform)
        table.add_prefixes(state, population=("state", 1))
        self.assertEqual(int(table.representative.sum()), 7)

        table.drop_population(("state", 1))
        self.assertEqual(list(table.population_masks()), ["uniform"])
        self.assertEqual(int(table.representative.sum()), 4)

    def test_dropping_leaves_the_prefixes_in_the_table(self):
        # They are still columns, and may still be in another population.
        uniform = _words(4)
        table = _table(uniform)
        table.add_prefixes(uniform + _words(3, offset=4), population=("state", 1))
        table.drop_population(("state", 1))

        self.assertEqual(table.num_prefixes, 7)
        self.assertEqual(int(table.population_masks()["uniform"].sum()), 4)

    def test_a_repeated_prefix_keeps_both_columns(self):
        # The initial draw is i.i.d., so a repeat is the sampler saying that
        # string is common.  The index collapses them, so the population must be
        # built from the columns, not from it -- otherwise the repeat silently
        # loses the weight it was drawn to carry.
        table = _table(_words(3) + _words(1))

        self.assertEqual(table.num_prefixes, 4)
        self.assertEqual(int(table.representative.sum()), 4)
        self.assertEqual(int(table.population_masks()["uniform"].sum()), 4)

    def test_redefining_a_population_replaces_it_rather_than_adding_to_it(self):
        # A round rebuilds its populations from what it drew.  Whatever else
        # reached the table meanwhile -- a mid-round top-up buying the family
        # search more prefixes -- is not the pool's to keep.
        uniform = _words(4)
        table = _table(uniform)
        table.add_prefixes(_words(200, offset=4), population="uniform")
        self.assertEqual(int(table.representative.sum()), 204)

        table.drop_population("uniform")
        table.add_prefixes(uniform, population="uniform")
        self.assertEqual(int(table.representative.sum()), 4)
        self.assertEqual(table.num_prefixes, 204, "the columns stay")

    def test_a_mask_is_positions_within_the_representative_set(self):
        # Not column numbers: consumers index a decision vector that was read
        # over the representative prefixes only.  Here the retired columns 4-6
        # sit between the two live populations, so the two differ.
        early, retired, late = _words(4), _words(3, offset=4), _words(2, offset=7)
        table = _table(early)
        table.add_prefixes(retired, population="scratch")
        table.add_prefixes(late, population=("state", 0))
        table.drop_population("scratch")

        masks = table.population_masks()
        self.assertEqual(table.num_prefixes, 9, "the retired columns stay")
        for label, mask in masks.items():
            self.assertEqual(len(mask), 6, f"{label} is over the 6 representative")
        self.assertEqual(list(np.flatnonzero(masks["uniform"])), [0, 1, 2, 3])
        # Columns 7 and 8, but positions 4 and 5, the gap having closed up.
        self.assertEqual(list(np.flatnonzero(masks[("state", 0)])), [4, 5])

    def test_dropping_one_that_was_never_there_is_not_an_error(self):
        table = _table(_words(4))
        table.drop_population(("state", 3))

        self.assertEqual(list(table.population_masks()), ["uniform"])


if __name__ == "__main__":
    unittest.main()
