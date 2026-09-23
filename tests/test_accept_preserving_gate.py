"""Which population may admit a suffix family, and which may veto it.

Admitting is the claim that the family separates the classes at all, and only a
sample drawn the way the oracle's own base rate is can carry it.  Vetoing is the
claim that some population is read as the class it is not, which any population
can carry.
"""

import unittest
from types import SimpleNamespace

from orthogonal_dfa.l_star.cluster import (
    ADMITTED,
    DRIFTED,
    UNCERTIFIED,
    drift_verdict,
    prefixes_to_certify,
)
from orthogonal_dfa.l_star.mask_table import UNIFORM

#: The thresholds a family is read with, and so the ones the split is held to.
ACCEPT, REJECT = 0.671, 0.333

_PST = SimpleNamespace(accept_thresh=ACCEPT, reject_thresh=REJECT)

#: A pool the family reads right: 900 of 1000 it calls accepting are, and 100 of
#: 1000 it calls rejecting are.  Large enough for both sides to clear their nulls.
_CLEAN_POOL = ((900, 1000), (100, 1000))


def _verdict(**by_population):
    """The verdict alone, for the cases that do not turn on who said it."""
    return drift_verdict(_PST, {UNIFORM: _CLEAN_POOL, **by_population})[0]


class TestOnlyThePoolAdmits(unittest.TestCase):
    def test_a_pool_that_separates_the_classes_is_admitted(self):
        self.assertEqual(_verdict(), ADMITTED)

    def test_a_pool_too_small_to_clear_its_null_says_nothing_yet(self):
        # The same rates over ten prefixes a side: right, and unprovable.
        self.assertEqual(
            drift_verdict(_PST, {UNIFORM: ((9, 10), (1, 10))})[0], UNCERTIFIED
        )

    def test_a_pool_with_nothing_on_one_side_can_still_admit(self):
        # The cut put every prefix it read on the accepting side, and the oracle
        # agrees about them.  There is no reject side to fail.
        self.assertEqual(
            drift_verdict(_PST, {UNIFORM: ((900, 1000), (0, 0))})[0], ADMITTED
        )

    def test_a_split_with_no_decisive_prefix_at_all_is_uncertified(self):
        # Every prefix read in the undecided band, so neither side holds one.
        self.assertEqual(
            drift_verdict(_PST, {UNIFORM: ((0, 0), (0, 0))}), (UNCERTIFIED, UNIFORM)
        )

    def test_without_a_pool_nothing_can_admit(self):
        # A state's prefixes are one class, so they cannot say whether the
        # family separates two.
        self.assertEqual(
            drift_verdict(_PST, {("state", 0): ((190, 200), (0, 0))})[0], UNCERTIFIED
        )


class TestAnyPopulationVetoes(unittest.TestCase):
    """A state's prefixes all reach one state, so they are all one class and land
    all on one side."""

    def test_a_one_sided_population_read_right_does_not_block(self):
        # 200 accepting prefixes the family also calls accepting.
        self.assertEqual(_verdict(**{"one_sided": ((190, 200), (0, 0))}), ADMITTED)

    def test_a_one_sided_population_read_backwards_vetoes(self):
        # The same 200 accepting prefixes, called rejecting: the oracle accepts
        # 95% of a side the family says is under 33%.
        self.assertEqual(_verdict(**{"one_sided": ((0, 0), (190, 200))}), DRIFTED)

    def test_a_population_too_small_to_veto_leaves_the_pool_to_it(self):
        # Three prefixes read backwards cannot reject anything at this level, so
        # the pool's own reading stands.
        self.assertEqual(_verdict(**{"tiny": ((0, 0), (3, 3))}), ADMITTED)

    def test_a_veto_outranks_a_pool_that_would_admit(self):
        # The pool separates the classes and one state is still inverted, which
        # is the case the per-population rate exists to catch.
        self.assertEqual(
            _verdict(**{"a": ((190, 200), (0, 0)), "b": ((0, 0), (190, 200))}),
            DRIFTED,
        )


class TestWhatTheTopUpAssumes(unittest.TestCase):
    def test_only_the_pool_is_scaled(self):
        # Scaling a population nobody draws from asks what a draw that is
        # never made would say, and cuts the pool's top-up to a fraction of what
        # it needs.
        pool = ((75, 100), (30, 100))
        pst = SimpleNamespace(
            accept_thresh=ACCEPT,
            reject_thresh=REJECT,
            table=SimpleNamespace(fully_observed=lambda: range(8)),
            config=SimpleNamespace(num_addtl_prefixes=2000),
        )

        wanted = prefixes_to_certify(
            pst, {UNIFORM: pool, ("state", 0): ((1, 6), (0, 0))}, 200, range(8)
        )

        self.assertEqual(
            wanted, prefixes_to_certify(pst, {UNIFORM: pool}, 200, range(8))
        )


class TestARefusalNamesAPopulation(unittest.TestCase):
    """What the search grows to answer the refusal."""

    def test_a_veto_names_the_population_read_backwards(self):
        verdict, blamed = drift_verdict(
            _PST, {UNIFORM: _CLEAN_POOL, ("state", 3): ((0, 0), (190, 200))}
        )

        self.assertEqual((DRIFTED, ("state", 3)), (verdict, blamed))

    def test_a_split_that_cannot_be_read_names_the_pool(self):
        # Only the pool can admit, so it is the one worth growing.
        verdict, blamed = drift_verdict(_PST, {UNIFORM: ((9, 10), (1, 10))})

        self.assertEqual((UNCERTIFIED, UNIFORM), (verdict, blamed))

    def test_an_admitted_family_blames_nobody(self):
        self.assertEqual((ADMITTED, None), drift_verdict(_PST, {UNIFORM: _CLEAN_POOL}))


if __name__ == "__main__":
    unittest.main()
