"""Which population may admit a suffix family, and which may veto it.

Admitting is the claim that the family separates the classes at all, and only a
sample drawn the way the oracle's own base rate is can carry it.  Vetoing is the
claim that some population is read as the class it is not, and every population
carries that -- which is why the rate is kept per population in the first place.
"""

import itertools
import unittest
from types import SimpleNamespace

from orthogonal_dfa.l_star.cluster import (
    ACCEPT_PRESERVING_ERROR_RATE,
    ADMITTED,
    DRIFTED,
    UNCERTIFIED,
    drift_verdict,
    prefixes_to_certify,
    veto_size,
)
from orthogonal_dfa.l_star.mask_table import UNIFORM
from orthogonal_dfa.l_star.statistics import binom_cdf

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

    def test_without_a_pool_nothing_can_admit(self):
        # A state's prefixes are one class, so they cannot say whether the
        # family separates two.
        self.assertEqual(
            drift_verdict(_PST, {("state", 0): ((190, 200), (0, 0))})[0], UNCERTIFIED
        )


class TestAnyPopulationVetoes(unittest.TestCase):
    """A state's prefixes all reach one state, so they are all one class and land
    all on one side.  That is the ordinary case, not a fault."""

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


def _vetoes(hits, n):
    """Whether the gate calls a state population of ``n`` reading ``hits`` drifted,
    against a pool it is happy with."""
    backwards = {("state", 0): ((hits, n), (0, 0))}
    return drift_verdict(_PST, {UNIFORM: _CLEAN_POOL, **backwards})[0] == DRIFTED


def _caught(n):
    """Chance the gate vetoes a population of ``n`` the family has inverted, which
    the oracle reads at ``REJECT`` rather than at nothing."""
    return sum(
        binom_cdf(hits, n, REJECT) - binom_cdf(hits - 1, n, REJECT)
        for hits in range(n + 1)
        if _vetoes(hits, n)
    )


def _smallest_that_can_fire():
    """Fewest prefixes at which a population reading as nothing but the other
    class vetoes: what sizing on the level alone buys."""
    return next(n for n in itertools.count(1) if _vetoes(0, n))


class TestWhatAVetoCosts(unittest.TestCase):
    """`veto_size` is what a population has to hold for a backwards reading to be
    caught, which is more than it has to hold to fire at all: an inverted
    population is read at ``reject_thresh``, not at nothing."""

    def test_an_inverted_population_is_caught_all_but_alpha_of_the_time(self):
        self.assertGreaterEqual(
            _caught(veto_size(_PST, 2)), 1 - ACCEPT_PRESERVING_ERROR_RATE
        )

    def test_a_size_that_can_only_fire_misses_more_than_that(self):
        firing = _smallest_that_can_fire()

        self.assertLess(_caught(firing), 1 - ACCEPT_PRESERVING_ERROR_RATE)
        self.assertGreater(veto_size(_PST, 2), firing)

    def test_a_population_reading_as_its_own_class_is_left_alone(self):
        n = veto_size(_PST, 2)

        self.assertFalse(_vetoes(round(n * ACCEPT), n))


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
