"""Which population may admit a suffix family, and which may veto it.

Admitting is the claim that the family separates the classes at all, and only a
sample drawn the way the oracle's own base rate is can carry it.  Vetoing is the
claim that some population is read as the class it is not, and every population
carries that -- which is why the rate is kept per population in the first place.
"""

import unittest
from types import SimpleNamespace

from orthogonal_dfa.l_star.cluster import ADMITTED, DRIFTED, UNCERTIFIED, drift_verdict
from orthogonal_dfa.l_star.mask_table import UNIFORM

#: The thresholds a family is read with, and so the ones the split is held to.
ACCEPT, REJECT = 0.671, 0.333

_PST = SimpleNamespace(accept_thresh=ACCEPT, reject_thresh=REJECT)

#: A pool the family reads right: 900 of 1000 it calls accepting are, and 100 of
#: 1000 it calls rejecting are.  Large enough for both sides to clear their nulls.
_CLEAN_POOL = ((900, 1000), (100, 1000))


def _verdict(**by_population):
    return drift_verdict(_PST, {UNIFORM: _CLEAN_POOL, **by_population})


class TestOnlyThePoolAdmits(unittest.TestCase):
    def test_a_pool_that_separates_the_classes_is_admitted(self):
        self.assertEqual(_verdict(), ADMITTED)

    def test_a_pool_too_small_to_clear_its_null_says_nothing_yet(self):
        # The same rates over ten prefixes a side: right, and unprovable.
        self.assertEqual(
            drift_verdict(_PST, {UNIFORM: ((9, 10), (1, 10))}), UNCERTIFIED
        )

    def test_without_a_pool_nothing_can_admit(self):
        # A state's prefixes are one class, so they cannot say whether the
        # family separates two.
        self.assertEqual(
            drift_verdict(_PST, {("state", 0): ((190, 200), (0, 0))}), UNCERTIFIED
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


if __name__ == "__main__":
    unittest.main()
