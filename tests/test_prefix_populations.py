"""The prefix populations the FNR limit is stated over.

The learner reads its rate over several populations of prefixes at once -- the
uniform pool, the harvested boundary strings, and one per state -- and holds each
to the limit on its own.  These are the properties that makes rest on.
"""

import unittest

import numpy as np

from orthogonal_dfa.l_star.mask_table import UNIFORM, MaskTable
from orthogonal_dfa.l_star.prefix_suffix_tracker import PrefixSuffixTracker

#: Anything: the table below is never asked a membership question.
_NO_ORACLE = None
#: The table is built with a population and then scoped to the test's own.
_SCRATCH = "scratch"


def _table(prefixes, populations):
    """A table holding ``prefixes``, scoped to ``label -> its members``."""
    table = MaskTable(_NO_ORACLE, prefixes, population=_SCRATCH)
    for label, members in populations.items():
        table.add_prefixes(members, population=label)
    table.drop_population(_SCRATCH)
    return table


def _tracker(table, *, boundary=0.5, margin=0.1):
    """A tracker holding ``table``, enough of one to read a decision vector."""
    pst = PrefixSuffixTracker.__new__(PrefixSuffixTracker)
    pst.table = table
    pst.decision_boundary = boundary
    pst.evidence_margin = margin
    return pst


def _words(n, offset=0):
    return [bytes([(i + offset) // 256, (i + offset) % 256]) for i in range(n)]


class TestTheRateIsPerPopulation(unittest.TestCase):
    def test_a_small_population_is_not_averaged_away(self):
        # 100 decisive prefixes and 10 indecisive ones: 9% across the union, but
        # the ten are a population of their own and every one of them straddles.
        decisive, straddling = _words(100), _words(10, offset=100)
        table = _table(
            decisive + straddling,
            {UNIFORM: decisive, ("state", 0): straddling},
        )
        pst = _tracker(table)
        # Both classes present, or the family reads as uninformative whatever
        # the populations say.
        decision = np.array([0.9] * 50 + [0.1] * 50 + [0.5] * 10)

        self.assertAlmostEqual(
            float(np.mean((decision >= 0.4) & (decision < 0.6))), 10 / 110
        )
        rate, worst = pst.fnr_from_decision(decision)
        self.assertEqual((rate, worst), (1.0, ("state", 0)))

    def test_the_rate_names_the_population_it_belongs_to(self):
        a, b = _words(20), _words(20, offset=20)
        table = _table(a + b, {UNIFORM: a, ("state", 3): b})
        pst = _tracker(table)
        # A fifth of ``b`` straddles and none of ``a`` does.
        decision = np.array([0.9] * 10 + [0.1] * 10 + [0.5] * 4 + [0.9] * 8 + [0.1] * 8)

        rate, worst = pst.fnr_from_decision(decision)
        self.assertEqual(worst, ("state", 3))
        self.assertAlmostEqual(rate, 0.2)


if __name__ == "__main__":
    unittest.main()
