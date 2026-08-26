"""The round check against a family that is not accept-preserving.

``identify_cluster_around`` can settle on the signature class next to the
accept-preserving one -- agreeing with it everywhere but the state a symbol short
of accepting -- while still containing the empty suffix.  Such a family is wrong
and perfectly decisive, so the FNR gate passes it.
"""

import unittest

from tests.test_near_miss_accept_preserving import learn_with_near_misses

#: Puts the near-accept state's prefix count just above the floor the round check
#: holds a state to.  Below it the state is excused; well above it the family
#: calibrates against those prefixes and cuts them correctly regardless.
SHARE = 0.3

#: Without the requirement 21% of seeds fail on their own, which makes this many a
#: 94% event.  Which seeds fail is a property of the suffixes a round draws.
SEEDS = range(12)


class TestAcceptPreservingFamily(unittest.TestCase):
    def test_a_drifted_family_does_not_reach_the_round_check(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                learn_with_near_misses(seed, share=SHARE)


if __name__ == "__main__":
    unittest.main()
