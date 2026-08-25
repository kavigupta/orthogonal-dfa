"""The round check against a family that has drifted out of epsilon's class.

``identify_cluster_around`` can settle on a signature class next to epsilon's --
one agreeing with it everywhere but the state a symbol short of accepting -- while
still containing epsilon.  Such a family is wrong but perfectly decisive, so the
FNR gate, which measures indecision, passes it.
"""

import unittest

from orthogonal_dfa.l_star.structures import AsymmetricBernoulli
from tests.lstar_common import learn_dfa_verified
from tests.test_near_miss_accept_preserving import NearMissSampler, _oracle

#: Puts the near-accept state's prefix count just above the floor the round check
#: holds a state to.  Below it the state is excused; far above it the family
#: calibrates against those prefixes and cuts them correctly with or without the
#: audit, so neither end of the range tests anything.
SHARE = 0.3

#: Without the audit 21% of seeds fail individually, so this many makes the
#: regression a 94% event.  The rate is what it is: the drift is a property of
#: which suffixes the round happens to draw, not of any one seed.
SEEDS = range(12)


class TestEpsilonAudit(unittest.TestCase):
    def test_a_drifted_family_does_not_reach_the_round_check(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                learn_dfa_verified(
                    _oracle,
                    min_signal_strength=0.2,
                    seed=seed,
                    noise_model=AsymmetricBernoulli(p_0=0.15, p_1=0.7),
                    sampler=NearMissSampler(share=SHARE),
                )


if __name__ == "__main__":
    unittest.main()
