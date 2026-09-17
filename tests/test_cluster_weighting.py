"""What a prefix counts for when `identify_cluster_around` ranks suffixes.

A suffix is picked for agreeing with the seed on the representative prefixes.
Counted one prefix at a time, a population of four is four prefixes against a
pool of a hundred, so a suffix that reads a whole state backwards ranks above
one that reads a tenth of the pool that way.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from orthogonal_dfa.l_star.cluster import identify_cluster_around
from orthogonal_dfa.l_star.mask_table import UNIFORM
from tests.cluster_stubs import Table

POOL, STATE = 100, 4
#: Where the pool's columns end and the state's begin.
SPLIT = POOL


def _populations():
    """``POOL`` uniform prefixes, then ``STATE`` resting at one state."""
    pool = np.zeros(POOL + STATE, dtype=bool)
    pool[:SPLIT] = True
    return {UNIFORM: pool, ("state", 0): ~pool}


def _masks():
    """Three suffixes: the seed, one disagreeing over a tenth of the pool, and
    one disagreeing over the whole state."""
    masks = np.ones((3, POOL + STATE), dtype=np.int8)
    masks[1, :10] = 0
    masks[2, SPLIT:] = 0
    return masks


class TestEveryPopulationWeighsTheSame(unittest.TestCase):
    def test_the_state_read_backwards_loses_to_a_tenth_of_the_pool(self):
        pst = SimpleNamespace(
            table=Table(_masks(), _populations()),
            config=SimpleNamespace(min_signal_strength=0.3),
        )

        family, _ = identify_cluster_around(pst, 0, 2, 0.5)

        self.assertEqual(sorted(family), [0, 1])


if __name__ == "__main__":
    unittest.main()
