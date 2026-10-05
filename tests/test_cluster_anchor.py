"""A pool where most suffixes shift the residue of a count mod 9, two shifts
outnumbering the suffixes that keep it: the family must lead with the keepers,
which alone read like the seed, however many suffixes agree with each other."""

import unittest

import numpy as np

from orthogonal_dfa.l_star.cluster import identify_cluster_around
from tests.lstar_common import cluster_pst

PREFIXES = 1200
KEEPERS = 230
#: Each shift misreads two of the nine residues, and both outnumber the keepers.
SHIFTED = {3: 400, 6: 400}
#: One-sided reads: a rejecting string reads 1 half the time.
RATES = (0.5, 0.8)
SIGNAL = 0.15
#: Where a family half keepers and half shifted puts the boundary, rather than
#: midway between the rates.
BOUNDARY = 0.6


def _pool(seed):
    """Reads with the seed first, and each row's shift."""
    rng = np.random.default_rng(seed)
    residue = np.arange(PREFIXES) % 9
    shifts = np.array(
        [0] * (KEEPERS + 1) + [c for c, n in SHIFTED.items() for _ in range(n)]
    )
    accepts = np.isin((residue[None, :] + shifts[:, None]) % 9, (3, 6))
    reads = rng.random(accepts.shape) < np.where(accepts, RATES[1], RATES[0])
    return reads.astype(np.int8), shifts


class TestClusterAnchor(unittest.TestCase):
    def test_the_family_leads_with_the_keepers(self):
        reads, shifts = _pool(0)
        vs, _ = identify_cluster_around(
            cluster_pst(reads, SIGNAL, seed=0), 0, 260, BOUNDARY
        )
        # The family asks for more than the keepers, so the rest pad it after them.
        self.assertGreater(np.mean(shifts[vs[: KEEPERS + 1]] == 0), 0.9)


if __name__ == "__main__":
    unittest.main()
