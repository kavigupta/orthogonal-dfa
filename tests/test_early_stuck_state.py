"""A state the family cannot place one symbol in, ahead of the rest of the probe.

Every walk that started at the empty string and disagreed from its first edge
would bisect down to that state and stop there, so a replay of the pass's walks
would never find a string it has not already seen.  Walks start past it.
"""

# pylint: disable=protected-access

import unittest
from types import SimpleNamespace

import numpy as np

from orthogonal_dfa.l_star.counterexample_synthesis import _boundary_source
from orthogonal_dfa.l_star.prefix_populations import PoolState
from orthogonal_dfa.l_star.prefix_sources import UniformSource
from orthogonal_dfa.l_star.provenance import Read
from orthogonal_dfa.l_star.sampler import UniformSampler

_ALPHABET = 4
_LENGTH = 16

#: The walk steps to 1 from anywhere while the tree places every prefix at 0, so
#: every walk disagrees from its first edge on.
_TRANSITIONS = {s: {c: 1 for c in range(_ALPHABET)} for s in (0, 1)}


class _Sifter:
    """Cannot place the one-symbol prefixes, nor those 12 to 14 symbols long.

    Bisecting a whole probe from the empty string asks about lengths 8, 4, 2
    and 1 only, so the later ones are met only by a walk anchored past the
    first.
    """

    def sift_and_boundary(self, seq):
        if len(seq) == 1 or 12 <= len(seq) <= 14:
            return None, bytes(seq) + b"?"
        return 0, None

    def halfway(self, _seq):
        return 0, []

    def prefill(self, seqs):
        pass


def _pst():
    return SimpleNamespace(
        sampler=UniformSampler(_LENGTH),
        rng=np.random.default_rng(0),
        alphabet_size=_ALPHABET,
    )


class TestReplayingTheWalksDoesNotRunDry(unittest.TestCase):
    def test_the_boundary_source_draws_as_many_strings_as_it_is_asked(self):
        pst = _pst()
        state = PoolState([])
        state.take(b"\x00?", Read(UniformSource(pst), None))
        _boundary_source(
            pst,
            SimpleNamespace(sifter=_Sifter(), learned=lambda: _TRANSITIONS, k=8),
            state,
            acc_threshold=0.9,
        )
        source = state.sources[state.harvesting]

        self.assertEqual(50, len({source.draw() for _ in range(50)}))


if __name__ == "__main__":
    unittest.main()
