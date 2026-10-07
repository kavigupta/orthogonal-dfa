"""A state the family cannot place one symbol in, ahead of the rest of the probe.

Every walk that starts at the empty string and disagrees from its first edge
bisects down to that state and stops there, so neither the counterexample pass
nor a replay of its walks ever reads the rest of the probe.
"""

# The tests drive the pass with stubs in place of the resolver's parts.
# pylint: disable=protected-access,super-init-not-called

import unittest
from types import SimpleNamespace

import numpy as np

from orthogonal_dfa.l_star.counterexample_synthesis import _top_up_boundary
from orthogonal_dfa.l_star.prefix_populations import PoolState
from orthogonal_dfa.l_star.provenance import Read
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.transition_resolver import TransitionResolver

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

    def prefill(self, seqs):
        pass


def _pst():
    return SimpleNamespace(
        sampler=UniformSampler(_LENGTH),
        rng=np.random.default_rng(0),
        alphabet_size=_ALPHABET,
    )


class _Resolver(TransitionResolver):
    def __init__(self):
        self.pst = _pst()
        self.sifter = _Sifter()
        self.indecisive = {}
        self._walked = Read(None, None)
        self.tree = SimpleNamespace(num_states=2, path_of=lambda s: s)
        self.population = SimpleNamespace(add=lambda *_, **__: None)
        self.dfa = SimpleNamespace(target=lambda s, c: None)
        self.edges = SimpleNamespace(close=lambda: None)

    def _total_delta(self):
        return _TRANSITIONS


class TestThePassReadsPastAnEarlyStateItCannotPlace(unittest.TestCase):
    def test_the_pass_meets_boundary_strings_after_the_first_symbol(self):
        resolver = _Resolver()
        resolver.counterexample_pass(max_probes=200, patience=200)

        self.assertTrue(
            any(len(string) > 2 for string in resolver.indecisive),
            f"only met {sorted(resolver.indecisive)}",
        )


class TestReplayingTheWalksDoesNotRunDry(unittest.TestCase):
    def test_a_top_up_draws_as_many_boundary_strings_as_it_wants(self):
        pst = _pst()
        state = PoolState([])
        _top_up_boundary(
            pst,
            SimpleNamespace(sifter=_Sifter()),
            SimpleNamespace(transitions=_TRANSITIONS),
            state,
            50,
        )

        self.assertEqual(50, len(state.held[state.harvesting]))


if __name__ == "__main__":
    unittest.main()
