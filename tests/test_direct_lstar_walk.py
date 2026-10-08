"""How :class:`TransitionResolver` walks a probe from its start, what it does
where the walk is blocked or a split attempt cannot go on, and the frozen
hypothesis's reading of the gate's draws.

Driven by stubs rather than synthesis: these are properties of the walk, and the
end-to-end targets that depend on them are noisy enough that a regression shows
up as a state count that also moves for unrelated reasons.
"""

# The tests exercise the walk internals directly.
# pylint: disable=protected-access

import unittest
from collections import deque
from types import SimpleNamespace

import numpy as np

from orthogonal_dfa.l_star.lstar import estimate_agreement_rate
from orthogonal_dfa.l_star.provenance import Read
from orthogonal_dfa.l_star.transition_resolver import (
    _RESOLVED,
    _UNCHECKED,
    CHECK,
    WALK,
    FrozenCheck,
    TransitionResolver,
)

_PROBE = bytes([0, 1, 0, 1])


class _StubSifter:
    """Places a string by ``places``, with ``seq + b"?"`` as the boundary of what
    it cannot place, and the middle of the band at ``middle``."""

    def __init__(self, places, middle=None):
        self.places = places
        self.middle = middle
        self.reads = 0
        self.undecided_search = None

    def sift_and_boundary(self, seq):
        leaf = self.places(bytes(seq))
        return (leaf, None) if leaf is not None else (None, bytes(seq) + b"?")

    def known_sift(self, seq):
        return self.places(bytes(seq))

    def halfway(self, _seq):
        return self.middle, []

    def prefill(self, seqs):
        pass

    def disagreement(self, _s, _sprime, _prefix):
        return None, self.undecided_search


class _StubPopulation:
    def __init__(self):
        self.recorded = []

    def add(self, string, at, **_draw):
        self.recorded.append((at, bytes(string)))

    def add_first(self, string, at, **_draw):
        self.recorded.append((at, bytes(string)))


class _Learner(TransitionResolver):
    # pylint: disable=super-init-not-called
    def __init__(self, sifter, transitions, k, *, witness=b""):
        self.sifter = sifter
        self.population = _StubPopulation()
        self.tree = SimpleNamespace(path_of=lambda s: s, num_states=2)
        self.dfa = SimpleNamespace(
            transitions=transitions, witness=lambda s, c: witness
        )
        self.k = k
        self.indecisive = {}
        self.dropped = {}
        self.recent = deque()
        self._walked = Read(None, None)
        self.pst = None


#: State 7 steps to 8 on a 0 and stays on a 1; nothing is learned out of 8.
_OPEN_AT_8 = {7: {0: 8, 1: 7}, 8: {}}
_EVERYWHERE = {7: {0: 7, 1: 7}}


class TestAProbeWalkedFromItsStart(unittest.TestCase):
    def test_it_seeds_the_start_at_the_leaf_it_sifts_to(self):
        learner = _Learner(_StubSifter(lambda seq: 7), _EVERYWHERE, 2)

        self.assertEqual(_RESOLVED, learner._check(_PROBE))
        self.assertEqual([(7, _PROBE[:2])], learner.population.recorded)

    def test_a_start_the_cut_cannot_place_is_unchecked_and_harvested(self):
        learner = _Learner(_StubSifter(lambda seq: None), _EVERYWHERE, 2)

        self.assertEqual(_UNCHECKED, learner._check(_PROBE))
        self.assertEqual({_PROBE[:2] + b"?"}, set(learner.indecisive))
        self.assertEqual([], learner.population.recorded)

    def test_an_open_edge_seeds_the_prefix_before_it_where_it_sifts_there(self):
        # The walk reaches 8 after 0, 1, 0 and finds no edge out of it on a 1.
        places = lambda seq: 8 if seq == _PROBE[:3] else 7
        learner = _Learner(_StubSifter(places), _OPEN_AT_8, 1)

        self.assertEqual(_RESOLVED, learner._check(_PROBE))
        self.assertIn((8, _PROBE[:3]), learner.population.recorded)

    def test_a_disagreement_whose_prefix_the_cut_cannot_place_is_dropped(self):
        # The walk stays at 7 while the whole probe sifts to 8, and the prefix the
        # search lands on can only be placed by the middle.
        def places(seq):
            if len(seq) == 4:
                return 8
            return None if len(seq) == 3 else 7

        learner = _Learner(_StubSifter(places, middle=7), _EVERYWHERE, 1)

        self.assertEqual(_RESOLVED, learner._check(_PROBE))
        self.assertEqual({_PROBE[:3] + b"?": None}, learner.dropped)

    def test_a_distinguisher_search_the_cut_cannot_finish_is_dropped(self):
        sifter = _StubSifter(lambda seq: 8 if len(seq) == 4 else 7)
        sifter.undecided_search = b"stuck"
        learner = _Learner(sifter, _EVERYWHERE, 1)

        self.assertEqual(_RESOLVED, learner._check(_PROBE))
        self.assertEqual({b"stuck": None}, learner.dropped)


class TestTheExportedStart(unittest.TestCase):
    def test_it_is_the_state_whose_runs_accept_where_the_root_does(self):
        # From 0 a 1 accepts; from 1 everything accepts.  The root reads probes
        # ending in 1 as accepting and the rest as not, which only 0 matches.
        learner = _Learner(_StubSifter(lambda seq: 0), {}, 1)
        learner.tree.num_states = 2
        learner.family = SimpleNamespace(middle_side=lambda seq, midfix: seq[-1] == 1)
        learner.recent = deque([bytes([0, 1]), bytes([1, 0]), bytes([0, 0])])
        transitions = {0: {0: 0, 1: 1}, 1: {0: 0, 1: 1}}

        self.assertEqual(0, learner._best_start(transitions, accepting={1}))

    def test_a_tie_goes_to_the_lowest_state(self):
        learner = _Learner(_StubSifter(lambda seq: 0), {}, 1)
        learner.family = SimpleNamespace(middle_side=lambda seq, midfix: None)
        learner.recent = deque([bytes([0])])

        self.assertEqual(0, learner._best_start({0: {0: 1}, 1: {0: 0}}, {1}))


class _Draws:
    """A sampler cycling through ``draws``."""

    def __init__(self, draws):
        self._draws = draws
        self.drawn = 0

    def sample(self, _rng, _alphabet_size):
        draw = self._draws[self.drawn % len(self._draws)]
        self.drawn += 1
        return draw


def _frozen(places, transitions, k, *, acc_threshold=0.9):
    learner = _Learner(_StubSifter(places), transitions, k)
    learner.dfa.transitions = transitions
    learner.pst = SimpleNamespace(rng=None, alphabet_size=2)
    return FrozenCheck(learner, acc_threshold=acc_threshold)


class TestTheFrozenHypothesisReadingTheGatesDraws(unittest.TestCase):
    def test_blocked_draws_count_against_their_rates_and_not_the_agreement(self):
        places = lambda seq: None if seq == _PROBE else 7
        check = _frozen(places, _EVERYWHERE, 2)

        check.observe(_PROBE)
        check.observe(_PROBE[:3])

        self.assertEqual((0, 2), (check.walk_blocked.hits, check.walk_blocked.draws))
        self.assertEqual((1, 2), (check.blocked.hits, check.blocked.draws))
        self.assertEqual((1, 1), (check.agreement.hits, check.agreement.draws))

    def test_a_decided_disagreement_is_kept_for_the_pass(self):
        check = _frozen(lambda seq: 8 if len(seq) == 4 else 7, _EVERYWHERE, 2)

        check.observe(_PROBE)

        self.assertEqual([_PROBE], check.disagreements)
        self.assertEqual((0, 1), (check.agreement.hits, check.agreement.draws))

    def test_a_walk_blocked_round_leaves_the_walks_finds(self):
        check = _frozen(lambda seq: None, _EVERYWHERE, 2)
        sampler = _Draws([_PROBE, _PROBE[::-1]])
        pst = SimpleNamespace(sampler=sampler, rng=None, alphabet_size=2)

        estimate_agreement_rate(pst, check, num_samples=2000)

        self.assertTrue(check.blocks)
        blocked = check.outcome()[0]
        self.assertEqual(WALK, blocked.kind)
        self.assertEqual(
            {_PROBE[:2] + b"?", _PROBE[::-1][:2] + b"?"}, set(blocked.found)
        )

    def test_a_sift_blocked_round_leaves_its_finds_with_the_passs_drops(self):
        check = _frozen(lambda seq: None if len(seq) == 4 else 7, _EVERYWHERE, 2)
        check._resolver.dropped = {b"dropped": None}
        pst = SimpleNamespace(sampler=_Draws([_PROBE]), rng=None, alphabet_size=2)

        estimate_agreement_rate(pst, check, num_samples=2000)

        blocked = check.outcome()[0]
        self.assertEqual(CHECK, blocked.kind)
        self.assertEqual({_PROBE + b"?", b"dropped"}, set(blocked.found))

    def test_the_passs_drops_are_held_though_nothing_blocks(self):
        check = _frozen(lambda seq: 7, _EVERYWHERE, 2)
        check._resolver.dropped = {b"dropped": None}
        pst = SimpleNamespace(sampler=_Draws([_PROBE]), rng=None, alphabet_size=2)

        self.assertEqual(1.0, estimate_agreement_rate(pst, check, num_samples=2000))

        self.assertFalse(check.blocks)
        blocked = check.outcome()[0]
        self.assertEqual((CHECK, [b"dropped"]), (blocked.kind, blocked.found))

    def test_the_draws_stop_once_every_rate_is_settled(self):
        check = _frozen(lambda seq: 7, _EVERYWHERE, 2, acc_threshold=0.5)
        sampler = _Draws([_PROBE])
        pst = SimpleNamespace(sampler=sampler, rng=None, alphabet_size=2)

        estimate_agreement_rate(pst, check, num_samples=2000)

        self.assertTrue(all(rate.side is not None for rate in check.rates))
        # Clean draws settle the blocked rates below 0.5 at the 17th, since
        # 0.5 ** 17 < 1e-5, and the agreement at its least, the 30th.
        self.assertEqual(17, int(np.ceil(np.log(1e-5) / np.log(0.5))))
        self.assertEqual(30, sampler.drawn)


if __name__ == "__main__":
    unittest.main()
