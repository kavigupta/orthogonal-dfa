"""How :class:`TransitionResolver` walks a probe from its start, what it does
where the walk is blocked or the walk and the sift part, and how it reads fresh
draws.

Driven by stubs rather than synthesis: these are properties of the walk, and the
end-to-end targets that depend on them are noisy enough that a regression shows
up as a state count that also moves for unrelated reasons.
"""

# The tests exercise the walk internals directly.
# pylint: disable=protected-access

import unittest
from collections import deque
from types import SimpleNamespace

from orthogonal_dfa.l_star.transition_resolver import TransitionResolver

_PROBE = bytes([0, 1, 0, 1])


class _StubSifter:
    """A one-node tree placing a string by ``places``, with ``seq + b"?"`` as the
    boundary of what it cannot place, and the middle of the band at ``middle``."""

    def __init__(self, places, middle=None):
        self.places = places
        self.middle = middle
        self.searched = []

    def sift_and_boundary(self, seq):
        leaf = self.places(bytes(seq))
        return (leaf, None) if leaf is not None else (None, bytes(seq) + b"?")

    def halfway(self, seq):
        leaf = self.places(bytes(seq))
        return self.middle if leaf is None else leaf

    def disagreement(self, s, sprime, prefix):
        self.searched.append((s, sprime, prefix))


class _StubPopulation:
    def __init__(self):
        self.recorded = []

    def add(self, string, at):
        self.recorded.append((at, bytes(string)))


class _Learner(TransitionResolver):
    # pylint: disable=super-init-not-called
    def __init__(self, sifter, transitions, k):
        self.transitions = transitions
        self.sifter = sifter
        self.population = _StubPopulation()
        self.tree = SimpleNamespace(path_of=lambda s: s, num_states=2, depth=2)
        self.dfa = SimpleNamespace(transitions=transitions, witness=lambda s, c: b"")
        self.k = k
        self.pst = SimpleNamespace(fnr_limit=0.1)
        self.draws = iter(())
        self.drawn = 0

    def _draw(self):
        self.drawn += 1
        return next(self.draws)

    def _totalised(self):
        return self.transitions, []


#: State 7 steps to 8 on a 0 and stays on a 1; nothing is learned out of 8.
_OPEN_AT_8 = {7: {0: 8, 1: 7}, 8: {}}
_EVERYWHERE = {7: {0: 7, 1: 7}}


def _parting(undecided):
    """Places the whole probe at 8, the prefixes whose lengths are in
    ``undecided`` nowhere, and the rest at 7, where the walk stays."""

    def places(seq):
        if len(seq) == len(_PROBE):
            return 8
        return None if len(seq) in undecided else 7

    return places


class TestAProbeWalkedFromItsStart(unittest.TestCase):
    def test_its_start_is_not_seeded_into_the_population(self):
        learner = _Learner(_StubSifter(lambda seq: 7), _EVERYWHERE, 2)

        self.assertFalse(learner._check(_PROBE))
        self.assertEqual([], learner.population.recorded)

    def test_an_open_edge_seeds_the_prefix_before_it_where_it_sifts_there(self):
        # The walk reaches 8 after 0, 1, 0 and finds no edge out of it on a 1.
        places = lambda seq: 8 if seq == _PROBE[:3] else 7
        learner = _Learner(_StubSifter(places), _OPEN_AT_8, 1)

        self.assertFalse(learner._check(_PROBE))
        self.assertEqual([(8, _PROBE[:3])], learner.population.recorded)

    def test_a_disagreement_down_to_an_edge_is_weighed_for_a_split(self):
        sifter = _StubSifter(_parting(set()))
        learner = _Learner(sifter, _EVERYWHERE, 1)

        self.assertFalse(learner._check(_PROBE))
        self.assertEqual([(b"", _PROBE[:3], _PROBE[3:])], sifter.searched)

    def test_a_disagreement_down_to_a_triple_is_quiet(self):
        sifter = _StubSifter(_parting({3}))
        learner = _Learner(sifter, _EVERYWHERE, 1)

        self.assertFalse(learner._check(_PROBE))
        self.assertEqual([], sifter.searched)


#: Each state stays where it is; 1 accepts.
_STAYS = {0: {0: 0, 1: 0}, 1: {0: 1, 1: 1}}
#: Every state steps to 0, which rejects, so every start disagrees with a root
#: that accepts.
_TO_REJECT = {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}}


def _gate(places, transitions, *, k=2, label=True):
    """A learner reading ``_PROBE`` again and again at its gate, the root reading
    every draw as ``label``."""
    learner = _Learner(_StubSifter(places), transitions, k)
    learner.tree.accepting_leaves = lambda: {1}
    learner.family = SimpleNamespace(middle_side=lambda seq, midfix: label)
    learner.window = deque()
    learner.draws = iter([_PROBE] * 4000)
    return learner


def _disagreeing(undecided):
    """Places the whole probe at 1, the prefixes whose lengths are in
    ``undecided`` nowhere, and the rest at 0, where a walk along ``_TO_REJECT``
    stays."""

    def places(seq):
        if len(seq) == len(_PROBE):
            return 1
        return None if len(seq) in undecided else 0

    return places


class TestReadingFreshDraws(unittest.TestCase):
    def test_the_start_agreeing_on_most_draws_is_the_readings(self):
        learner = _gate(lambda seq: 0, _STAYS)

        reading = learner.read_fresh(acc_threshold=0.9)

        self.assertEqual((1, 1.0), (reading.start, reading.agreement))
        # Only the first draw, before start 1 led, was read from k.
        self.assertEqual([], reading.disagreements)

    def test_a_draw_the_best_start_disagrees_on_is_read_from_k(self):
        reading = _gate(_disagreeing(set()), _TO_REJECT).read_fresh(acc_threshold=0.9)

        self.assertEqual(0.0, reading.agreement)
        self.assertEqual(_PROBE, reading.disagreements[0])
        self.assertEqual(([], 0), (reading.triples, reading.pairs))

    def test_a_triple_leaves_its_middles_boundary_string(self):
        reading = _gate(_disagreeing({3}), _TO_REJECT).read_fresh(acc_threshold=0.9)

        self.assertEqual([_PROBE[:3] + b"?"], reading.triples)

    def test_a_pair_leaves_nothing_but_is_counted(self):
        learner = _gate(_disagreeing({2, 3}), _TO_REJECT, k=1)
        reading = learner.read_fresh(acc_threshold=0.9)

        self.assertEqual([], reading.triples)
        self.assertEqual(len(reading.disagreements), reading.pairs)

    def test_reading_stops_once_both_its_tests_settle(self):
        learner = _gate(lambda seq: 0, _STAYS)
        learner.read_fresh(acc_threshold=0.5)

        # The tests are read at 30, 60, 120, ... draws.  The agreement settles at
        # the first; no whole sift cut short below the root settles under the
        # stub's (2 - 1) * 0.1 only by the third, since 0.9 ** 60 > 1e-3 >
        # 0.9 ** 120.
        self.assertEqual(120, learner.drawn)

    def test_wholes_cut_short_below_the_root_too_often_are_kept_by_midfix(self):
        places = lambda seq: None if seq == _PROBE else 0
        reading = _gate(places, _STAYS).read_fresh(acc_threshold=0.9)

        self.assertEqual([("end", b"?")], reading.ends)

    def test_the_passs_starts_cut_short_too_often_are_kept_by_midfix(self):
        learner = _gate(lambda seq: 0, _STAYS)
        learner.window = deque([b"?"] * 20 + [None] * 129)

        self.assertEqual([("start", b"?")], learner.read_fresh(acc_threshold=0.9).ends)

    def test_a_replay_reads_a_fresh_draw_as_the_gate_did(self):
        learner = _gate(_disagreeing({3}), _TO_REJECT)
        reading = learner.read_fresh(acc_threshold=0.9)
        learner.draws = iter([_PROBE])

        self.assertEqual([_PROBE[:3] + b"?"], learner.replay(reading))


if __name__ == "__main__":
    unittest.main()
