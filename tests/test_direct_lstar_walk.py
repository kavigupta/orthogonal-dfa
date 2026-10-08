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


def _read(places, *, k=2, middle=7, acc_threshold=0.9):
    return _reader(places, k=k, middle=middle).read_fresh(acc_threshold=acc_threshold)


def _reader(places, *, k=2, middle=7):
    learner = _Learner(_StubSifter(places, middle=middle), _EVERYWHERE, k)
    learner.draws = iter([_PROBE] * 2000)
    return learner


class TestReadingFreshDraws(unittest.TestCase):
    def test_a_decided_disagreement_is_kept_for_the_pass(self):
        reading = _read(_parting(set()))

        self.assertEqual(0.0, reading.agreement)
        self.assertEqual(_PROBE, reading.disagreements[0])
        self.assertEqual(([], 0), (reading.triples, reading.pairs))

    def test_a_triple_leaves_its_middles_boundary_string(self):
        reading = _read(_parting({3}))

        self.assertEqual([_PROBE[:3] + b"?"], reading.triples)

    def test_a_pair_leaves_nothing_but_is_counted(self):
        reading = _read(_parting({2, 3}), k=1)

        self.assertEqual([], reading.triples)
        self.assertEqual(len(reading.disagreements), reading.pairs)

    def test_a_draw_the_cut_cannot_place_is_read_at_the_middle(self):
        for middle, agreement in ((7, 1.0), (8, 0.0)):
            reading = _read(lambda seq: None if seq == _PROBE else 7, middle=middle)
            self.assertEqual(agreement, reading.agreement)
            self.assertEqual([], reading.disagreements)

    def test_an_open_edge_disagrees(self):
        places = lambda seq: 8 if seq == _PROBE[:3] else 7
        learner = _Learner(_StubSifter(places), _OPEN_AT_8, 2)
        learner.draws = iter([_PROBE] * 2000)

        reading = learner.read_fresh(acc_threshold=0.9)

        self.assertEqual(0.0, reading.agreement)

    def test_reading_stops_once_both_its_tests_settle(self):
        learner = _reader(lambda seq: 7)
        reading = learner.read_fresh(acc_threshold=0.5)

        # The tests are read at 30, 60, 120, ... draws.  The agreement settles at
        # the first; no end cut short below the root settles under the stub's
        # 2 * (2 - 1) * 0.1 only by the second, since 0.8 ** 30 > 1e-3 > 0.8 ** 60.
        self.assertEqual(60, learner.drawn)
        self.assertEqual(1.0, reading.agreement)

    def test_ends_cut_short_below_the_root_too_often_are_kept_by_midfix(self):
        # Every draw's whole is cut short at the stub's node below the root.
        reading = _read(lambda seq: None if seq == _PROBE else 7)

        self.assertEqual([("end", b"?")], reading.ends)

    def test_ends_cut_short_at_the_root_are_not(self):
        sifter = _StubSifter(lambda seq: None if seq == _PROBE else 7)
        sifter.sift_and_boundary = lambda seq: (
            (None, bytes(seq)) if seq == _PROBE else (7, None)
        )
        learner = _Learner(sifter, _EVERYWHERE, 2)
        learner.draws = iter([_PROBE] * 2000)

        self.assertEqual([], learner.read_fresh(acc_threshold=0.9).ends)

    def test_a_replay_reads_a_fresh_draw_for_its_triples_middle(self):
        learner = _Learner(_StubSifter(_parting({3})), _EVERYWHERE, 2)
        learner.draws = iter([_PROBE])

        self.assertEqual([_PROBE[:3] + b"?"], learner.replay(_EVERYWHERE))


if __name__ == "__main__":
    unittest.main()
