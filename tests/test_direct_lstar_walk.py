"""Where :meth:`TransitionResolver._process` starts its walk, and what a
disagreeing probe it cannot place leaves in the bisection population.

Driven by stubs rather than synthesis: the anchor is a property of the walk, and
the end-to-end targets that depend on it are noisy enough that a regression shows
up as a state count that also moves for unrelated reasons.
"""

# The tests exercise the walk internals directly.
# pylint: disable=protected-access

import unittest
from types import SimpleNamespace

from orthogonal_dfa.l_star.provenance import Read
from orthogonal_dfa.l_star.transition_resolver import (
    _RESOLVED,
    _UNCHECKED,
    TransitionResolver,
)


class _StubSifter:
    """Places a string only once it is at least ``places_at`` symbols long."""

    def __init__(self, places_at, state=7):
        self.places_at = places_at
        self.state = state
        self.asked = []

    def sift_and_boundary(self, seq):
        self.asked.append(tuple(seq))
        if len(seq) < self.places_at:
            return None, tuple(seq) + ("bail",)
        return self.state, None


class _StubPopulation:
    def __init__(self):
        self.recorded = []

    def add(self, string, at, **_draw):
        self.recorded.append((at, tuple(string)))


class _Learner(TransitionResolver):
    """Captures what the walk hands to the disagreement test."""

    # pylint: disable=super-init-not-called
    def __init__(self, sifter, initial=7):
        self.sifter = sifter
        self.initial = initial
        self.population = _StubPopulation()
        self.tree = SimpleNamespace(path_of=lambda s: s)
        self.dfa = SimpleNamespace(access={})
        self.indecisive = {}
        self.bisected = {}
        self._walked = Read(None, None)
        self.acted = None

    def _initial(self):
        return self.initial

    def _act_on_disagreement(self, w, states, agree_point):
        self.acted = (list(w), list(states), agree_point)
        return _RESOLVED


# Two states, so a walk that follows delta visibly alternates.
DELTA = {7: {0: 8, 1: 7}, 8: {0: 7, 1: 8}}


class TestTheGatesStart(unittest.TestCase):
    def test_every_read_in_the_band_on_it_is_held_once_per_tree(self):
        learner = _Bisecting()
        learner.tree = SimpleNamespace(
            num_states=2, base_family=[], classify=lambda seq, d: 7
        )
        learner.pst = SimpleNamespace(decision_boundary=0.5, oracle=None)
        learner.sifter.halfway = lambda seq: (7, [b"a", b"b"])
        learner._initial_at = (None, None)

        self.assertEqual(7, learner._initial())
        self.assertEqual({b"a", b"b"}, set(learner.bisected))


class TestProcessAnchor(unittest.TestCase):
    def test_walks_from_where_the_middle_of_the_band_places_the_empty_string(self):
        """As the gate does, so a probe disagrees with the walk where the gate
        would count it."""
        learner = _Learner(_StubSifter(places_at=0))
        learner._process([0, 1, 0, 1], DELTA)

        w, states, agree_point = learner.acted
        self.assertEqual(agree_point, 0)
        self.assertEqual(w, [0, 1, 0, 1])
        self.assertEqual(states, [7, 8, 8, 7, 7])

    def test_a_start_the_cut_agrees_with_at_the_anchor_is_kept(self):
        learner = _Learner(_StubSifter(places_at=2), initial=8)
        learner._process([0, 1, 0, 1], DELTA)

        _, states, agree_point = learner.acted
        self.assertEqual(agree_point, 2)
        self.assertEqual(states, [8, 7, 7, 8, 8])
        self.assertEqual(learner.bisected, {})

    def test_a_start_the_cut_parts_from_restarts_the_walk_at_the_anchor(self):
        """So the pass still finds what lies past a misread start; the start's
        reads in the band are held once per tree, not per probe."""
        learner = _Learner(_StubSifter(places_at=2), initial=7)
        learner._process([0, 1, 0, 1], DELTA)

        _, states, agree_point = learner.acted
        self.assertEqual(agree_point, 2)
        self.assertEqual(states, [None, None, 7, 8, 8])
        self.assertEqual(learner.bisected, {})

    def test_records_the_anchor_not_the_empty_string(self):
        learner = _Learner(_StubSifter(places_at=2))
        learner._process([0, 1, 0, 1], DELTA)

        self.assertEqual(learner.population.recorded, [(7, (0, 1))])

    def test_harvests_every_prefix_it_could_not_place(self):
        learner = _Learner(_StubSifter(places_at=2))
        learner._process([0, 1, 0, 1], DELTA)

        self.assertEqual(set(learner.indecisive), {("bail",), (0, "bail")})

    def test_walks_from_the_start_where_no_prefix_places(self):
        learner = _Learner(_StubSifter(places_at=99))
        learner._process([0, 1, 0], DELTA)

        _, states, _ = learner.acted
        self.assertEqual(states, [7, 8, 8, 7])
        self.assertEqual(learner.population.recorded, [])


class _Bisecting(TransitionResolver):
    """Walks stay at 7 while the tree moves four-symbol strings to 8 and cannot
    place the two-symbol prefix the disagreement search reads first."""

    # pylint: disable=super-init-not-called
    def __init__(self):
        def sift(seq):
            if len(seq) == 2:
                return None, bytes(seq) + b"?"
            return (8, None) if len(seq) == 4 else (7, None)

        self.sifter = SimpleNamespace(sift_and_boundary=sift)
        self.indecisive = {}
        self.bisected = {}
        self._walked = Read(None, None)


class TestTheDisagreementSearchHarvestsApart(unittest.TestCase):
    def test_what_the_search_cannot_place_is_kept_out_of_indecisive(self):
        learner = _Bisecting()

        status = learner._act_on_disagreement(bytes([0, 1, 0, 1]), [7] * 5, 0)

        self.assertEqual(_UNCHECKED, status)
        self.assertEqual({bytes([0, 1]) + b"?": learner._walked}, learner.bisected)
        self.assertEqual({}, learner.indecisive)


class _UndecidedAtTheEnd(TransitionResolver):
    """Cannot place the whole probe; the gate's reading of it, which meets two
    strings in the band, lands at ``leaf``."""

    # pylint: disable=super-init-not-called
    def __init__(self, leaf):
        self.sifter = SimpleNamespace(
            sift_and_boundary=lambda seq: (None, bytes(seq) + b"?"),
            halfway=lambda seq: (leaf, [bytes(seq) + b"?", bytes(seq) + b"!"]),
        )
        self.indecisive = {}
        self.bisected = {}
        self._walked = Read(None, None)


class TestAnUndecidedFinalRead(unittest.TestCase):
    def test_one_the_gate_reads_away_from_the_walk_holds_every_read_in_band(self):
        learner = _UndecidedAtTheEnd(leaf=8)

        status = learner._act_on_disagreement(bytes([0, 1]), [7] * 3, 0)

        self.assertEqual(_UNCHECKED, status)
        self.assertEqual(
            {bytes([0, 1]) + b"?", bytes([0, 1]) + b"!"}, set(learner.bisected)
        )
        self.assertEqual({}, learner.indecisive)

    def test_one_the_gate_reads_along_the_walk_stays_a_boundary_string(self):
        learner = _UndecidedAtTheEnd(leaf=7)

        learner._act_on_disagreement(bytes([0, 1]), [7] * 3, 0)

        self.assertEqual({}, learner.bisected)
        self.assertEqual({bytes([0, 1]) + b"?": learner._walked}, learner.indecisive)


class _OnAPlaceholder(TransitionResolver):
    """Walks stay at 7 while the tree moves only the whole four-symbol probe to
    8, over an edge the DFA does not hold."""

    # pylint: disable=super-init-not-called
    def __init__(self):
        self.sifter = SimpleNamespace(
            sift_and_boundary=lambda seq: (8, None) if len(seq) == 4 else (7, None)
        )
        self.dfa = SimpleNamespace(target=lambda s, c: None)
        self.indecisive = {}
        self.bisected = {}
        self._walked = Read(None, None)


class TestADisagreementOnAPlaceholderEdge(unittest.TestCase):
    def test_holds_the_prefix_before_it(self):
        learner = _OnAPlaceholder()

        status = learner._act_on_disagreement(bytes([0, 1, 0, 1]), [7] * 5, 0)

        self.assertEqual(_RESOLVED, status)
        self.assertEqual({bytes([0, 1, 0]): learner._walked}, learner.bisected)
