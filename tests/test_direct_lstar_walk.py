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

    def test_a_start_the_cut_parts_from_is_held_and_the_walk_restarts(self):
        """The gate counts every such probe on the empty string's read alone, so
        that read is what the next family must settle; the walk then starts
        from the anchor, so the pass still finds what lies past it."""
        learner = _Learner(_StubSifter(places_at=2), initial=7)
        learner._process([0, 1, 0, 1], DELTA)

        _, states, agree_point = learner.acted
        self.assertEqual(agree_point, 2)
        self.assertEqual(states, [None, None, 7, 8, 8])
        self.assertEqual(set(learner.bisected), {("bail",)})

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
    """Cannot place the whole probe; the middle of the band sends it away from
    the walk or not, as the test chooses."""

    # pylint: disable=super-init-not-called
    def __init__(self, departs):
        self.sifter = SimpleNamespace(
            sift_and_boundary=lambda seq: (None, bytes(seq) + b"?"),
            middle_departs=lambda seq, leaf: departs,
        )
        self.indecisive = {}
        self.bisected = {}
        self._walked = Read(None, None)


class TestAnUndecidedFinalRead(unittest.TestCase):
    def test_one_the_middle_sends_away_from_the_walk_joins_the_bisection(self):
        learner = _UndecidedAtTheEnd(departs=True)

        status = learner._act_on_disagreement(bytes([0, 1]), [7] * 3, 0)

        self.assertEqual(_UNCHECKED, status)
        self.assertEqual({bytes([0, 1]) + b"?": learner._walked}, learner.bisected)
        self.assertEqual({}, learner.indecisive)

    def test_one_the_middle_sends_along_the_walk_stays_a_boundary_string(self):
        learner = _UndecidedAtTheEnd(departs=False)

        learner._act_on_disagreement(bytes([0, 1]), [7] * 3, 0)

        self.assertEqual({}, learner.bisected)
        self.assertEqual({bytes([0, 1]) + b"?": learner._walked}, learner.indecisive)
