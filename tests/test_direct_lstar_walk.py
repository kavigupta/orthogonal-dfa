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
from collections import Counter
from types import SimpleNamespace

from orthogonal_dfa.l_star.transition_resolver import (
    MEMBERS,
    OPEN_EDGES,
    PAIR_TRIP,
    PAIRS,
    STOPPED,
    TRIPLES,
    TransitionResolver,
)

_PROBE = bytes([0, 1, 0, 1])


class _StubSifter:
    """A one-node tree placing a string by ``places``, with ``seq + b"?"`` as the
    boundary of what it cannot place, and the middle of the band at ``middle``."""

    def __init__(self, places, middle=None, search=(None, None)):
        self.places = places
        self.middle = middle
        self.search = search
        self.searched = []

    def sift_and_boundary(self, seq):
        leaf = self.places(bytes(seq))
        return (leaf, None) if leaf is not None else (None, bytes(seq) + b"?")

    def halfway(self, seq):
        leaf = self.places(bytes(seq))
        return self.middle if leaf is None else leaf

    def disagreement(self, s, sprime, prefix):
        self.searched.append((s, sprime, prefix))
        return self.search


class _StubPopulation:
    def __init__(self):
        self.recorded = []

    def add(self, string, at):
        self.recorded.append((at, bytes(string)))

    def add_first(self, string, at):
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
        self.pst = SimpleNamespace(
            fnr_limit=0.1,
            alphabet_size=2,
            sampler=SimpleNamespace(length=len(_PROBE)),
            config=SimpleNamespace(min_signal_strength=0.3, split_pval=0.001),
        )
        self.unsplit = Counter()
        self.stopped = []
        self.readings = 0
        self.draws = iter(())
        self.drawn = 0

    def _draw(self):
        self.drawn += 1
        return next(self.draws)

    def _totalised(self):
        """The learned edges, with each open one self-looped."""
        return {
            q: {c: out.get(c, q) for c in (0, 1)} for q, out in self.transitions.items()
        }, []


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
        # It found no distinguisher, which counts against the edge.
        self.assertEqual(Counter({(7, _PROBE[3]): 1}), learner.unsplit)

    def test_an_undecided_witness_is_kept_and_counts_against_the_edge(self):
        # The edge's witness is the empty string, which nothing places.
        learner = _Learner(_StubSifter(_parting({0})), _EVERYWHERE, 1)

        self.assertFalse(learner._check(_PROBE))
        self.assertEqual([b"?"], learner.stopped)
        self.assertEqual(Counter({(7, _PROBE[3]): 1}), learner.unsplit)

    def test_an_undecided_read_in_the_search_is_kept(self):
        sifter = _StubSifter(_parting(set()), search=(None, b"stuck"))
        learner = _Learner(sifter, _EVERYWHERE, 1)

        self.assertFalse(learner._check(_PROBE))
        self.assertEqual([b"stuck"], learner.stopped)
        self.assertEqual(Counter({(7, _PROBE[3]): 1}), learner.unsplit)

    def test_a_replay_keeps_what_stops_the_guards_without_counting_it(self):
        learner = _Learner(_StubSifter(_parting({0})), _EVERYWHERE, 1)
        learner.draws = iter([_PROBE])

        found = learner.replay(SimpleNamespace(learned=_EVERYWHERE), STOPPED)

        self.assertEqual([b"?"], found)
        self.assertEqual(Counter(), learner.unsplit)

    def _asking_for_members(self, unsplit):
        sifter = _StubSifter(_parting(set()), search=(b"m", None))
        learner = _Learner(sifter, _EVERYWHERE, 1)
        learner.family = SimpleNamespace(test_idx=range(10))
        learner.splits = SimpleNamespace(verdict=lambda s, m: "undecided")
        learner.unsplit[7, _PROBE[3]] = unsplit
        return learner

    def test_asking_for_members_resets_the_quiet_run(self):
        learner = self._asking_for_members(0)

        self.assertTrue(learner._check(_PROBE))
        self.assertEqual(1, learner.unsplit[7, _PROBE[3]])

    def test_asking_for_members_on_an_edge_given_up_on_is_quiet(self):
        learner = self._asking_for_members(0)
        given_up = learner._give_up_after()
        learner.unsplit[7, _PROBE[3]] = given_up

        self.assertFalse(learner._check(_PROBE))
        self.assertEqual(given_up + 1, learner.unsplit[7, _PROBE[3]])

    def test_a_disagreement_down_to_a_triple_is_quiet(self):
        sifter = _StubSifter(_parting({3}))
        learner = _Learner(sifter, _EVERYWHERE, 1)

        self.assertFalse(learner._check(_PROBE))
        self.assertEqual([], sifter.searched)


class TestTheProbeBudget(unittest.TestCase):
    @staticmethod
    def _budget(leaves):
        learner = _Learner(_StubSifter(lambda seq: 7), _EVERYWHERE, 1)
        learner.tree = SimpleNamespace(num_states=leaves)
        learner.family = SimpleNamespace(test_idx=range(10))
        return learner.probe_budget(10)

    def test_it_grows_with_the_leaves(self):
        self.assertLess(self._budget(2), self._budget(3))


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
    learner.family = SimpleNamespace(
        middle_side=lambda seq, midfix: label, test_idx=range(31)
    )
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

    def test_a_refusal_samples_edges_are_rerun_until_given_up(self):
        learner = _gate(_disagreeing(set()), _TO_REJECT)
        reading = learner.read_fresh(acc_threshold=0.9)

        self.assertEqual(0.0, reading.agreement)
        self.assertEqual(_PROBE, reading.disagreements[0])
        self.assertEqual({}, reading.harvests)
        self.assertNotIn(PAIRS, reading.fired)

        # The search lands on the edge out of 0 on the probe's last symbol; six
        # split tests on it without a split give it up at the signal stubbed.
        learner.unsplit[0, _PROBE[-1]] = 6
        learner.draws = iter([_PROBE] * 4000)
        self.assertEqual([], learner.read_fresh(acc_threshold=0.9).disagreements)

    def test_a_triple_leaves_its_middles_boundary_string(self):
        reading = _gate(_disagreeing({3}), _TO_REJECT).read_fresh(acc_threshold=0.9)

        self.assertEqual([_PROBE[:3] + b"?"], reading.harvests[TRIPLES])

    def test_a_pair_leaves_nothing_but_is_counted(self):
        learner = _gate(_disagreeing({2, 3}), _TO_REJECT, k=1)
        reading = learner.read_fresh(acc_threshold=0.9)

        self.assertNotIn(TRIPLES, reading.harvests)
        self.assertEqual(
            [_PROBE[:2] + b"?", _PROBE[:3] + b"?"], reading.harvests[PAIRS]
        )
        self.assertLessEqual({PAIRS, PAIR_TRIP}, reading.fired)
        # The gate's 30 draws, then the refusal sample's, which stops at its
        # first look, where the pairs fire.
        self.assertEqual(30 + 30, learner.drawn)

    def test_a_gate_unsettled_at_its_last_look_is_refused(self):
        # Each start agrees on every other draw, exactly the threshold.
        learner = _gate(lambda seq: 0, _STAYS)
        learner.family.middle_side = lambda seq, midfix: seq == _PROBE
        learner.draws = iter([_PROBE, _PROBE[::-1]] * 1240)

        reading = learner.read_fresh(acc_threshold=0.5)

        self.assertFalse(reading.passed)
        # Past the gate's 2000 draws, the refusal sample read.
        self.assertGreater(learner.drawn, 2000)
        self.assertIsNotNone(reading.fired)

    def test_a_refusal_sample_with_nothing_to_hold_or_rerun_reads_to_its_end(self):
        learner = _gate(lambda seq: 0, _TO_REJECT)
        reading = learner.read_fresh(acc_threshold=0.9)

        self.assertEqual(30 + 480, learner.drawn)
        self.assertEqual((set(), []), (reading.fired, reading.disagreements))

    def test_a_passing_gate_stops_once_the_agreement_settles_and_reads_no_ends(
        self,
    ):
        learner = _gate(lambda seq: None if seq == _PROBE else 0, _STAYS)

        reading = learner.read_fresh(acc_threshold=0.5)

        # Read at 30 draws, its first look, and no more drawn for the ends.
        self.assertEqual(30, learner.drawn)
        self.assertEqual([], reading.ends)

    def test_a_refusing_gate_keeps_wholes_cut_short_too_often_by_midfix(self):
        places = lambda seq: None if seq == _PROBE else 0
        reading = _gate(places, _TO_REJECT).read_fresh(acc_threshold=0.9)

        self.assertIn(("end", b"?"), reading.ends)

    def test_a_refusing_gate_keeps_starts_cut_short_too_often_by_midfix(self):
        learner = _gate(lambda seq: None if len(seq) == 2 else 0, _TO_REJECT)

        self.assertEqual([("start", b"?")], learner.read_fresh(acc_threshold=0.9).ends)

    def test_a_read_an_unlearned_edge_left_undecided_is_held(self):
        # Start 1 has no edge on the probe's second symbol, and the cut cannot
        # place the prefix past it; no start accepts what the root does.
        places = lambda seq: {1: 1, 2: None}.get(len(seq), 0)
        learner = _gate(places, {0: {0: 0, 1: 0}, 1: {0: 1}}, k=1)
        learner.tree.accepting_leaves = set

        reading = learner.read_fresh(acc_threshold=0.9)

        self.assertEqual([_PROBE[:2] + b"?"], reading.harvests[OPEN_EDGES])

    def test_any_unlearned_edges_member_is_held(self):
        places = lambda seq: 1 if len(seq) <= 1 else 0
        learner = _gate(places, {0: {0: 0, 1: 0}, 1: {0: 1}}, k=1)
        learner.tree.accepting_leaves = set

        reading = learner.read_fresh(acc_threshold=0.9)

        self.assertEqual([_PROBE[:1]], reading.harvests[MEMBERS])

    def test_a_replay_reads_a_fresh_draw_as_the_gate_did(self):
        learner = _gate(_disagreeing({3}), _TO_REJECT)
        reading = learner.read_fresh(acc_threshold=0.9)
        learner.draws = iter([_PROBE])

        self.assertEqual([_PROBE[:3] + b"?"], learner.replay(reading, TRIPLES))


if __name__ == "__main__":
    unittest.main()
