"""How :class:`TransitionResolver` walks a probe from its start, what it does
where the walk is blocked or a split attempt cannot go on, how it reads fresh
draws, and the start it exports.

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
        self.reads = 0
        self.undecided_search = None

    def sift_and_boundary(self, seq):
        self.reads += 1
        leaf = self.places(bytes(seq))
        return (leaf, None) if leaf is not None else (None, bytes(seq) + b"?")

    def halfway(self, seq):
        leaf = self.places(bytes(seq))
        return self.middle if leaf is None else leaf

    def disagreement(self, _s, _sprime, _prefix):
        return None, self.undecided_search


class _StubPopulation:
    def __init__(self):
        self.recorded = []

    def add(self, string, at):
        self.recorded.append((at, bytes(string)))

    add_first = add


class _Learner(TransitionResolver):
    # pylint: disable=super-init-not-called
    def __init__(self, sifter, transitions, k, *, fnr_limit=0.1):
        self.sifter = sifter
        self.population = _StubPopulation()
        self.tree = SimpleNamespace(path_of=lambda s: s, num_states=2)
        self.dfa = SimpleNamespace(transitions=transitions, witness=lambda s, c: b"")
        self.k = k
        self.dropped = {}
        self.recent = deque()
        self.pst = SimpleNamespace(fnr_limit=fnr_limit)
        self.draws = iter(())

    def _draw(self):
        return next(self.draws)


#: State 7 steps to 8 on a 0 and stays on a 1; nothing is learned out of 8.
_OPEN_AT_8 = {7: {0: 8, 1: 7}, 8: {}}
_EVERYWHERE = {7: {0: 7, 1: 7}}


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

    def test_an_open_edge_from_a_prefix_sifting_elsewhere_is_searched(self):
        # The walk reaches 8 after 0, 1, 0, which the cut places at 7, and finds
        # no edge out of 8; the search lands on the edge into 8.
        sifter = _StubSifter(lambda seq: 7)
        sifter.undecided_search = b"searched"
        learner = _Learner(sifter, _OPEN_AT_8, 1)

        self.assertFalse(learner._check(_PROBE))
        self.assertEqual({b"searched": None}, learner.dropped)

    def test_a_disagreement_whose_prefix_the_cut_cannot_place_is_dropped(self):
        # The walk stays at 7 while the whole probe sifts to 8, and the prefix the
        # search lands on can only be placed by the middle.
        def places(seq):
            if len(seq) == 4:
                return 8
            return None if len(seq) == 3 else 7

        learner = _Learner(_StubSifter(places, middle=7), _EVERYWHERE, 1)

        self.assertFalse(learner._check(_PROBE))
        self.assertEqual({_PROBE[:3] + b"?": None}, learner.dropped)

    def test_a_distinguisher_search_the_cut_cannot_finish_is_dropped(self):
        sifter = _StubSifter(lambda seq: 8 if len(seq) == 4 else 7)
        sifter.undecided_search = b"stuck"
        learner = _Learner(sifter, _EVERYWHERE, 1)

        self.assertFalse(learner._check(_PROBE))
        self.assertEqual({b"stuck": None}, learner.dropped)


def _read(places, draws, *, whole, middle=7, acc_threshold=0.9):
    learner = _Learner(_StubSifter(places, middle=middle), _EVERYWHERE, 2)
    learner.draws = iter(draws * 2000)
    return learner.read_fresh(whole=whole, acc_threshold=acc_threshold)


#: Places every string but the whole probe at 7.
_ALL_BUT_PROBE = lambda seq: None if seq == _PROBE else 7


class TestReadingFreshDraws(unittest.TestCase):
    def test_draws_blocking_at_more_than_the_limits_share_of_reads_block(self):
        # Every other draw blocks over two reads each: a quarter of the reads.
        reading = _read(_ALL_BUT_PROBE, [_PROBE, _PROBE[:3]], whole=True)

        self.assertTrue(reading.blocks)
        self.assertEqual([_PROBE + b"?"], reading.found)

    def test_a_blocked_draw_is_read_at_the_middle_for_the_agreement(self):
        for middle, agreement in ((7, 1.0), (8, 0.0)):
            reading = _read(_ALL_BUT_PROBE, [_PROBE], whole=True, middle=middle)
            self.assertEqual(agreement, reading.agreement)
            self.assertEqual([], reading.disagreements)

    def test_a_refusal_with_nothing_decided_to_rerun_blocks(self):
        # Half the draws block, over a quarter of the reads: under a 0.3 limit.
        draws = [_PROBE, _PROBE[:3]]
        for middle, blocks in ((7, False), (8, True)):
            learner = _Learner(
                _StubSifter(_ALL_BUT_PROBE, middle=middle),
                _EVERYWHERE,
                2,
                fnr_limit=0.3,
            )
            learner.draws = iter(draws * 1000)
            reading = learner.read_fresh(whole=True, acc_threshold=0.9)
            self.assertEqual(blocks, reading.blocks)

    def test_the_walk_check_does_not_sift_the_whole_draw(self):
        reading = _read(_ALL_BUT_PROBE, [_PROBE], whole=False)

        self.assertFalse(reading.blocks)
        self.assertEqual([], reading.found)

    def test_an_open_edge_disagrees(self):
        places = lambda seq: 8 if seq == _PROBE[:3] else 7
        learner = _Learner(_StubSifter(places), _OPEN_AT_8, 2)
        learner.draws = iter([_PROBE] * 2000)

        reading = learner.read_fresh(whole=True, acc_threshold=0.9)

        self.assertEqual(0.0, reading.agreement)
        self.assertEqual([], reading.disagreements)

    def test_an_open_edge_from_a_prefix_sifting_elsewhere_is_kept_for_the_pass(self):
        learner = _Learner(_StubSifter(lambda seq: 7), _OPEN_AT_8, 1)
        learner.draws = iter([_PROBE] * 2000)

        reading = learner.read_fresh(whole=True, acc_threshold=0.9)

        self.assertEqual(_PROBE, reading.disagreements[0])
        self.assertFalse(reading.blocks)

    def test_a_decided_disagreement_is_kept_for_the_pass(self):
        reading = _read(lambda seq: 8 if len(seq) == 4 else 7, [_PROBE], whole=True)

        self.assertEqual(0.0, reading.agreement)
        self.assertEqual(_PROBE, reading.disagreements[0])

    def test_a_clean_gate_reads_until_both_its_tests_settle(self):
        reading = _read(lambda seq: 7, [_PROBE], whole=True, acc_threshold=0.5)

        # Two clean reads a draw settle the block test below 0.1 at the 66th
        # read, since 0.9 ** 66 < 1e-3 < 0.9 ** 65, after the agreement's 30
        # draws.
        self.assertEqual(33, len(reading.draws))
        self.assertFalse(reading.blocks)
        self.assertEqual(1.0, reading.agreement)

    def test_the_walk_check_stops_once_its_block_test_settles(self):
        reading = _read(lambda seq: None, [_PROBE], whole=False)

        # One blocked read a draw settles above 0.1 at the fourth.
        self.assertEqual(4, len(reading.draws))
        self.assertTrue(reading.blocks)
        self.assertEqual([_PROBE[:2]], reading.found)

    def test_a_replay_reads_a_fresh_draw_for_what_it_leaves(self):
        learner = _Learner(_StubSifter(lambda seq: None), _EVERYWHERE, 2)
        learner.draws = iter([_PROBE] * 4)
        reading = learner.read_fresh(whole=False, acc_threshold=0.9)

        learner.draws = iter([_PROBE[::-1]])
        self.assertEqual([_PROBE[::-1][:2]], reading.replay())


class TestTheExportedStart(unittest.TestCase):
    def test_it_is_the_state_whose_runs_accept_where_the_root_does(self):
        # From 0 a 1 accepts; from 1 everything accepts.  The root reads probes
        # ending in 1 as accepting and the rest as not, which only 0 matches.
        learner = _Learner(_StubSifter(lambda seq: 0), {}, 1)
        learner.family = SimpleNamespace(middle_side=lambda seq, midfix: seq[-1] == 1)
        learner.recent = deque([bytes([0, 1]), bytes([1, 0]), bytes([0, 0])])
        transitions = {0: {0: 0, 1: 1}, 1: {0: 0, 1: 1}}

        self.assertEqual(0, learner._best_start(transitions, accepting={1}))

    def test_a_tie_goes_to_the_lowest_state(self):
        # Each state's runs accept one of the two probes the root accepts.
        learner = _Learner(_StubSifter(lambda seq: 0), {}, 1)
        learner.family = SimpleNamespace(middle_side=lambda seq, midfix: True)
        learner.recent = deque([bytes([0]), bytes([0, 0])])

        self.assertEqual(0, learner._best_start({0: {0: 1}, 1: {0: 0}}, {1}))


if __name__ == "__main__":
    unittest.main()
