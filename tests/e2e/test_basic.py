"""Noisy oracles and hand-written targets the learner is expected to learn."""

import unittest

import pytest
from automata.fa.dfa import DFA
from parameterized import parameterized

from orthogonal_dfa.l_star import preconditions as P
from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.examples.bernoulli_parity import (
    AllFramesClosedOracle,
    BernoulliParityOracle,
    BernoulliRegex,
)
from orthogonal_dfa.l_star.learn import learn_dfa as learn_dfa_unchecked
from orthogonal_dfa.l_star.structures import NoisyOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary
from tests.lstar_common import (
    DEFAULT_SAMPLER,
    assert_terminates,
    assertDFA,
    assertDoesNotMeetProperty,
)
from tests.lstar_common import learn_dfa_verified as learn_dfa


def _confounded_frame_product():
    """A 28-state target: the product of a phase/frame automaton and a parity
    confounder over a 5-symbol alphabet.

    frame:  symbols 3,4 advance a phase (mod 3); symbols 0,1,2 close frame f0 at
            phase 0 and f1 at phase 1.  Reject once both are closed.
    parity: (count of 3,4 is odd) AND (count of 3 is odd).
    accept = (not both frames closed) XOR parity.

    State ``(phase6, f0, f1, ones3)`` -- ``phase6`` is the 3,4 count mod 6 (its
    mod-3 is the frame phase, its parity feeds the confounder); ``ones3`` is the
    count of 3s mod 2.  Minimizes to 28 states.
    """

    def step(state, symbol):
        phase6, f0, f1, ones3 = state
        if symbol >= 3:
            return ((phase6 + 1) % 6, f0, f1, ones3 ^ (symbol == 3))
        phase = phase6 % 3
        if phase == 0:
            return (phase6, True, f1, ones3)
        if phase == 1:
            return (phase6, f0, True, ones3)
        return state

    def accepts(state):
        phase6, f0, f1, ones3 = state
        parity = (phase6 % 2 == 1) and ones3
        return (not (f0 and f1)) ^ parity

    start = (0, False, False, False)
    states, transitions, frontier = {start}, {}, [start]
    while frontier:
        state = frontier.pop()
        transitions[state] = {}
        for symbol in range(5):
            nxt = step(state, symbol)
            transitions[state][symbol] = nxt
            if nxt not in states:
                states.add(nxt)
                frontier.append(nxt)
    return DFA(
        states=set(states),
        input_symbols=set(range(5)),
        transitions=transitions,
        initial_state=start,
        final_states={s for s in states if accepts(s)},
        allow_partial=False,
    ).minify()


#: Phase-structured sampling keeps the frame substructure live (under uniform
#: sampling both frames close at once, only the ~4-state parity survives).
_CONFOUNDED_SAMPLER = SuperSampler(
    KmerVocabulary(kmers=((3, 0, 2), (3, 0, 0), (3, 2, 0)), base_alphabet_size=4), 36
)


def _poor_case_target(transitions):
    return DFA(
        states={0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        input_symbols={0, 1},
        transitions=transitions,
        initial_state=0,
        final_states={1},
        allow_partial=False,
    )


POOR_CASE_TARGET = _poor_case_target(
    {
        0: {1: 9, 0: 9},
        1: {1: 1, 0: 1},
        2: {1: 1, 0: 8},
        3: {1: 2, 0: 8},
        4: {1: 5, 0: 3},
        5: {1: 6, 0: 3},
        6: {1: 1, 0: 3},
        7: {1: 4, 0: 8},
        8: {1: 7, 0: 8},
        9: {1: 8, 0: 8},
    }
)

ANOTHER_POOR_CASE_TARGET = _poor_case_target(
    {
        0: {1: 8, 0: 0},
        1: {1: 1, 0: 1},
        2: {1: 1, 0: 6},
        3: {1: 9, 0: 2},
        4: {1: 3, 0: 8},
        5: {1: 8, 0: 4},
        6: {1: 3, 0: 9},
        7: {1: 8, 0: 6},
        8: {1: 8, 0: 5},
        9: {1: 3, 0: 7},
    }
)

#: Targets E-L* cannot learn to the default threshold, kept because they are
#: the counterexamples that found the behaviour, not because they are typical.
POOR_CASE_TARGETS = [
    ("counterexample", POOR_CASE_TARGET),
    ("another_counterexample", ANOTHER_POOR_CASE_TARGET),
]


def _assert_bar_is_what_the_target_allows(testcase, target):
    """Why these targets get a lowered threshold, and why 0.97 in particular.

    No other state transitions into state 0, so a prefix ends there only by
    never leaving it, and none of length 40 does.  E-L* names a state by the
    prefixes that end in it, so it cannot anchor one here and must re-root
    where it can, misclassifying whatever state 0 decides differently.

    What is left caps short of the 0.99 the preconditions ask for, and close
    enough to the default 0.98 that the rounds spent reaching for it land or
    not by noise -- but comfortably above 0.97, which is where assertDFA's own
    tolerance already sat.
    """
    ceiling = P.covered_accuracy_ceiling(target, length=40)
    testcase.assertLess(ceiling, 0.99, "target would be admissible; no need to lower")
    testcase.assertGreater(ceiling, 0.975, "0.97 is not the bar this target allows")


class TestLStar(unittest.TestCase):
    def test_modulo_harder(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliParityOracle(modulo=9, allowed_moduluses=(3, 6)), noise_model, seed
        )
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.2, seed=0)
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)

    @pytest.mark.slow
    def test_modulo_even_harder(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliParityOracle(modulo=9, allowed_moduluses=(3, 6)), noise_model, seed
        )
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.1, seed=0)
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)

    def test_confounded_frame_product(self):
        """A confounder turns a phase/frame automaton into a 28-state product
        whose class-preserving suffixes are rare.  The learner cannot resolve it
        in one round: it grows the pool over a couple of rounds until the
        partition stabilizes, then converges on all 28 states."""
        target = _confounded_frame_product()
        self.assertEqual(len(target.states), 28)
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            DFAOracle(target), noise_model, seed
        )
        dfa = learn_dfa_unchecked(
            oracle_creator,
            min_signal_strength=0.45,
            seed=0,
            sampler=_CONFOUNDED_SAMPLER,
        )
        self.assertEqual(len(dfa.states), 28)
        assertDFA(self, dfa, oracle_creator, symbols=5, sampler=_CONFOUNDED_SAMPLER)

    def test_two_subsequences_with_alternation(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliRegex(regex=r".*1111.*(1111|0000)11.*"), noise_model, seed
        )
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)

    def test_specific_alternation_with_nothing_at_end_3_syms(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliRegex(regex=r".*(111|000).*", alphabet_size=3), noise_model, seed
        )
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        assertDFA(self, dfa, oracle_creator, symbols=3, sampler=DEFAULT_SAMPLER)

    @parameterized.expand(POOR_CASE_TARGETS)
    def test_poor_case_learned_at_a_reachable_bar(self, _name, target):
        _assert_bar_is_what_the_target_allows(self, target)
        oracle_creator = lambda nm, s, _dfa=target: NoisyOracle(DFAOracle(_dfa), nm, s)
        dfa = learn_dfa(
            oracle_creator, min_signal_strength=0.3, seed=0, acc_threshold=0.97
        )
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)

    @parameterized.expand(POOR_CASE_TARGETS)
    def test_poor_case_terminates_at_the_default_bar(self, _name, target):
        # The lowered threshold is what these targets can reach, not what the
        # learner needs to survive: asked for accuracy the target does not
        # have, synthesis must run out of patience and return rather than keep
        # buying prefixes for a gate no family will satisfy. Only termination
        # is asserted -- the DFA it settles on is expected to be imperfect, so
        # there is nothing correct to check it against.
        oracle_creator = lambda nm, s, _dfa=target: NoisyOracle(DFAOracle(_dfa), nm, s)

        assert_terminates(
            lambda: learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0),
            seconds=300,
            message="synthesis did not terminate within the timeout",
        )


@pytest.mark.slow
class TestLStarORF(unittest.TestCase):
    @parameterized.expand([(signal,) for signal in (0.3, 0.2)])
    def test_no_orf(self, signal):
        oracle_creator = lambda nm, s: NoisyOracle(AllFramesClosedOracle(), nm, s)
        dfa = learn_dfa(oracle_creator, min_signal_strength=signal, seed=0)
        assertDFA(self, dfa, oracle_creator, symbols=4, sampler=DEFAULT_SAMPLER)


class TestLStarFast(unittest.TestCase):
    def test_modulo(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliParityOracle(modulo=9, allowed_moduluses=(3, 6)), noise_model, seed
        )
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)

    def test_specific_subsequence(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliRegex(regex=r".*1010101.*"), noise_model, seed
        )
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)

    def test_two_subsequences(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliRegex(regex=r".*1111.*1111.*"), noise_model, seed
        )
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        assertDFA(self, dfa, oracle_creator, sampler=DEFAULT_SAMPLER)

    def test_specific_alternation(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliRegex(regex=r".*(1111|0000)11.*"), noise_model, seed
        )
        dfa = learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0)
        assertDFA(
            self,
            dfa,
            oracle_creator,
            exclude_pattern=lambda s: s[:5] == bytes([1] * 5),
            sampler=DEFAULT_SAMPLER,
        )

    def test_specific_alternation_with_nothing_at_end_does_not_meet_property(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliRegex(regex=r".*(11111|00000).*"), noise_model, seed
        )

        def counterexample_generator(suffix):
            if suffix[0] == 1:
                return bytes([1]) * 4
            return bytes([0]) * 4

        assertDoesNotMeetProperty(
            self, oracle_creator, counterexample_generator, sampler=DEFAULT_SAMPLER
        )

    def test_specific_alternation_with_only_one_at_end_does_not_meet_property(self):
        oracle_creator = lambda noise_model, seed: NoisyOracle(
            BernoulliRegex(regex=r".*(11111|00000)1.*"), noise_model, seed
        )

        def counterexample_generator(suffix):
            if suffix[0] == 1:
                return bytes([1]) * 5
            return bytes([0]) * 4

        assertDoesNotMeetProperty(
            self, oracle_creator, counterexample_generator, sampler=DEFAULT_SAMPLER
        )

    def test_transient_states_terminate(self):
        # Regression for issue #128. This target -- {w : |w| >= 3 and
        # w[2] == '0'} -- has transient states (0, 1, 2) that a fixed-length
        # prefix sampler never lands on, so they are unresolvable and the DFA
        # is not learnable with this sampler. Synthesis must still *terminate*
        # rather than grow the prefix set forever. We assert only that it
        # returns within a generous timeout, not what it returns: the learned
        # DFA is expected to be imperfect, so there is no correct output to
        # check.
        dfa = DFA(
            states={0, 1, 2, 3, 4},
            input_symbols={0, 1},
            transitions={
                0: {0: 1, 1: 1},
                1: {0: 2, 1: 2},
                2: {0: 3, 1: 4},
                3: {0: 3, 1: 3},
                4: {0: 4, 1: 4},
            },
            initial_state=0,
            final_states={3},
            allow_partial=False,
        )
        oracle_creator = lambda nm, s, _dfa=dfa: NoisyOracle(DFAOracle(_dfa), nm, s)

        assert_terminates(
            lambda: learn_dfa(oracle_creator, min_signal_strength=0.45, seed=0),
            seconds=300,
            message="synthesis did not terminate within the timeout (issue #128)",
        )
