"""A target whose rejecting armed state merges into the absorbing accept one.

``Q`` is non-final but leaks into accepting ``A`` on most symbols and escapes on
one, so a family built from the suffixes that do not escape votes ``Q`` accept.
The tree then inherits that verdict and the hypothesis is built from the tree,
so the two AGREE on ``Q`` -- DFA/DT consistency clears its target and synthesis
stops there, returning two states of four.
"""

import unittest

from automata.fa.dfa import DFA
from parameterized import parameterized

from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.learn import learn_dfa
from orthogonal_dfa.l_star.preconditions import satisfies_preconditions
from orthogonal_dfa.l_star.structures import NoisyOracle
from tests.lstar_common import (
    assertion_allowed_error,
    compute_dfa_accuracy,
    endpoint_mass,
)

LENGTH = 40
SIGNAL = 0.2


def build_trap(alphabet: int, arms: int, disarm: int) -> DFA:
    """``arms`` symbols reach ``Q``; from ``Q``, ``disarm`` escapes and the
    symbols above it leak into the absorbing accept state."""
    arm = range(alphabet - arms, alphabet)
    transitions = {
        "c0": {c: ("Q" if c in arm else "c0") for c in range(alphabet)},
        "Q": {
            c: ("Q" if c < disarm else "e1" if c == disarm else "A")
            for c in range(alphabet)
        },
        "e1": {c: "c0" for c in range(alphabet)},
        "A": {c: "A" for c in range(alphabet)},
    }
    return DFA(
        states={"c0", "Q", "e1", "A"},
        input_symbols=set(range(alphabet)),
        transitions=transitions,
        initial_state="c0",
        final_states={"A"},
        allow_partial=False,
    )


#: ``(alphabet, arms, disarm)``.
TRAPS = [(200, 4, 170), (200, 4, 180)]

#: Seeds the merge is measured over, one test apiece.  Which of them merge is a
#: property of the suffixes a round happens to draw, so a bar on one seed says
#: nothing; at the measured rate of 2 in 10, a clean run of this many is a
#: 1-in-9 event if the rate were unchanged.  A seed each rather than a loop:
#: one of these learns is most of a test's clock.
SEEDS = 10


class TestTrapTargets(unittest.TestCase):
    @parameterized.expand(list(TRAPS))
    def test_admitted_and_still_heavy(self, alphabet, arms, disarm):
        target = build_trap(alphabet, arms, disarm)
        report = satisfies_preconditions(target, length=LENGTH, short_circuit=False)
        self.assertTrue(report.satisfied, report.reasons)
        # A merge only costs accuracy while the merged class carries mass, and
        # is only reachable while class-preserving suffixes are scarce.
        self.assertGreater(endpoint_mass(target, alphabet, LENGTH)["Q"], 0.05)
        self.assertLess(report.class_preserving_fraction, 0.10)


class TestTrapLearned(unittest.TestCase):
    """``Q`` is rejecting and carries 7% of the endpoint mass, so a family that
    votes it into accepting ``A`` costs that much accuracy at the distribution
    the DFA is graded on -- unlike ``e1``, which merges into a rejecting state
    and costs only its own routing.

    The covered-accuracy ceiling is 1.0 here: the initial state is covered, so
    nothing about the sampler stops the target being learned exactly.
    """

    @parameterized.expand([(seed,) for seed in range(SEEDS)])
    def test_the_armed_state_is_not_merged(self, seed):
        alphabet, arms, disarm = TRAPS[0]
        target = build_trap(alphabet, arms, disarm)
        oracle_creator = lambda nm, s, _d=target: NoisyOracle(DFAOracle(_d), nm, s)
        dfa = learn_dfa(oracle_creator, min_signal_strength=SIGNAL, seed=seed)
        accuracy, fp, fn = compute_dfa_accuracy(dfa, oracle_creator, symbols=alphabet)
        if accuracy < 1 - assertion_allowed_error:
            self.fail(
                f"merged the armed state (accuracy {accuracy:.4f}, "
                f"{len(dfa.states)} of {len(target.states)} states). "
                f"FP: {len(fp)}, FN: {len(fn)}"
            )
