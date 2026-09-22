"""Targets where a class was merged into an absorbing accept state.

``Q`` is non-final but leaks into accepting ``A`` on most symbols and escapes on
one, so a family built from the suffixes that do not escape votes ``Q`` accept.
The tree then inherits that verdict and the hypothesis is built from the tree,
so the two AGREE on ``Q`` -- DFA/DT consistency clears its target and synthesis
used to stop there, returning two states of four.

Reading the hypothesis's own states against the oracle sees what that comparison
cannot, since it does not ask the tree anything.
"""

import unittest

import numpy as np
from automata.fa.dfa import DFA
from parameterized import parameterized

from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.learn import learn_dfa
from orthogonal_dfa.l_star.preconditions import satisfies_preconditions
from orthogonal_dfa.l_star.structures import NoisyOracle
from tests.lstar_common import assertion_allowed_error, compute_dfa_accuracy

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


def endpoint_mass(target: DFA, alphabet: int) -> dict:
    """Exact share of uniform length-``LENGTH`` strings ending in each state."""
    states = sorted(target.states)
    index = {s: i for i, s in enumerate(states)}
    step = np.zeros((len(states), len(states)))
    for q in states:
        for c in range(alphabet):
            step[index[q], index[target.transitions[q][c]]] += 1 / alphabet
    mass = np.zeros(len(states))
    mass[index[target.initial_state]] = 1.0
    for _ in range(LENGTH):
        mass = mass @ step
    return dict(zip(states, mass))


#: ``(alphabet, arms, disarm)``.  The first merged ``Q`` on 3 of 8 seeds before
#: the homogeneity check; the second on 1 of 8.
TRAPS = [(200, 4, 170), (200, 4, 180)]


class TestTrapTargets(unittest.TestCase):
    @parameterized.expand([(k, a, d) for k, a, d in TRAPS])
    def test_admitted_and_still_heavy(self, alphabet, arms, disarm):
        target = build_trap(alphabet, arms, disarm)
        report = satisfies_preconditions(target, length=LENGTH, short_circuit=False)
        self.assertTrue(report.satisfied, report.reasons)
        # A merge only costs accuracy while the merged class carries mass, and
        # is only reachable while class-preserving suffixes are scarce.
        self.assertGreater(endpoint_mass(target, alphabet)["Q"], 0.05)
        self.assertLess(report.class_preserving_fraction, 0.10)


class TestTrapLearned(unittest.TestCase):
    @parameterized.expand([(k, a, d, seed) for k, a, d in TRAPS for seed in range(4)])
    def test_learned_within_the_bar(self, alphabet, arms, disarm, seed):
        target = build_trap(alphabet, arms, disarm)
        oracle_creator = lambda nm, s, _d=target: NoisyOracle(DFAOracle(_d), nm, s)
        dfa = learn_dfa(oracle_creator, min_signal_strength=SIGNAL, seed=seed)
        accuracy, fp, fn = compute_dfa_accuracy(dfa, oracle_creator, symbols=alphabet)
        if accuracy < 1 - assertion_allowed_error:
            self.fail(
                f"DFA incorrect (accuracy {accuracy:.4f}, "
                f"{len(dfa.states)} of {len(target.states)} states). "
                f"FP: {len(fp)}, FN: {len(fn)}"
            )
