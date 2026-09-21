"""A legal target whose rounds cut ~11% of a population backwards, and which the
loop repairs anyway.

The accept-preserving band bounds a round's wrongly-cut share at
``(1 - eps/signal)/2`` -- 0.232 at the shipped rates -- and every target in the
suite until now measured 0.0 against it, which made the bound look unreachable.
Rounds here miscut 0.107-0.116, so the bound has a factor of two in hand rather
than a factor of infinity.

The returned DFA is still accurate to 0.9999, because the tree goes on splitting
the miscut population after the family has cut it. That is the invariant worth
guarding: a large per-round coverage error does not reach the answer.
"""

import unittest

import numpy as np
import pytest
from automata.fa.dfa import DFA

from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.learn import learn_dfa
from orthogonal_dfa.l_star.preconditions import satisfies_preconditions
from orthogonal_dfa.l_star.structures import NoisyOracle
from tests.lstar_common import assertion_allowed_error, compute_dfa_accuracy

ALPHABET = 200
LENGTH = 40
#: The rate the round-level miscut was measured at; 0.3 resolves the target in a
#: single clean cut and exercises nothing.
SIGNAL = 0.2
#: Symbols moving ``c0`` into the armed state.
ARM = range(196, ALPHABET)
#: The one symbol that disarms it, so the one a suffix must avoid to preserve
#: every state's class.
DISARM = 180
#: Symbols the armed state holds on.  The dwell they buy is what makes the armed
#: class heavy: mass(Q) ~ acceptance_rate * dwell / LENGTH.
HOLD = range(0, DISARM)


def build_target() -> DFA:
    """Four states over 200 symbols, of which the armed ``Q`` is non-final."""
    transitions = {
        "c0": {c: ("Q" if c in ARM else "c0") for c in range(ALPHABET)},
        "Q": {
            c: ("Q" if c in HOLD else "e1" if c == DISARM else "A")
            for c in range(ALPHABET)
        },
        "e1": {c: "c0" for c in range(ALPHABET)},
        "A": {c: "A" for c in range(ALPHABET)},
    }
    return DFA(
        states={"c0", "Q", "e1", "A"},
        input_symbols=set(range(ALPHABET)),
        transitions=transitions,
        initial_state="c0",
        final_states={"A"},
        allow_partial=False,
    )


def endpoint_mass(target: DFA) -> dict:
    """Exact share of uniform length-``LENGTH`` strings ending in each state."""
    states = sorted(target.states)
    index = {s: i for i, s in enumerate(states)}
    step = np.zeros((len(states), len(states)))
    for q in states:
        for c in range(ALPHABET):
            step[index[q], index[target.transitions[q][c]]] += 1 / ALPHABET
    mass = np.zeros(len(states))
    mass[index[target.initial_state]] = 1.0
    for _ in range(LENGTH):
        mass = mass @ step
    return dict(zip(states, mass))


class TestHeavyCoverageErrorTarget(unittest.TestCase):
    """The construction, not the learner: cheap, and guards the stressor."""

    def test_target_is_admitted_and_still_hard(self):
        target = build_target()
        report = satisfies_preconditions(target, length=LENGTH, short_circuit=False)
        self.assertTrue(report.satisfied, report.reasons)
        # A miscut needs the family's clean suffixes to stay scarce and the
        # miscut class to stay heavy.  Either drifting would leave the test below
        # passing without exercising anything.
        self.assertLess(report.class_preserving_fraction, 0.10)
        self.assertGreater(endpoint_mass(target)["Q"], 0.08)


@pytest.mark.slow
class TestHeavyCoverageErrorLearns(unittest.TestCase):
    def test_learns_despite_heavy_miscut(self):
        # One seed: nine measured seeds all miscut 0.107-0.116 at round level and
        # all returned 0.9999, so the draw is not what this is guarding against.
        target = build_target()
        oracle_creator = lambda nm, s, _d=target: NoisyOracle(DFAOracle(_d), nm, s)
        dfa = learn_dfa(oracle_creator, min_signal_strength=SIGNAL, seed=0)
        accuracy, fp, fn = compute_dfa_accuracy(dfa, oracle_creator, symbols=ALPHABET)
        if accuracy < 1 - assertion_allowed_error:
            self.fail(
                f"DFA incorrect (accuracy {accuracy:.4f}). "
                f"FP: {len(fp)}, FN: {len(fn)}"
            )
