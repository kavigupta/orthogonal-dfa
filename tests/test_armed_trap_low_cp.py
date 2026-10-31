"""KNOWN FAILURE.  An admitted target the learner gets wrong on some seeds.

``Q`` is non-final but leaks into the absorbing accept state ``A`` on 29 of 200
symbols and escapes only via symbol 170, so most suffixes carry it to an
accepting endpoint and a family has to be built from the 2.3% that do not.

At ``cp = 0.0230`` -- admitted, the bar is 0.02 -- the learner returns two states
of four on roughly a third of seeds, collapsing ``Q`` into ``A``: everything past
an arming symbol is accepted, where the target needs a further symbol from ``Q``
and can escape back to ``c0``.  ``Q`` holds 7% of the prefix mass and the
returned DFA is 7.75% wrong.

The remaining seeds return 0.9998, so this is not a target the learner cannot
address -- it is one where whether the rounds recover ``Q`` comes out of the
draw.
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

ALPHABET = 200
LENGTH = 40
SIGNAL = 0.2
#: Symbols moving ``c0`` into the armed state.
ARM = range(ALPHABET - 4, ALPHABET)
#: The one symbol that disarms it, so the one a suffix must use to tell ``Q``
#: from ``A``.  Symbols below it hold in ``Q``; those above leak to accept.
DISARM = 170


def build_target() -> DFA:
    transitions = {
        "c0": {c: ("Q" if c in ARM else "c0") for c in range(ALPHABET)},
        "Q": {
            c: ("Q" if c < DISARM else "e1" if c == DISARM else "A")
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


class TestArmedTrapLowCpTarget(unittest.TestCase):
    def test_preconditions_admit_it(self):
        report = satisfies_preconditions(
            build_target(), length=LENGTH, short_circuit=False
        )
        self.assertTrue(report.satisfied, report.reasons)
        self.assertGreater(endpoint_mass(build_target())["Q"], 0.05)


class TestArmedTrapLowCp(unittest.TestCase):
    """Seeds 0 and 3 fail, 1 and 2 pass; the range is the first four, not a pick."""

    @parameterized.expand([(seed,) for seed in range(4)])
    def test_learned_within_the_bar(self, seed):
        target = build_target()
        oracle_creator = lambda nm, s, _d=target: NoisyOracle(DFAOracle(_d), nm, s)
        dfa = learn_dfa(oracle_creator, min_signal_strength=SIGNAL, seed=seed)
        accuracy, fp, fn = compute_dfa_accuracy(dfa, oracle_creator, symbols=ALPHABET)
        if accuracy < 1 - assertion_allowed_error:
            self.fail(
                f"DFA incorrect (accuracy {accuracy:.4f}, "
                f"{len(dfa.states)} of {len(target.states)} states). "
                f"FP: {len(fp)}, FN: {len(fn)}"
            )
