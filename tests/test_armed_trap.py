"""A target whose rare state sits inside an absorbing accept state.

Most suffixes carry ``Q`` to an accepting endpoint, so the family votes it accept
even though it is non-final.  At ``cp = 0.0315`` a 149-suffix family draws about
5 that would separate it, which the rest outvote.

``Q`` holds 1.3% of the prefix mass, and losing it merges states.
"""

import unittest

from automata.fa.dfa import DFA

from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.learn import learn_dfa
from orthogonal_dfa.l_star.preconditions import satisfies_preconditions
from orthogonal_dfa.l_star.structures import NoisyOracle
from orthogonal_dfa.l_star.tracker import RecordingTracker
from tests.lstar_common import (
    assertion_allowed_error,
    compute_dfa_accuracy,
    endpoint_mass,
)

ALPHABET = 5
LENGTH = 40
SIGNAL = 0.2


def build_target() -> DFA:
    """Three arming symbols reach ``Q``; from ``Q``, 4 accepts and 3 escapes."""
    transitions = {
        "c0": {0: "c0", 1: "c0", 2: "c0", 3: "c0", 4: "c1"},
        "c1": {0: "c0", 1: "c0", 2: "c0", 3: "c0", 4: "c2"},
        "c2": {0: "c0", 1: "c0", 2: "c0", 3: "c0", 4: "Q"},
        "Q": {0: "Q", 1: "Q", 2: "Q", 3: "e1", 4: "A"},
        "e1": {0: "A", 1: "A", 2: "A", 3: "c0", 4: "A"},
        "A": {c: "A" for c in range(ALPHABET)},
    }
    return DFA(
        states={"c0", "c1", "c2", "Q", "e1", "A"},
        input_symbols=set(range(ALPHABET)),
        transitions=transitions,
        initial_state="c0",
        final_states={"A"},
        allow_partial=False,
    )


def miscut_mass(target: DFA, suffixes) -> float:
    """Prefix mass a family cuts against the language.

    A prefix in state ``q`` draws the vote ``(1/2 - s) + 2 s psi(q)``, where
    ``psi`` is the share of the family carrying ``q`` to an accepting endpoint.
    ``psi`` is a function of the state, so a state is cut whole or not at all.
    The 0.75 sits inside the accept threshold (``1/2 + margin / (2 * signal)``,
    0.7685 at the shipped rates) so that changing those rates does not move this
    measurement.
    """
    mass = endpoint_mass(target, ALPHABET, LENGTH)
    cut = 0.0
    for q in target.states:
        if q in target.final_states:
            continue
        ends = sum(_endpoint(target, q, v) in target.final_states for v in suffixes)
        if ends / len(suffixes) > 0.75:
            cut += mass[q]
    return cut


def _endpoint(target: DFA, q, word):
    for c in word:
        q = target.transitions[q][c]
    return q


class TestArmedTrapTarget(unittest.TestCase):
    def test_preconditions_admit_it(self):
        report = satisfies_preconditions(
            build_target(), length=LENGTH, short_circuit=False
        )
        self.assertTrue(report.satisfied, report.reasons)
        # The miscut needs class-preserving suffixes scarce enough to be outvoted.
        self.assertLess(report.class_preserving_fraction, 0.10)


class TestArmedTrap(unittest.TestCase):
    def test_learned_within_the_bar(self):
        # Seed 0 loses Q; seed 1 recovers all six states, so the seed is pinned.
        target = build_target()
        oracle_creator = lambda nm, s, _d=target: NoisyOracle(DFAOracle(_d), nm, s)
        tracker = RecordingTracker()
        dfa = learn_dfa(
            oracle_creator, min_signal_strength=SIGNAL, seed=0, tracker=tracker
        )

        # The stressor, read off the run rather than off the target: without a
        # round that cuts Q backwards there is nothing here to hold a bar against.
        cut = max(miscut_mass(target, sufs) for sufs, _ in tracker.families)
        self.assertGreater(
            cut,
            0.005,
            f"no round cut more than {cut:.4f} of the prefix mass against the "
            f"language, where this target has measured 0.0132.  The merge this "
            f"test exists to hold a bar against is no longer being provoked, so "
            f"the assertion below proves nothing -- you might need to find "
            f"another reproducing target.",
        )

        accuracy, fp, fn = compute_dfa_accuracy(dfa, oracle_creator, symbols=ALPHABET)
        if accuracy < 1 - assertion_allowed_error:
            self.fail(
                f"DFA incorrect (accuracy {accuracy:.4f}). "
                f"FP: {len(fp)}, FN: {len(fn)}"
            )
