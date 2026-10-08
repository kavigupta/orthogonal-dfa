"""Targets with a rejecting state that most suffixes carry into accepting."""

import unittest

import pytest
from automata.fa.dfa import DFA
from parameterized import parameterized

from orthogonal_dfa.l_star import preconditions as P
from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.learn import learn_dfa as learn_dfa_unchecked
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.structures import NoisyOracle
from orthogonal_dfa.l_star.tracker import RecordingTracker
from tests.lstar_common import (
    DEFAULT_SAMPLER,
    assert_not_merged,
    assertion_allowed_error,
    compute_dfa_accuracy,
    endpoint_mass,
)

# The armed target: a rejecting state Q that most suffixes carry to the accepting
# sink A, so a family can vote Q into A while DFA/DT consistency stays at target.

ARMED_ALPHABET = 12
#: Symbols that arm the trap.  One: two of them let the round separate Q and the target
#: is learned exactly.
ARMED_ARMS = 1
#: Symbols Q holds on.  The rest reach A, so this sets how many suffixes preserve
#: Q's class -- holds/alphabet to the sampler's length, which has to clear the 2% the
#: preconditions ask for without clearing it by much.
ARMED_HOLDS = 10
ARMED_LENGTH = 20
ARMED_SIGNAL = 0.3

#: Seeds each merge rate is measured over, one test apiece: which of them merge is a
#: property of the suffixes a round draws, so a bar on one seed says nothing.
MERGE_SEEDS = 10


def build_armed_target() -> DFA:
    arm = set(range(ARMED_ALPHABET - ARMED_ARMS, ARMED_ALPHABET))
    return DFA(
        states={"c0", "Q", "A"},
        input_symbols=set(range(ARMED_ALPHABET)),
        transitions={
            "c0": {c: ("Q" if c in arm else "c0") for c in range(ARMED_ALPHABET)},
            "Q": {c: ("Q" if c < ARMED_HOLDS else "A") for c in range(ARMED_ALPHABET)},
            "A": {c: "A" for c in range(ARMED_ALPHABET)},
        },
        initial_state="c0",
        final_states={"A"},
        allow_partial=False,
    )


class TestArmedMergeTarget(unittest.TestCase):
    """The construction, not the learner: cheap, and guards the stressor."""

    def test_admitted_and_learnable(self):
        target = build_armed_target()
        sampler = UniformSampler(ARMED_LENGTH)
        report = P.satisfies_preconditions(
            target, length=ARMED_LENGTH, short_circuit=False, sampler=sampler
        )
        self.assertTrue(report.satisfied, report.reasons)
        # The merge is only reachable while the suffixes that preserve Q are scarce,
        # and only costly while Q carries mass.  Either drifting leaves the test below
        # passing without exercising anything.
        self.assertLess(report.class_preserving_fraction, 0.05)
        # Nothing about the sampler stops this being learned exactly, so the shortfall
        # below is the learner's and not the target's.
        self.assertAlmostEqual(
            P.covered_accuracy_ceiling(target, length=ARMED_LENGTH, sampler=sampler),
            1.0,
        )


class TestArmedMergeLearned(unittest.TestCase):
    @parameterized.expand([(seed,) for seed in range(MERGE_SEEDS)])
    def test_the_armed_state_is_not_merged(self, seed):
        target = build_armed_target()
        oracle_creator = lambda nm, s, _d=target: NoisyOracle(DFAOracle(_d), nm, s)
        sampler = UniformSampler(ARMED_LENGTH)
        dfa = learn_dfa_unchecked(
            oracle_creator,
            min_signal_strength=ARMED_SIGNAL,
            seed=seed,
            sampler=sampler,
        )
        # Graded at the length it was learned at: a hypothesis that merges Q still
        # reads well on strings long enough that almost all of them reach A anyway.
        assert_not_merged(
            self,
            dfa,
            target,
            oracle_creator=oracle_creator,
            symbols=ARMED_ALPHABET,
            sampler=sampler,
        )


TRAP_LENGTH = 40
TRAP_SIGNAL = 0.2


def build_trap(alphabet: int, arms: int, disarm: int) -> DFA:
    """arms symbols reach Q; from Q, disarm escapes and the symbols above
    it leak into the absorbing accept state."""
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


#: (alphabet, arms, disarm).
TRAPS = [(200, 4, 170), (200, 4, 180)]


class TestTrapTargets(unittest.TestCase):
    @parameterized.expand(list(TRAPS))
    def test_admitted_and_still_heavy(self, alphabet, arms, disarm):
        target = build_trap(alphabet, arms, disarm)
        report = P.satisfies_preconditions(
            target, length=TRAP_LENGTH, short_circuit=False
        )
        self.assertTrue(report.satisfied, report.reasons)
        # A merge only costs accuracy while the merged class carries mass, and is only
        # reachable while class-preserving suffixes are scarce.
        self.assertGreater(endpoint_mass(target, TRAP_LENGTH)["Q"], 0.05)
        self.assertLess(report.class_preserving_fraction, 0.10)


#: A seed whose first DFA the certificate refuses runs about ten minutes.  main merges
#: on 3 of 8 seeds (#307), so three passing leave a quarter chance that a learner
#: merging as often slips by.
TRAP_SEEDS = 3


@pytest.mark.slow
class TestTrapLearned(unittest.TestCase):
    """Q is rejecting and carries 7% of the endpoint mass, so a family that votes it
    into accepting A costs that much accuracy at the distribution the DFA is graded
    on -- unlike e1, which merges into a rejecting state and costs only its own
    routing."""

    @parameterized.expand([(seed,) for seed in range(TRAP_SEEDS)])
    def test_the_armed_state_is_not_merged(self, seed):
        alphabet, arms, disarm = TRAPS[0]
        target = build_trap(alphabet, arms, disarm)
        oracle_creator = lambda nm, s, _d=target: NoisyOracle(DFAOracle(_d), nm, s)
        sampler = UniformSampler(TRAP_LENGTH)
        dfa = learn_dfa_unchecked(
            oracle_creator, min_signal_strength=TRAP_SIGNAL, seed=seed, sampler=sampler
        )
        assert_not_merged(
            self,
            dfa,
            target,
            oracle_creator=oracle_creator,
            symbols=alphabet,
            sampler=sampler,
        )


# The five-symbol trap of #305: Q holds 1.3% of the prefix mass inside an absorbing
# accept state, and most suffixes carry it there.

FIVE_ALPHABET = 5
FIVE_LENGTH = 40
FIVE_SIGNAL = 0.2


def build_five_symbol_trap() -> DFA:
    """Three arming symbols reach Q; from Q, 4 accepts and 3 escapes."""
    transitions = {
        "c0": {0: "c0", 1: "c0", 2: "c0", 3: "c0", 4: "c1"},
        "c1": {0: "c0", 1: "c0", 2: "c0", 3: "c0", 4: "c2"},
        "c2": {0: "c0", 1: "c0", 2: "c0", 3: "c0", 4: "Q"},
        "Q": {0: "Q", 1: "Q", 2: "Q", 3: "e1", 4: "A"},
        "e1": {0: "A", 1: "A", 2: "A", 3: "c0", 4: "A"},
        "A": {c: "A" for c in range(FIVE_ALPHABET)},
    }
    return DFA(
        states={"c0", "c1", "c2", "Q", "e1", "A"},
        input_symbols=set(range(FIVE_ALPHABET)),
        transitions=transitions,
        initial_state="c0",
        final_states={"A"},
        allow_partial=False,
    )


def _miscut_mass(target: DFA, suffixes) -> float:
    """Prefix mass a family cuts against the language.

    A prefix in state q draws the vote (1/2 - s) + 2 s psi(q), where
    psi is the share of the family carrying q to an accepting endpoint.
    psi is a function of the state, so a state is cut whole or not at all.
    The 0.75 sits inside the accept threshold (1/2 + margin / (2 * signal),
    0.7685 at the shipped rates) so that changing those rates does not move this
    measurement.
    """
    mass = endpoint_mass(target, FIVE_LENGTH)
    cut = 0.0
    for q in target.states:
        if q in target.final_states:
            continue
        ends = sum(_end_state(target, q, v) in target.final_states for v in suffixes)
        if ends / len(suffixes) > 0.75:
            cut += mass[q]
    return cut


def _end_state(target: DFA, q, word):
    for c in word:
        q = target.transitions[q][c]
    return q


class TestFiveSymbolTrapTarget(unittest.TestCase):
    def test_preconditions_admit_it(self):
        report = P.satisfies_preconditions(
            build_five_symbol_trap(), length=FIVE_LENGTH, short_circuit=False
        )
        self.assertTrue(report.satisfied, report.reasons)
        # The miscut needs class-preserving suffixes scarce enough to be outvoted.
        self.assertLess(report.class_preserving_fraction, 0.10)


@pytest.mark.slow
class TestFiveSymbolTrap(unittest.TestCase):
    def test_learned_within_the_bar(self):
        # Seed 0 loses Q; seed 1 recovers all six states, so the seed is pinned.
        target = build_five_symbol_trap()
        oracle_creator = lambda nm, s, _d=target: NoisyOracle(DFAOracle(_d), nm, s)
        tracker = RecordingTracker()
        dfa = learn_dfa_unchecked(
            oracle_creator, min_signal_strength=FIVE_SIGNAL, seed=0, tracker=tracker
        )

        # The stressor, read off the run rather than off the target: without a
        # round that cuts Q backwards there is nothing here to hold a bar against.
        cut = max(_miscut_mass(target, sufs) for sufs, _ in tracker.families)
        self.assertGreater(
            cut,
            0.005,
            f"no round cut more than {cut:.4f} of the prefix mass against the "
            f"language, where this target has measured 0.0132.  The merge this "
            f"test exists to hold a bar against is no longer being provoked, so "
            f"the assertion below proves nothing -- you might need to find "
            f"another reproducing target.",
        )

        accuracy, fp, fn = compute_dfa_accuracy(
            dfa, oracle_creator, symbols=FIVE_ALPHABET, sampler=DEFAULT_SAMPLER
        )
        if accuracy < 1 - assertion_allowed_error:
            self.fail(
                f"DFA incorrect (accuracy {accuracy:.4f}). "
                f"FP: {len(fp)}, FN: {len(fn)}"
            )
