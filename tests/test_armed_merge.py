"""A rejecting state the learner folds into the accepting one, on most seeds.

``Q`` is non-final but reaches the absorbing accept state ``A`` on every symbol it does not
hold on, so all but a thin slice of suffixes carry it to an accepting endpoint and a family
built from the rest votes it accept.  The tree inherits that verdict and the hypothesis is
built from the tree, so the two AGREE on ``Q``: DFA/DT consistency clears its target and
synthesis stops with ``Q`` inside ``A``.

Three states rather than four.  A fourth state that escapes ``Q`` back to the counter spends
part of the same budget: ``class_preserving_fraction`` is the share of suffixes that never
drive ``Q`` into ``A``, and an escape route counts against it, so for a given share the state
must hold on fewer symbols and carries less mass.  Dropping it doubles ``Q``'s mass and leaves
only one way to return three states, which is the merge this file is about.

The merge is expensive here where a merge of two REJECTING classes is not: ``Q`` rejects and
``A`` accepts, so every string that truly ends in ``Q`` is accepted by the hypothesis, and
``Q`` carries 15% of the endpoint mass.
"""

import unittest

from automata.fa.dfa import DFA
from parameterized import parameterized

from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.learn import learn_dfa
from orthogonal_dfa.l_star.preconditions import (
    covered_accuracy_ceiling,
    satisfies_preconditions,
)
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.structures import NoisyOracle
from tests.lstar_common import assert_not_merged

ALPHABET = 12
#: Symbols that arm the trap.  One: two of them let the round separate ``Q`` and the target is
#: learned exactly.
ARMS = 1
#: Symbols ``Q`` holds on.  The rest reach ``A``, so this sets how many suffixes preserve
#: ``Q``'s class -- `holds/alphabet` to the sampler's length, which has to clear the 2% the
#: preconditions ask for without clearing it by much.
HOLDS = 10
LENGTH = 20
SIGNAL = 0.3

#: Seeds the rate is measured over, one test apiece: which of them merge is a property of the
#: suffixes a round draws, and a learn here is seconds rather than a test's whole clock.
SEEDS = 10


def build_target() -> DFA:
    arm = set(range(ALPHABET - ARMS, ALPHABET))
    return DFA(
        states={"c0", "Q", "A"},
        input_symbols=set(range(ALPHABET)),
        transitions={
            "c0": {c: ("Q" if c in arm else "c0") for c in range(ALPHABET)},
            "Q": {c: ("Q" if c < HOLDS else "A") for c in range(ALPHABET)},
            "A": {c: "A" for c in range(ALPHABET)},
        },
        initial_state="c0",
        final_states={"A"},
        allow_partial=False,
    )


class TestArmedMergeTarget(unittest.TestCase):
    """The construction, not the learner: cheap, and guards the stressor."""

    def test_admitted_and_learnable(self):
        target = build_target()
        sampler = UniformSampler(LENGTH)
        report = satisfies_preconditions(
            target, length=LENGTH, short_circuit=False, sampler=sampler
        )
        self.assertTrue(report.satisfied, report.reasons)
        # The merge is only reachable while the suffixes that preserve ``Q`` are scarce, and
        # only costly while ``Q`` carries mass.  Either drifting leaves the test below passing
        # without exercising anything.
        self.assertLess(report.class_preserving_fraction, 0.05)
        # Nothing about the sampler stops this being learned exactly, so the shortfall below
        # is the learner's and not the target's.
        self.assertAlmostEqual(
            covered_accuracy_ceiling(target, length=LENGTH, sampler=sampler), 1.0
        )


class TestArmedMergeLearned(unittest.TestCase):
    @parameterized.expand([(seed,) for seed in range(SEEDS)])
    def test_the_armed_state_is_not_merged(self, seed):
        target = build_target()
        oracle_creator = lambda nm, s, _d=target: NoisyOracle(DFAOracle(_d), nm, s)
        sampler = UniformSampler(LENGTH)
        dfa = learn_dfa(
            oracle_creator,
            min_signal_strength=SIGNAL,
            seed=seed,
            sampler=sampler,
        )
        # Graded at the length it was learned at: a hypothesis that merges ``Q`` still reads
        # well on strings long enough that almost all of them reach ``A`` anyway.
        assert_not_merged(
            self,
            dfa,
            target,
            oracle_creator=oracle_creator,
            symbols=ALPHABET,
            sampler=sampler,
        )
