"""The merged-state check on members drawn from known true states of the armed
target, read against the pool the learner's own search builds: members of one state
must not split, and a minority of the other label must."""

import unittest
from collections import Counter
from functools import lru_cache

import numpy as np
import pytest
from parameterized import parameterized

from orthogonal_dfa.l_star.cluster import sample_suffix_family
from orthogonal_dfa.l_star.counterexample_synthesis import SPLIT_SCAN_ALPHA
from orthogonal_dfa.l_star.dfa_utils import (
    count_paths_to_state,
    sample_string_reaching_state,
    uniform_weights,
)
from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.examples.bernoulli_parity import AllFramesClosedOracle
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.prefix_populations import PoolState
from orthogonal_dfa.l_star.prefix_suffix_tracker import SearchConfig
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.state_split import split_by_looks, state_split
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli, NoisyOracle
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary
from tests.test_lstar import ARMED_ALPHABET, ARMED_LENGTH, build_armed_target

NOISE = [(0.2, 0.8), (0.35, 0.65), (0.6, 0.9), (0.35, 0.95), (0.05, 0.65)]
#: Seed 4 at (0.35, 0.65) builds a pool whose search leaves most of its
#: class-preserving suffixes out of the family.
SEEDS = [0, 4]

MERGED_MINORITY_MASS = SearchConfig.merged_minority_mass


def _oracle(p_0, p_1, seed):
    return NoisyOracle(
        DFAOracle(build_armed_target()), AsymmetricBernoulli(p_0=p_0, p_1=p_1), seed
    )


@lru_cache(maxsize=None)
def _pool(p_0, p_1, seed):
    """The suffixes a round leaves fully observed once its family search is done,
    which is the pool the check reads at the end of that round."""
    pst = build_pst(
        lambda nm, s: NoisyOracle(DFAOracle(build_armed_target()), nm, s),
        min_signal_strength=(p_1 - p_0) / 2,
        seed=seed,
        sampler=UniformSampler(ARMED_LENGTH),
        noise_model=AsymmetricBernoulli(p_0=p_0, p_1=p_1),
    )
    uniform = [p for p, keep in zip(pst.table.prefixes, pst.table.representative) if keep]
    sample_suffix_family(pst, pst.table.intern_suffix(b""), PoolState(uniform))
    return tuple(pst.table.suffix(v) for v in pst.table.fully_observed())


def _true_mass(state):
    target = build_armed_target()
    reaching = count_paths_to_state(target, state, ARMED_LENGTH, uniform_weights(target))
    return reaching[ARMED_LENGTH][target.initial_state] / ARMED_ALPHABET**ARMED_LENGTH


class _Mixture:
    """Members of the union of the states in ``masses``, each drawn in proportion
    to its mass there, remembering which each one is."""

    def __init__(self, masses, rng):
        target = build_armed_target()
        self._target = target
        self._weights = uniform_weights(target)
        self._states = sorted(masses)
        mass = np.array([masses[q] for q in self._states])
        self._p = mass / mass.sum()
        self._reaching = {
            q: count_paths_to_state(target, q, ARMED_LENGTH, self._weights)
            for q in self._states
        }
        self._rng = rng
        self.truth = {}

    def __call__(self, count):
        drawn = []
        for k in self._rng.choice(len(self._states), size=count, p=self._p):
            state = self._states[k]
            prefix = sample_string_reaching_state(
                self._target, self._reaching[state], self._rng, self._weights
            )
            self.truth[prefix] = state
            drawn.append(prefix)
        return drawn

    def suffix(self):
        return UniformSampler(ARMED_LENGTH).sample(self._rng, alphabet_size=ARMED_ALPHABET)


def _check(masses, p_0, p_1, seed):
    rng = np.random.default_rng(seed)
    draw = _Mixture(masses, rng)
    split = split_by_looks(
        draw,
        _pool(p_0, p_1, seed),
        _oracle(p_0, p_1, seed),
        signal=(p_1 - p_0) / 2,
        minority_share=MERGED_MINORITY_MASS / sum(masses.values()),
        alpha=SPLIT_SCAN_ALPHA,
        rng=rng,
    )
    return split, draw.truth


class TestOneStateDoesNotSplit(unittest.TestCase):
    @parameterized.expand(
        [
            (state, p_0, p_1, seed)
            for state in ("c0", "Q", "A")
            for p_0, p_1 in NOISE
            for seed in SEEDS
        ]
    )
    def test_pure(self, state, p_0, p_1, seed):
        split, _ = _check({state: _true_mass(state)}, p_0, p_1, seed)
        self.assertIsNone(split)


class TestOppositeLabelsSplit(unittest.TestCase):
    def _assert_split_along(self, masses, p_0, p_1, seed):
        split, truth = _check(masses, p_0, p_1, seed)
        self.assertIsNotNone(split, "a minority of the other label went unseen")
        majority = [
            Counter(truth[m] for m in group).most_common(1)[0][0]
            for group in split.groups
        ]
        self.assertEqual(sorted(majority), sorted(masses))

    @parameterized.expand([(*noise, seed) for noise in NOISE for seed in SEEDS])
    def test_at_their_own_masses(self, p_0, p_1, seed):
        masses = {q: _true_mass(q) for q in ("A", "Q")}
        self._assert_split_along(masses, p_0, p_1, seed)

    @parameterized.expand([(*noise, seed) for noise in NOISE for seed in SEEDS])
    def test_the_least_minority_worth_finding(self, p_0, p_1, seed):
        masses = {"A": _true_mass("A"), "Q": MERGED_MINORITY_MASS}
        self._assert_split_along(masses, p_0, p_1, seed)


@pytest.mark.slow
class TestAllFramesClosedDoesNotSplit(unittest.TestCase):
    def test_every_true_state(self):
        # The target at signal 0.2 is where a check against the boundary's band
        # flagged a pure state: the band sits on a boundary the family re-estimates.
        vocab = KmerVocabulary(
            kmers=((3, 0, 2), (3, 2, 0), (3, 0, 0)), base_alphabet_size=4
        )
        base = AllFramesClosedOracle()
        pst = build_pst(
            lambda nm, s: NoisyOracle(LiftedOracle(base, vocab, seed=s), nm, s),
            min_signal_strength=0.2,
            seed=0,
            sampler=SuperSampler(vocab, 40),
        )
        uniform = [
            p for p, keep in zip(pst.table.prefixes, pst.table.representative) if keep
        ]
        sample_suffix_family(pst, pst.table.intern_suffix(b""), PoolState(uniform))
        target = LiftedOracle(base, vocab, seed=0).target_dfa()
        split = [
            state
            for state in sorted(target.states)
            if state_split(pst, target, state, alpha=SPLIT_SCAN_ALPHA) is not None
        ]
        self.assertEqual(split, [])
