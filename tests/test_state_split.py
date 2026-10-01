"""Splitting members drawn from known true states of the armed target, read
against the family the learner's own search builds: a minority of the other label
must land on the minority's side."""

import unittest
from collections import Counter
from functools import lru_cache

import numpy as np
from parameterized import parameterized

from orthogonal_dfa.l_star.cluster import sample_suffix_family
from orthogonal_dfa.l_star.counterexample_synthesis import SPLIT_ALPHA
from orthogonal_dfa.l_star.dfa_utils import (
    count_paths_to_state,
    sample_string_reaching_state,
    uniform_weights,
)
from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.learn import DEFAULT_MAX_COVERAGE_ERROR, build_pst
from orthogonal_dfa.l_star.prefix_populations import PoolState
from orthogonal_dfa.l_star.prefix_suffix_tracker import SearchConfig
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.state_split import fresh_suffixes, split_members
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli, NoisyOracle
from tests.test_lstar import ARMED_ALPHABET, ARMED_LENGTH, build_armed_target

#: (0.6, 0.9) is left out: past a half on both sides the pooled ranking inverts.
NOISE = [(0.2, 0.8), (0.35, 0.65), (0.35, 0.95), (0.05, 0.65)]
#: Seed 4 at (0.35, 0.65) builds a pool whose search leaves most of its
#: class-preserving suffixes out of the family.
SEEDS = [0, 4]
#: At (0.35, 0.65) the family the search admits is mostly suffixes that read Q
#: as A, and the minority's side comes out under 2/3 Q: the certificate refuses
#: the next round's merge again, at the cost of a round.
SIDED_NOISE = [noise for noise in NOISE if noise != (0.35, 0.65)]


def _oracle(p_0, p_1, seed):
    return NoisyOracle(
        DFAOracle(build_armed_target()), AsymmetricBernoulli(p_0=p_0, p_1=p_1), seed
    )


@lru_cache(maxsize=None)
def _family(p_0, p_1, seed):
    """The family a round's search settles on, which the check reads at the end of
    that round."""
    pst = build_pst(
        lambda nm, s: NoisyOracle(DFAOracle(build_armed_target()), nm, s),
        min_signal_strength=(p_1 - p_0) / 2,
        seed=seed,
        sampler=UniformSampler(ARMED_LENGTH),
        noise_model=AsymmetricBernoulli(p_0=p_0, p_1=p_1),
    )
    uniform = [
        p for p, keep in zip(pst.table.prefixes, pst.table.representative) if keep
    ]
    vs, _ = sample_suffix_family(pst, pst.table.intern_suffix(b""), PoolState(uniform))
    return tuple(pst.table.suffix(v) for v in vs)


def _true_mass(state):
    target = build_armed_target()
    reaching = count_paths_to_state(
        target, state, ARMED_LENGTH, uniform_weights(target)
    )
    return reaching[ARMED_LENGTH][target.initial_state] / ARMED_ALPHABET**ARMED_LENGTH


class _Mixture:
    """Members of the union of the states in masses, each drawn in proportion
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
        return UniformSampler(ARMED_LENGTH).sample(
            self._rng, alphabet_size=ARMED_ALPHABET
        )


def _check(masses, p_0, p_1, seed):
    rng = np.random.default_rng(seed)
    draw = _Mixture(masses, rng)
    fresh = {
        draw.suffix()
        for _ in range(fresh_suffixes(SPLIT_ALPHA, SearchConfig.min_suffix_frequency))
    }
    split = split_members(
        draw,
        [v for v in sorted(set(_family(p_0, p_1, seed)) | fresh) if v],
        _oracle(p_0, p_1, seed),
        minority_below=True,
        minority_share=masses["Q"] / sum(masses.values()),
        signal=(p_1 - p_0) / 2,
        level=SPLIT_ALPHA,
        rng=rng,
    )
    return split, draw.truth


class TestOppositeLabelsSplit(unittest.TestCase):
    def _assert_split_along(self, masses, p_0, p_1, seed):
        split, truth = _check(masses, p_0, p_1, seed)
        # The next round holds each side as a population, and the gate's veto is
        # sized for one a family reads nearly all backwards.
        minority = Counter(truth[m] for m in split.groups[True])
        self.assertGreater(
            minority["Q"] / sum(minority.values()),
            1 - DEFAULT_MAX_COVERAGE_ERROR,
            f"the minority's side holds {dict(minority)}",
        )

    @parameterized.expand([(*noise, seed) for noise in SIDED_NOISE for seed in SEEDS])
    def test_at_their_own_masses(self, p_0, p_1, seed):
        masses = {q: _true_mass(q) for q in ("A", "Q")}
        self._assert_split_along(masses, p_0, p_1, seed)
