"""Sampling a string that reaches a state, for a learner that does not sample
evenly.

``denoise_accept_labels`` decides a state's label from strings reaching it, so
those strings have to be the ones the learner will meet: a walk that is uniform
over them answers about a population the learner never draws.
"""

import unittest

import numpy as np
from automata.fa.dfa import DFA

from orthogonal_dfa.l_star.dfa_utils import (
    count_paths_to_state,
    per_state_sample,
    sample_string_reaching_state,
    uniform_weights,
)
from orthogonal_dfa.l_star.sampler import Sampler, UniformSampler
from tests.dfas import PARITY

LENGTH = 30
RARE = 0.02


class _RareFirstSymbol(Sampler):
    """Symbol 0 at ``RARE`` of positions, the rest sharing the remainder.

    Stands in for any learner whose alphabet is not drawn evenly -- one symbol
    standing for a motif that is rare in the strings it draws, say.
    """

    def __init__(self, length: int, alphabet_size: int = 3):
        self.length = length
        self._alphabet_size = alphabet_size

    def sample(self, rng, alphabet_size):
        rest = (1 - RARE) / (alphabet_size - 1)
        probs = [RARE] + [rest] * (alphabet_size - 1)
        return rng.choice(alphabet_size, size=self.length, p=probs).tolist()


def _sink(alphabet_size):
    """One state, so a walk over it is the raw distribution and nothing else."""
    return DFA(
        states={0},
        input_symbols=set(range(alphabet_size)),
        transitions={0: {s: 0 for s in range(alphabet_size)}},
        initial_state=0,
        final_states={0},
        allow_partial=False,
    )


def _rate_of_symbol_zero(strings):
    return np.mean([[s == 0 for s in w] for w in strings])


class TestSymbolWeights(unittest.TestCase):
    def test_uniform_weighs_every_symbol_the_same(self):
        # Integers, so the path counts stay a count of strings.
        self.assertEqual(UniformSampler(LENGTH).symbol_weights(None, 3), [1, 1, 1])

    def test_only_the_ratios_of_the_weights_are_read(self):
        # Scaling every weight scales the mass at each depth by the same factor,
        # which the walk divides out -- so weights need not be probabilities.
        rng = np.random.default_rng(0)
        scaled = [3 * w for w in _RareFirstSymbol(LENGTH).symbol_weights(rng, 3)]
        plain = _RareFirstSymbol(LENGTH).symbol_weights(np.random.default_rng(0), 3)
        walk = lambda ws, seed: [
            sample_string_reaching_state(
                _sink(3),
                count_paths_to_state(_sink(3), 0, LENGTH, ws),
                np.random.default_rng(seed),
                ws,
            )
            for _ in range(200)
        ]
        self.assertAlmostEqual(
            _rate_of_symbol_zero(walk(scaled, 1)),
            _rate_of_symbol_zero(walk(plain, 1)),
            delta=0.01,
        )

    def test_weights_recover_the_samplers_own_rate(self):
        sampler = _RareFirstSymbol(LENGTH)
        weights = sampler.symbol_weights(np.random.default_rng(0), 3)
        self.assertAlmostEqual(weights[0], RARE, delta=0.01)


class TestWeightedWalk(unittest.TestCase):
    def test_the_walk_matches_the_sampler(self):
        sampler = _RareFirstSymbol(LENGTH)
        rng = np.random.default_rng(0)
        weights = sampler.symbol_weights(rng, 3)
        counts = count_paths_to_state(_sink(3), 0, LENGTH, weights)
        walked = [
            sample_string_reaching_state(_sink(3), counts, rng, weights)
            for _ in range(400)
        ]
        drawn = [sampler.sample(rng, 3) for _ in range(400)]
        self.assertAlmostEqual(
            _rate_of_symbol_zero(walked), _rate_of_symbol_zero(drawn), delta=0.02
        )

    def test_an_even_walk_does_not(self):
        # The failure this guards: 1/3 of positions where the learner puts 2%.
        rng = np.random.default_rng(0)
        counts = count_paths_to_state(_sink(3), 0, LENGTH, uniform_weights(_sink(3)))
        walked = [
            sample_string_reaching_state(
                _sink(3), counts, rng, uniform_weights(_sink(3))
            )
            for _ in range(400)
        ]
        self.assertGreater(_rate_of_symbol_zero(walked), 10 * RARE)

    def test_the_weights_have_to_reach_the_walk(self):
        # Counts alone score a symbol by the state it leads to, so where every
        # symbol leads to the same state they cannot express a preference.
        sampler = _RareFirstSymbol(LENGTH)
        rng = np.random.default_rng(0)
        weights = sampler.symbol_weights(rng, 3)
        counts = count_paths_to_state(_sink(3), 0, LENGTH, weights)
        walked = [
            sample_string_reaching_state(
                _sink(3), counts, rng, uniform_weights(_sink(3))
            )
            for _ in range(400)
        ]
        self.assertGreater(_rate_of_symbol_zero(walked), 10 * RARE)

    def test_it_matches_drawing_and_keeping_what_reaches(self):
        """Against ground truth: draw from the sampler, keep the strings that
        reach the state.  A branch leading somewhere with many completions is
        not more likely for it -- the sampler has to reach it too."""
        mod3 = DFA(
            states={0, 1, 2},
            input_symbols={0, 1},
            transitions={q: {0: (q + 1) % 3, 1: q} for q in range(3)},
            initial_state=0,
            final_states={1},
            allow_partial=False,
        )
        length, target = 12, 1
        sampler = _RareFirstSymbol(length, 2)
        rng = np.random.default_rng(0)
        weights = sampler.symbol_weights(rng, 2)

        def endpoint(w):
            q = 0
            for c in w:
                q = mod3.transitions[q][c]
            return q

        kept = []
        while len(kept) < 3000:
            w = sampler.sample(rng, 2)
            if endpoint(w) == target:
                kept.append(w)
        counts = count_paths_to_state(mod3, target, length, weights)
        walked = [
            sample_string_reaching_state(mod3, counts, rng, weights)
            for _ in range(3000)
        ]
        # Tight enough to separate this from weighting only the walk, which
        # leaves the completions behind a branch counted evenly and lands at
        # 1.05 where the sampler gives 1.00.
        self.assertAlmostEqual(
            np.mean([sum(1 for s in w if s == 0) for w in walked]),
            np.mean([sum(1 for s in w if s == 0) for w in kept]),
            delta=0.03,
        )

    def test_a_reachability_constraint_still_holds(self):
        # Weighting changes which strings are likely, never which are possible.
        rng = np.random.default_rng(0)
        weights = _RareFirstSymbol(LENGTH, 2).symbol_weights(rng, 2)
        counts = count_paths_to_state(PARITY, 1, LENGTH, weights)
        for _ in range(200):
            string = sample_string_reaching_state(PARITY, counts, rng, weights)
            self.assertEqual(sum(string) % 2, 1)


if __name__ == "__main__":
    unittest.main()


class TestPerStateSampleFollowsTheSampler(unittest.TestCase):
    def test_a_skewed_sampler_gets_skewed_strings(self):
        # Long enough that asking for distinct strings is not itself most of the
        # story: 40 of 2**19 leaves the weighting room to show.
        rng = np.random.default_rng(0)
        weights = _RareFirstSymbol(LENGTH, 2).symbol_weights(rng, 2)
        skewed = per_state_sample(PARITY, rng, LENGTH, 40, weights=weights)
        even = per_state_sample(
            PARITY, rng, LENGTH, 40, weights=uniform_weights(PARITY)
        )
        self.assertLess(_rate_of_symbol_zero(skewed), 4 * RARE)
        self.assertGreater(_rate_of_symbol_zero(even), 0.4)

    def test_an_even_sampler_still_draws_by_index(self):
        # The cheap path is only right where the two agree, so it must still be
        # taken: same pool from the same seed as before weights existed.
        rng = np.random.default_rng(7)
        pool = per_state_sample(
            PARITY, rng, LENGTH, 40, weights=uniform_weights(PARITY)
        )
        again = per_state_sample(
            PARITY,
            np.random.default_rng(7),
            LENGTH,
            40,
            weights=uniform_weights(PARITY),
        )
        self.assertEqual(sorted(map(tuple, pool)), sorted(map(tuple, again)))
