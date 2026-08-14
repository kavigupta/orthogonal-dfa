# pylint: disable=duplicate-code
import itertools
import unittest

import numpy as np
from automata.fa.dfa import DFA
from parameterized import parameterized

from orthogonal_dfa.l_star.dfa_utils import (
    count_paths_to_state,
    per_state_sample,
    rank_string_reaching_state,
    states_intermediate,
    unrank_string_reaching_state,
)

# Even number of 1s; strongly connected over {0, 1}.
PARITY = DFA(
    states={0, 1},
    input_symbols={0, 1},
    transitions={0: {0: 0, 1: 1}, 1: {0: 1, 1: 0}},
    initial_state=0,
    final_states={0},
    allow_partial=False,
)

# mod-3 counter on 1s; strongly connected over {0, 1}.
MOD3 = DFA(
    states={0, 1, 2},
    input_symbols={0, 1},
    transitions={0: {0: 0, 1: 1}, 1: {0: 1, 1: 2}, 2: {0: 2, 1: 0}},
    initial_state=0,
    final_states={1},
    allow_partial=False,
)

# States 0,1,2 are transient: no fixed-length string ends in them (reachable == 0);
# 3 and 4 are absorbing sinks.
DIFFICULT09 = DFA(
    states={0, 1, 2, 3, 4},
    input_symbols={0, 1},
    transitions={
        0: {0: 1, 1: 1},
        1: {0: 2, 1: 2},
        2: {0: 3, 1: 4},
        3: {0: 3, 1: 3},
        4: {0: 4, 1: 4},
    },
    initial_state=0,
    final_states={3},
    allow_partial=False,
)

# Three-symbol alphabet, mod-2 counter on symbol 2.
TERNARY = DFA(
    states={0, 1},
    input_symbols={0, 1, 2},
    transitions={0: {0: 0, 1: 0, 2: 1}, 1: {0: 1, 1: 1, 2: 0}},
    initial_state=0,
    final_states={0},
    allow_partial=False,
)

# Single absorbing state: every string of any length reaches it, so the number
# reachable is |alphabet| ** length -- used to exercise the > int64 branch.
SELF_LOOP = DFA(
    states={0},
    input_symbols={0, 1},
    transitions={0: {0: 0, 1: 0}},
    initial_state=0,
    final_states={0},
    allow_partial=False,
)

ALL_DFAS = [
    ("parity", PARITY),
    ("mod3", MOD3),
    ("difficult09", DIFFICULT09),
    ("ternary", TERNARY),
    ("self_loop", SELF_LOOP),
]


def enumerate_by_end_state(dfa, length):
    """Map each state to every length-``length`` string that ends there."""
    syms = sorted(dfa.input_symbols)
    by_state = {}
    for string in itertools.product(syms, repeat=length):
        end = states_intermediate(dfa.initial_state, string, dfa)[-1]
        by_state.setdefault(end, []).append(list(string))
    return by_state


class TestRankUnrank(unittest.TestCase):
    @parameterized.expand(
        [(name, dfa, length) for name, dfa in ALL_DFAS for length in range(5)]
    )
    def test_rank_unrank_is_a_bijection(self, _name, dfa, length):
        by_state = enumerate_by_end_state(dfa, length)
        for state in sorted(dfa.states):
            counts = count_paths_to_state(dfa, state, length)
            reachable = counts[length][dfa.initial_state]
            strings = by_state.get(state, [])
            self.assertEqual(len(strings), reachable)

            ranks = [rank_string_reaching_state(dfa, counts, s) for s in strings]
            # Ranks are exactly 0..reachable-1, each used once.
            self.assertEqual(sorted(ranks), list(range(reachable)))
            # unrank inverts rank.
            for s, r in zip(strings, ranks):
                self.assertEqual(unrank_string_reaching_state(dfa, counts, r), s)
            # unrank of every index lands on this state with the right length.
            for r in range(reachable):
                s = unrank_string_reaching_state(dfa, counts, r)
                self.assertEqual(len(s), length)
                self.assertEqual(
                    states_intermediate(dfa.initial_state, s, dfa)[-1], state
                )

    def test_rank_unrank_beyond_int64(self):
        # 2 ** 70 length-70 strings reach the single state -- well past int64.
        length = 70
        counts = count_paths_to_state(SELF_LOOP, 0, length)
        reachable = counts[length][SELF_LOOP.initial_state]
        self.assertEqual(reachable, 2**length)
        for index in [0, 1, 2**63, 2**63 + 1, 2**70 - 1, 12345678901234567890]:
            s = unrank_string_reaching_state(SELF_LOOP, counts, index)
            self.assertEqual(len(s), length)
            self.assertEqual(rank_string_reaching_state(SELF_LOOP, counts, s), index)


class TestPerStateSample(unittest.TestCase):
    @parameterized.expand(
        [
            (name, dfa, length, per_state)
            for name, dfa in ALL_DFAS
            for length in range(1, 5)
            for per_state in (1, 3, 8, 100)
        ]
    )
    def test_coverage_and_distinctness(self, _name, dfa, length, per_state):
        rng = np.random.default_rng(0)
        pool = per_state_sample(dfa, rng, length, per_state)

        # Every string has the requested length and is globally distinct.
        self.assertTrue(all(len(s) == length for s in pool))
        self.assertEqual(len(pool), len({tuple(s) for s in pool}))

        by_state = enumerate_by_end_state(dfa, length)
        counts_in_pool = {}
        for s in pool:
            end = states_intermediate(dfa.initial_state, s, dfa)[-1]
            counts_in_pool[end] = counts_in_pool.get(end, 0) + 1

        for state in dfa.states:
            reachable = len(by_state.get(state, []))
            expected = min(per_state, reachable)
            self.assertEqual(counts_in_pool.get(state, 0), expected)

    @parameterized.expand([(name, dfa) for name, dfa in ALL_DFAS])
    def test_existing_fills_quota_without_oversampling(self, _name, dfa):
        length, per_state = 4, 6
        rng = np.random.default_rng(1)

        existing = per_state_sample(dfa, rng, length, per_state)
        # Feeding the previous pool back in must not grow it: the quota is
        # already met, so nothing new is added.
        again = per_state_sample(dfa, rng, length, per_state, existing=existing)

        def by_state_counts(pool):
            out = {}
            for s in pool:
                end = states_intermediate(dfa.initial_state, s, dfa)[-1]
                out[end] = out.get(end, 0) + 1
            return out

        self.assertEqual(by_state_counts(existing), by_state_counts(again))

    def test_existing_wrong_length_is_ignored(self):
        # An existing string of the wrong length cannot count toward coverage.
        length, per_state = 3, 4
        rng = np.random.default_rng(2)
        junk = [[0, 1]]  # length 2, not 3
        pool = per_state_sample(PARITY, rng, length, per_state, existing=junk)
        self.assertTrue(all(len(s) == length for s in pool))
        by_state = enumerate_by_end_state(PARITY, length)
        counts_in_pool = {}
        for s in pool:
            end = states_intermediate(PARITY.initial_state, s, PARITY)[-1]
            counts_in_pool[end] = counts_in_pool.get(end, 0) + 1
        for state in PARITY.states:
            self.assertEqual(
                counts_in_pool.get(state, 0),
                min(per_state, len(by_state.get(state, []))),
            )

    def test_existing_strings_are_reused_when_present(self):
        # Existing strings that reach a state are kept rather than re-sampled
        # around, up to the quota.
        length, per_state = 3, 2
        rng = np.random.default_rng(3)
        by_state = enumerate_by_end_state(PARITY, length)
        # Two length-3 strings that end in state 1 (odd number of 1s).
        existing = [s for s in by_state[1]][:2]
        pool = per_state_sample(PARITY, rng, length, per_state, existing=existing)
        pool_reaching_1 = [
            s
            for s in pool
            if states_intermediate(PARITY.initial_state, s, PARITY)[-1] == 1
        ]
        self.assertEqual(
            {tuple(s) for s in pool_reaching_1}, {tuple(s) for s in existing}
        )

    def test_large_space_uses_string_sampler_branch(self):
        # reachable = 2 ** 70 forces the > int64 fallback; still returns the
        # requested count of distinct, correct-length strings.
        rng = np.random.default_rng(4)
        pool = per_state_sample(SELF_LOOP, rng, 70, 8)
        self.assertEqual(len(pool), 8)
        self.assertEqual(len({tuple(s) for s in pool}), 8)
        self.assertTrue(all(len(s) == 70 for s in pool))

    def test_sampling_is_uniform_without_replacement(self):
        # Over many draws of a size-4 subset from the 8 length-3 strings, each
        # string should appear about half the time.  Loose bound -- this checks
        # the distribution, not any single seed.
        length, per_state, runs = 3, 4, 8000
        space = list(itertools.product((0, 1), repeat=length))
        rng = np.random.default_rng(5)
        appearances = {s: 0 for s in space}
        for _ in range(runs):
            pool = per_state_sample(SELF_LOOP, rng, length, per_state)
            self.assertEqual(len(pool), per_state)
            for s in pool:
                appearances[tuple(s)] += 1
        for s in space:
            freq = appearances[s] / runs
            self.assertAlmostEqual(freq, per_state / len(space), delta=0.05)


if __name__ == "__main__":
    unittest.main()
