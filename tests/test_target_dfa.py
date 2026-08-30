"""Each oracle's ``target_dfa`` has to answer the same language the oracle does."""

import unittest

import numpy as np
from automata.fa.dfa import DFA
from parameterized import parameterized

from orthogonal_dfa.l_star.examples.benchmark_generator import (
    DFAOracle,
    sample_random_dfa,
)
from orthogonal_dfa.l_star.examples.bernoulli_parity import (
    AllFramesClosedOracle,
    BernoulliParityOracle,
    BernoulliRegex,
)
from orthogonal_dfa.l_star.examples.spliceai_oracle import SpliceModelOracle
from orthogonal_dfa.l_star.structures import NoisyOracle, SymmetricBernoulli

NOISELESS = SymmetricBernoulli(p_correct=1.0)


def _oracles():
    """One of each oracle that claims a DFA, named for the failure message."""
    yield "parity_mod2", NoisyOracle(BernoulliParityOracle(), NOISELESS, 0)
    yield "parity_mod9", NoisyOracle(
        BernoulliParityOracle(modulo=9, allowed_moduluses=(3, 6)), NOISELESS, 0
    )
    yield "regex_subsequence", NoisyOracle(
        BernoulliRegex(regex=r".*1010101.*"), NOISELESS, 0
    )
    yield "regex_two_runs", NoisyOracle(
        BernoulliRegex(regex=r".*1111.*1111.*"), NOISELESS, 0
    )
    # A dead end, so the compiled DFA is partial before it is completed.
    yield "regex_dead_end", NoisyOracle(BernoulliRegex(regex=r"1*"), NOISELESS, 0)
    yield "regex_three_symbols", NoisyOracle(
        BernoulliRegex(regex=r".*(111|000).*", alphabet_size=3), NOISELESS, 0
    )
    yield "all_frames_closed", NoisyOracle(AllFramesClosedOracle(), NOISELESS, 0)
    yield "dfa_backed", NoisyOracle(
        DFAOracle(sample_random_dfa(np.random.default_rng(0), num_states=6)),
        NOISELESS,
        0,
    )


def _accepts(dfa: DFA, word) -> bool:
    state = dfa.initial_state
    for symbol in word:
        state = dfa.transitions[state][symbol]
    return state in dfa.final_states


class TestTargetDFA(unittest.TestCase):
    @parameterized.expand(list(_oracles()))
    def test_agrees_with_the_oracle(self, _name, oracle):
        dfa = oracle.target_dfa()
        self.assertIsNotNone(dfa)
        rng = np.random.default_rng(0)
        for length in (0, 1, 3, 40):
            for _ in range(60):
                word = rng.integers(0, oracle.alphabet_size, size=length).tolist()
                self.assertEqual(
                    _accepts(dfa, word),
                    oracle.membership_query(word),
                    f"{_name} disagrees on {word}",
                )

    @parameterized.expand(list(_oracles()))
    def test_is_total_over_the_alphabet(self, _name, oracle):
        """Every prefix has to land somewhere, so callers can walk without guarding."""
        dfa = oracle.target_dfa()
        expected = set(range(oracle.alphabet_size))
        self.assertEqual(set(dfa.input_symbols), expected, _name)
        for state, row in dfa.transitions.items():
            self.assertEqual(set(row), expected, f"{_name} state {state} is partial")

    def test_a_non_regular_oracle_says_so(self):
        """The default is None, so an oracle we cannot write a DFA for admits it."""
        self.assertIsNone(SpliceModelOracle.target_dfa(None))


if __name__ == "__main__":
    unittest.main()
