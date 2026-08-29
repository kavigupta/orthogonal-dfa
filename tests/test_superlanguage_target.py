"""The language a base DFA induces over the super alphabet, as a DFA."""

import unittest

import numpy as np
from automata.fa.dfa import DFA

from orthogonal_dfa.l_star.examples.bernoulli_parity import AllFramesClosedOracle
from orthogonal_dfa.l_star.structures import SymmetricBernoulli
from orthogonal_dfa.superlanguage.target import super_target_dfa
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

TAG, TGA, TAA = (3, 0, 2), (3, 2, 0), (3, 0, 0)

# {w : w[0] == A}, which reads a symbol a wildcard could have filled either way.
FIRST_IS_A = DFA(
    states={0, 1, 2},
    input_symbols={0, 1, 2, 3},
    transitions={
        0: {0: 1, 1: 2, 2: 2, 3: 2},
        1: {c: 1 for c in range(4)},
        2: {c: 2 for c in range(4)},
    },
    initial_state=0,
    final_states={1},
    allow_partial=False,
)


class TestSuperTargetDfa(unittest.TestCase):
    def setUp(self):
        self.vocab = KmerVocabulary(kmers=(TAG, TGA, TAA), base_alphabet_size=4)
        self.base = AllFramesClosedOracle(
            noise_model=SymmetricBernoulli(1.0), seed=0
        ).target_dfa()
        self.dfa = super_target_dfa(self.vocab, self.base)

    def test_agrees_with_the_base_on_what_a_super_string_compiles_to(self):
        # The definition: a super-string is in the language when the base strings
        # it compiles to are.  Drawn evenly here rather than the way the learner
        # draws, since either reaches every part of the DFA.
        rng = np.random.default_rng(11)
        strings = [
            rng.integers(0, self.vocab.alphabet_size, size=12).tolist()
            for _ in range(600)
        ]
        np.testing.assert_array_equal(
            np.array([self.dfa.accepts_input(s) for s in strings], dtype=bool),
            np.array(
                [self.base.accepts_input(self.vocab.compile(s, rng)) for s in strings],
                dtype=bool,
            ),
        )

    def test_a_fill_cannot_change_the_answer(self):
        # Every compilation of a super-string has to land the same way, or the
        # language is not a function of the super-string at all.
        rng = np.random.default_rng(5)
        for _ in range(200):
            s = rng.integers(0, self.vocab.alphabet_size, size=12).tolist()
            answers = {
                self.base.accepts_input(self.vocab.compile(s, rng)) for _ in range(8)
            }
            self.assertEqual(len(answers), 1, f"{s} compiles both ways")

    def test_smaller_than_the_base_target(self):
        # A stop codon is one super-symbol rather than a three-symbol path.
        self.assertLess(len(self.dfa.states), len(self.base.states))

    def test_wildcards_cannot_forge_a_stop(self):
        # [TAG, X, TAG, X, TAG] closes all three frames whatever the X fill.
        x = self.vocab.unknown_symbol
        self.assertTrue(self.dfa.accepts_input([0, x, 0, x, 0]))
        self.assertFalse(self.dfa.accepts_input([0, x, 0]))

    def test_rejects_a_base_that_reads_the_fill(self):
        # FIRST_IS_A answers on a symbol a wildcard could have filled either way,
        # so no super-string has one answer and there is no DFA to build.
        with self.assertRaises(AssertionError):
            super_target_dfa(self.vocab, FIRST_IS_A)


if __name__ == "__main__":
    unittest.main()
