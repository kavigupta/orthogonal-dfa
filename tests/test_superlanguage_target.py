"""The lifted oracle's language, read as a DFA over the super alphabet."""

import unittest

import numpy as np
from automata.fa.dfa import DFA

from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.examples.bernoulli_parity import AllFramesClosedOracle
from orthogonal_dfa.l_star.structures import Oracle, SymmetricBernoulli
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
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


class _NoTargetOracle(Oracle):
    """Keeps ``Oracle``'s default ``target_dfa``, which answers None."""

    @property
    def alphabet_size(self):
        return 4

    def membership_query(self, string):
        return False


class TestSuperTargetDfa(unittest.TestCase):
    def setUp(self):
        self.vocab = KmerVocabulary(kmers=(TAG, TGA, TAA), base_alphabet_size=4)
        self.base = AllFramesClosedOracle(noise_model=SymmetricBernoulli(1.0), seed=0)
        self.oracle = LiftedOracle(self.base, self.vocab, num_compilations=1, seed=0)

    def test_agrees_with_the_lifted_oracle(self):
        dfa = self.oracle.target_dfa()
        sampler = SuperSampler(self.vocab, 40)
        rng = np.random.default_rng(11)
        strings = [sampler.sample(rng, self.vocab.alphabet_size) for _ in range(1000)]
        np.testing.assert_array_equal(
            np.array([dfa.accepts_input(s) for s in strings], dtype=bool),
            self.oracle.membership_queries(strings),
        )

    def test_smaller_than_the_base_target(self):
        # A stop codon is one super-symbol rather than a three-symbol path.
        self.assertLess(
            len(self.oracle.target_dfa().states), len(self.base.target_dfa().states)
        )

    def test_wildcards_cannot_forge_a_stop(self):
        # [TAG, X, TAG, X, TAG] closes all three frames whatever the X fill.
        dfa = self.oracle.target_dfa()
        x = self.vocab.unknown_symbol
        self.assertTrue(dfa.accepts_input([0, x, 0, x, 0]))
        self.assertFalse(dfa.accepts_input([0, x, 0]))

    def test_none_when_the_base_has_no_target(self):
        oracle = LiftedOracle(_NoTargetOracle(), self.vocab, num_compilations=1, seed=0)
        self.assertIsNone(oracle.target_dfa())

    def test_rejects_a_base_oracle_that_reads_the_fill(self):
        oracle = LiftedOracle(
            DFAOracle(SymmetricBernoulli(1.0), 0, FIRST_IS_A),
            self.vocab,
            num_compilations=1,
            seed=0,
        )
        with self.assertRaises(AssertionError):
            oracle.target_dfa()


if __name__ == "__main__":
    unittest.main()


class TestSuperSamplerWeights(unittest.TestCase):
    """The walk that picks strings reaching a state reads these, so they have to
    be what the sampler does rather than a description of it."""

    def test_declared_weights_are_what_it_draws(self):
        vocab = KmerVocabulary(kmers=((3, 0, 2), (3, 2, 0)), base_alphabet_size=4)
        sampler = SuperSampler(vocab, 40)
        declared = sampler.symbol_weights(vocab.alphabet_size)
        rng = np.random.default_rng(0)
        drawn = [
            s for _ in range(500) for s in sampler.sample(rng, vocab.alphabet_size)
        ]
        for symbol, weight in enumerate(declared):
            seen = drawn.count(symbol) / len(drawn)
            self.assertAlmostEqual(seen, weight, delta=0.01, msg=f"symbol {symbol}")

    def test_a_kmer_is_as_likely_as_its_base_symbols(self):
        # Prefix-free, so a kmer is emitted exactly where its own base symbols
        # fall: 4**-3 for a 3-mer over 4 base symbols.
        vocab = KmerVocabulary(kmers=((3, 0, 2),), base_alphabet_size=4)
        weights = SuperSampler(vocab, 40).symbol_weights(vocab.alphabet_size)
        self.assertAlmostEqual(weights[0], 4**-3)
        self.assertAlmostEqual(sum(weights), 1.0)
