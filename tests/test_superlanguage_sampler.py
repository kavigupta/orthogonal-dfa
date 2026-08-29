"""Drawing super-strings the way the compiled base strings stay uniform."""

import unittest

import numpy as np

from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

TAG, TGA, TAA = (3, 0, 2), (3, 2, 0), (3, 0, 0)


class TestSuperSampler(unittest.TestCase):
    def setUp(self):
        self.vocab = KmerVocabulary(kmers=(TAG, TGA, TAA), base_alphabet_size=4)
        self.sampler = SuperSampler(self.vocab, length=25)

    def test_length_and_range(self):
        out = self.sampler.sample(np.random.default_rng(0), self.vocab.alphabet_size)
        self.assertEqual(len(out), 25)
        self.assertTrue(all(0 <= s < self.vocab.alphabet_size for s in out))

    def test_length_attribute_is_symbol_count(self):
        self.assertEqual(self.sampler.length, 25)

    def test_samples_parse_of_uniform(self):
        out = self.sampler.sample(np.random.default_rng(3), self.vocab.alphabet_size)
        rng = np.random.default_rng(3)
        self.assertEqual(
            self.vocab.parse(self.vocab.compile(out, rng)),
            self.vocab.canonicalize(out),
        )

    def test_alphabet_size_mismatch_asserts(self):
        with self.assertRaises(AssertionError):
            self.sampler.sample(np.random.default_rng(0), self.vocab.alphabet_size + 1)


class TestSymbolWeights(unittest.TestCase):
    """The walk that picks a string reaching a state reads these, so they have to
    be what the sampler does rather than a description of it."""

    def test_declared_weights_are_what_it_draws(self):
        vocab = KmerVocabulary(kmers=(TAG, TGA), base_alphabet_size=4)
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
        vocab = KmerVocabulary(kmers=(TAG,), base_alphabet_size=4)
        weights = SuperSampler(vocab, 40).symbol_weights(vocab.alphabet_size)
        self.assertAlmostEqual(weights[0], 4**-3)
        self.assertAlmostEqual(sum(weights), 1.0)

    def test_weights_mismatch_asserts(self):
        vocab = KmerVocabulary(kmers=(TAG,), base_alphabet_size=4)
        with self.assertRaises(AssertionError):
            SuperSampler(vocab, 40).symbol_weights(vocab.alphabet_size + 1)


if __name__ == "__main__":
    unittest.main()
