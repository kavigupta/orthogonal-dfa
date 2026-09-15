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

    def test_alphabet_size_mismatch_asserts(self):
        with self.assertRaises(AssertionError):
            self.sampler.sample(np.random.default_rng(0), self.vocab.alphabet_size + 1)


class TestSymbolWeights(unittest.TestCase):
    """The walk that picks a string reaching a state reads these, so they have to
    be what the sampler does rather than a description of it."""

    def test_declared_weights_are_what_it_draws(self):
        # Long strings, so this measures the rate the parse settles at rather
        # than the first emission, which is the only one drawn from an
        # unconditioned stream.  Tolerance is four standard errors of the count
        # itself: an absolute one would let a kmer at 4**-3 be wrong by half.
        vocab = KmerVocabulary(kmers=(TAG, TGA), base_alphabet_size=4)
        sampler = SuperSampler(vocab, 100_000)
        declared = sampler.symbol_weights(vocab.alphabet_size)
        rng = np.random.default_rng(0)
        counts = np.zeros(vocab.alphabet_size)
        for _ in range(5):
            counts += np.bincount(
                np.frombuffer(sampler.sample(rng, vocab.alphabet_size), np.uint8),
                minlength=vocab.alphabet_size,
            )
        drawn, total = counts / counts.sum(), counts.sum()
        for symbol, weight in enumerate(declared):
            error = 4 * np.sqrt(weight * (1 - weight) / total)
            self.assertAlmostEqual(
                drawn[symbol],
                weight,
                delta=error,
                msg=f"symbol {symbol}: {drawn[symbol]:.5f} against {weight:.5f} "
                f"({(drawn[symbol] - weight) / weight:+.1%})",
            )

    def test_the_closed_form_holds_for_the_first_emission_only(self):
        # 4**-3 is the chance a 3-mer starts a stream nothing is yet known about.
        # That is the first emission; every later one follows a parse decision,
        # so the settled rate is not it, and the weights are the settled rate.
        vocab = KmerVocabulary(kmers=(TAG,), base_alphabet_size=4)
        rng = np.random.default_rng(0)
        first = SuperSampler(vocab, 1)
        draws = 50_000
        seen = (
            np.bincount(
                [first.sample(rng, vocab.alphabet_size)[0] for _ in range(draws)],
                minlength=vocab.alphabet_size,
            )
            / draws
        )
        closed = 4**-3
        self.assertAlmostEqual(
            seen[0], closed, delta=4 * np.sqrt(closed * (1 - closed) / draws)
        )
        settled = SuperSampler(vocab, 40).symbol_weights(vocab.alphabet_size)
        self.assertNotAlmostEqual(settled[0], closed, places=4)
        self.assertAlmostEqual(sum(settled), 1.0)

    def test_weights_mismatch_asserts(self):
        vocab = KmerVocabulary(kmers=(TAG,), base_alphabet_size=4)
        with self.assertRaises(AssertionError):
            SuperSampler(vocab, 40).symbol_weights(vocab.alphabet_size + 1)


if __name__ == "__main__":
    unittest.main()
