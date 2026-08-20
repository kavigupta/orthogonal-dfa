"""Tests for the superlanguage vocabulary: the kmer + wildcard alphabet and its
invertible, fiber-uniform compilation to and from the base alphabet.

Uses the genomic stop codons as the kmers, since they are what the superlanguage
is built for, but nothing here needs anything beyond the vocabulary itself.
"""

import itertools
import unittest

import numpy as np

from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

# ACGT base alphabet: A=0, C=1, G=2, T=3.
TAG, TGA, TAA = (3, 0, 2), (3, 2, 0), (3, 0, 0)


class TestKmerVocabulary(unittest.TestCase):
    def test_alphabet_shape(self):
        v = KmerVocabulary(kmers=(TAG, TGA), base_alphabet_size=4, num_wildcards=2)
        self.assertEqual(v.num_kmers, 2)
        self.assertEqual(v.alphabet_size, 4)  # two kmers + X + Y
        self.assertEqual(v.unknown_symbol, 2)
        self.assertEqual(v.wildcard_symbols, (2, 3))
        self.assertTrue(v.is_unknown(2))
        self.assertTrue(v.is_unknown(3))
        self.assertFalse(v.is_unknown(0))

    def test_single_wildcard_vocabulary(self):
        v = KmerVocabulary(kmers=(TAG, TGA, TAA), base_alphabet_size=4, num_wildcards=1)
        self.assertEqual(v.alphabet_size, 4)
        self.assertEqual(v.wildcard_symbols, (3,))

    def test_compiled_length(self):
        v = KmerVocabulary(kmers=((0, 1), (2, 3, 0)), base_alphabet_size=4)
        self.assertEqual(v.compiled_length(0), 2)
        self.assertEqual(v.compiled_length(1), 3)
        self.assertEqual(v.compiled_length(v.unknown_symbol), 1)

    def test_compile_all_kmer_is_concatenation(self):
        # No X slots -> compile is deterministic concatenation of the kmers.
        v = KmerVocabulary(kmers=(TAG, TGA), base_alphabet_size=4)
        rng = np.random.default_rng(0)
        self.assertEqual(v.compile([0, 1, 0], rng), list(TAG) + list(TGA) + list(TAG))

    def test_rejects_bad_kmers(self):
        with self.assertRaises(AssertionError):
            KmerVocabulary(kmers=((0, 9),), base_alphabet_size=4)  # 9 out of range
        with self.assertRaises(AssertionError):
            KmerVocabulary(kmers=((),), base_alphabet_size=4)  # empty kmer
        with self.assertRaises(AssertionError):
            KmerVocabulary(kmers=((0,), (0,)), base_alphabet_size=4)  # duplicate
        with self.assertRaises(AssertionError):
            # prefix-related: (0,1) is a prefix of (0,1,2)
            KmerVocabulary(kmers=((0, 1), (0, 1, 2)), base_alphabet_size=4)


class TestParseCompile(unittest.TestCase):
    def setUp(self):
        self.vocab = KmerVocabulary(kmers=(TAG, TGA, TAA), base_alphabet_size=4)
        self.X = self.vocab.unknown_symbol

    def test_parse_greedy_match(self):
        # TAG then a lone A (not a stop) then TGA.
        base = list(TAG) + [0] + list(TGA)
        self.assertEqual(self.vocab.parse(base), [0, self.X, 1])

    def test_roundtrip_parse_of_compile(self):
        # The wildcards compile identically, so the round trip recovers the
        # super-string up to which wildcard was used.
        rng = np.random.default_rng(0)
        for _ in range(2000):
            n = int(rng.integers(1, 12))
            s = [int(rng.integers(self.vocab.alphabet_size)) for _ in range(n)]
            self.assertEqual(
                self.vocab.parse(self.vocab.compile(s, rng)),
                self.vocab.canonicalize(s),
            )

    def test_wildcards_compile_identically(self):
        # X and Y are interchangeable: swapping them leaves the fiber unchanged,
        # so the base oracle cannot distinguish them.
        x, y = self.vocab.wildcard_symbols[:2]
        a = self.vocab.compile([0, x, 1, x], np.random.default_rng(4))
        b = self.vocab.compile([0, y, 1, y], np.random.default_rng(4))
        self.assertEqual(a, b)

    def test_compile_of_parse_is_uniform(self):
        """``compile(parse(x))`` recovers a uniform base string for uniform ``x``."""
        rng = np.random.default_rng(1)
        xs = [rng.integers(4, size=24).tolist() for _ in range(2000)]
        compiled = self.vocab.compile_many(
            [self.vocab.parse(x) for x in xs],
            [np.random.default_rng(i) for i in range(len(xs))],
        )
        counts = np.bincount(np.concatenate(compiled), minlength=4)
        freqs = counts / counts.sum()
        self.assertLess(np.abs(freqs - 0.25).max(), 0.01)

    def test_compile_is_uniform_over_the_whole_fiber(self):
        """Every base string that parses back to ``s`` must come out equally often.

        Brute-forces the fiber of one super-string and chi-squares the sampler
        against it -- the sharp version of the marginal check above, and what the
        fiber-counting weights buy: drawing uniformly among the *locally* legal
        symbols instead skews this badly (chi2 ~ 11x the degrees of freedom).
        """
        x = self.vocab.unknown_symbol
        s = [x, x, 0, x, x, x]  # five wildcards around one TAG -> 8 base symbols
        want = self.vocab.canonicalize(s)
        length = sum(self.vocab.compiled_length(sym) for sym in s)
        fiber = {}
        for candidate in itertools.product(range(4), repeat=length):
            if self.vocab.parse(list(candidate)) == want:
                fiber[candidate] = len(fiber)

        count = 40 * len(fiber)
        compiled = self.vocab.compile_many(
            [s] * count, [np.random.default_rng(i) for i in range(count)]
        )
        counts = np.zeros(len(fiber))
        for base_string in compiled:
            key = tuple(base_string)
            self.assertIn(key, fiber, "compiled outside the fiber")
            counts[fiber[key]] += 1
        chi2 = ((counts - count / len(fiber)) ** 2 / (count / len(fiber))).sum()
        # Under uniformity chi2 ~ dof; allow a wide margin so the test is about
        # systematic skew, not luck. Equal weighting would land near 11 * dof.
        self.assertLess(chi2, 2 * (len(fiber) - 1))

    def test_compile_never_spells_a_stop_in_wildcard_regions(self):
        # A wildcard-only string must never compile to a base string containing a
        # stop codon (that is exactly what invertibility guarantees).
        rng = np.random.default_rng(2)
        stops = {TAG, TGA, TAA}
        wild = self.vocab.wildcard_symbols
        for _ in range(3000):
            s = [int(rng.choice(wild)) for _ in range(20)]
            b = self.vocab.compile(s, rng)
            self.assertFalse(
                any(tuple(b[i : i + 3]) in stops for i in range(len(b) - 2))
            )


if __name__ == "__main__":
    unittest.main()
