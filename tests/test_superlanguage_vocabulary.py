"""Tests for the superlanguage vocabulary: the kmer + wildcard alphabet and the
greedy parse from the base alphabet into it.

Uses the genomic stop codons as the kmers, since they are what the superlanguage
is built for, but nothing here needs anything beyond the vocabulary itself.
"""

import unittest

from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

# ACGT base alphabet: A=0, C=1, G=2, T=3.
TAG, TGA, TAA = (3, 0, 2), (3, 2, 0), (3, 0, 0)


def wildcards(vocab):
    return tuple(range(vocab.num_kmers, vocab.alphabet_size))


class TestKmerVocabulary(unittest.TestCase):
    def test_alphabet_shape(self):
        v = KmerVocabulary(kmers=(TAG, TGA), base_alphabet_size=4, num_wildcards=2)
        self.assertEqual(v.num_kmers, 2)
        self.assertEqual(v.alphabet_size, 4)  # two kmers + X + Y
        self.assertEqual(v.unknown_symbol, 2)
        self.assertEqual(wildcards(v), (2, 3))
        self.assertTrue(v.is_unknown(2))
        self.assertTrue(v.is_unknown(3))
        self.assertFalse(v.is_unknown(0))

    def test_single_wildcard_vocabulary(self):
        v = KmerVocabulary(kmers=(TAG, TGA, TAA), base_alphabet_size=4, num_wildcards=1)
        self.assertEqual(v.alphabet_size, 4)
        self.assertEqual(wildcards(v), (3,))

    def test_vocabulary_with_no_kmers(self):
        v = KmerVocabulary(kmers=(), base_alphabet_size=4, num_wildcards=2)
        self.assertEqual(v.alphabet_size, 2)
        self.assertEqual(v.parse([0, 3, 1]), [v.unknown_symbol] * 3)

    def test_canonicalize_collapses_the_wildcards(self):
        v = KmerVocabulary(kmers=(TAG, TGA), base_alphabet_size=4, num_wildcards=2)
        x, y = wildcards(v)
        self.assertEqual(v.canonicalize([0, x, 1, y]), [0, x, 1, x])

    def test_rejects_bad_kmers(self):
        with self.assertRaises(AssertionError):
            KmerVocabulary(kmers=((0, 9),), base_alphabet_size=4)  # 9 out of range
        with self.assertRaises(AssertionError):
            KmerVocabulary(kmers=((),), base_alphabet_size=4)  # empty kmer
        with self.assertRaises(AssertionError):
            KmerVocabulary(kmers=((0,), (0,)), base_alphabet_size=4)  # duplicate
        with self.assertRaises(AssertionError):
            # prefix-related: (0,1) is a prefix of (0,1,2), so parse could not tell
            # which of the two a base string starting 0 1 2 spells
            KmerVocabulary(kmers=((0, 1), (0, 1, 2)), base_alphabet_size=4)


class TestParse(unittest.TestCase):
    def setUp(self):
        self.vocab = KmerVocabulary(kmers=(TAG, TGA, TAA), base_alphabet_size=4)
        self.X = self.vocab.unknown_symbol

    def test_parse_greedy_match(self):
        # TAG then a lone A (not a stop) then TGA.
        base = list(TAG) + [0] + list(TGA)
        self.assertEqual(self.vocab.parse(base), [0, self.X, 1])

    def test_parse_of_the_empty_string(self):
        self.assertEqual(self.vocab.parse([]), [])

    def test_a_partial_kmer_at_the_end_is_wildcards(self):
        self.assertEqual(self.vocab.parse([3, 0]), [self.X, self.X])

    def test_parse_is_leftmost_when_occurrences_overlap(self):
        # AA occurs at 0 and at 1 in AAA; the leftmost wins and the odd symbol
        # is left over as a wildcard.
        vocab = KmerVocabulary(kmers=((0, 0),), base_alphabet_size=4)
        self.assertEqual(vocab.parse([0, 0, 0]), [0, vocab.unknown_symbol])
        self.assertEqual(vocab.parse([0, 0, 0, 0]), [0, 0])

    def test_a_kmer_inside_an_earlier_match_is_not_seen(self):
        # ACG is taken whole, so the CG starting one symbol in is never looked at.
        vocab = KmerVocabulary(kmers=((0, 1, 2), (1, 2)), base_alphabet_size=4)
        self.assertEqual(vocab.parse([0, 1, 2]), [0])
        self.assertEqual(vocab.parse([3, 1, 2]), [vocab.unknown_symbol, 1])


if __name__ == "__main__":
    unittest.main()
