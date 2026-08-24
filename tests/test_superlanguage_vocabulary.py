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

    def test_vocabulary_with_no_kmers(self):
        # All wildcard, no kmers: every super-symbol is one base symbol, which is
        # what max_kmer_length reports so callers can size a base string for it.
        v = KmerVocabulary(kmers=(), base_alphabet_size=4, num_wildcards=2)
        self.assertEqual(v.max_kmer_length, 1)
        self.assertEqual(v.alphabet_size, 2)
        rng = np.random.default_rng(0)
        s = [v.unknown_symbol, v.wildcard_symbols[1], v.unknown_symbol]
        out = v.compile(s, rng)
        self.assertEqual(len(out), 3)
        self.assertEqual(v.parse(out), v.canonicalize(s))

    def test_max_kmer_length(self):
        v = KmerVocabulary(kmers=((0, 1), (2, 3, 0)), base_alphabet_size=4)
        self.assertEqual(v.max_kmer_length, 3)

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
            # prefix-related: (0,1) is a prefix of (0,1,2), so what follows the
            # short one could grow it into the long one
            KmerVocabulary(kmers=((0, 1), (0, 1, 2)), base_alphabet_size=4)

    def test_rejects_kmers_that_crowd_out_the_wildcards(self):
        # Before 000 every symbol starts a kmer -- 0 spells (0,0,0), 1 spells
        # (1,0,0), 2 spells (2,0) -- so a wildcard could not be placed there and
        # any super-string with one there would have no compilation.
        with self.assertRaises(AssertionError):
            KmerVocabulary(kmers=((1, 0, 0), (2, 0), (0, 0, 0)), base_alphabet_size=3)

    def test_accepted_vocabularies_can_compile_anything(self):
        """The two constructor checks are there to make compile total, so whatever
        can be built compiles every super-string over it."""
        for kmers in [
            (TAG, TGA, TAA),
            ((0, 0),),
            ((0, 1), (1, 0)),
            ((0, 1, 2), (1, 2)),
        ]:
            vocab = KmerVocabulary(kmers=kmers, base_alphabet_size=4)
            rng = np.random.default_rng(0)
            for n in (1, 2, 3):
                for s in itertools.product(range(vocab.alphabet_size), repeat=n):
                    out = vocab.compile(list(s), rng)
                    self.assertEqual(
                        vocab.parse(out), vocab.canonicalize(list(s)), (kmers, s)
                    )


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
        against it. Sized so that dropping the fiber counts and drawing evenly
        among the legal symbols lands near 6x the degrees of freedom against 0.9x
        for the real one; at a tenth of these samples it only reaches 1.5x and
        slips under the bound.
        """
        x = self.vocab.unknown_symbol
        s = [x, x, 0, x, x, x]  # five wildcards around one TAG -> 8 base symbols
        want = self.vocab.canonicalize(s)
        length = sum(self.vocab.compiled_length(sym) for sym in s)
        fiber = {}
        for candidate in itertools.product(range(4), repeat=length):
            if self.vocab.parse(list(candidate)) == want:
                fiber[candidate] = len(fiber)

        count = 400 * len(fiber)
        compiled = self.vocab.compile_many(
            [s] * count, [np.random.default_rng(i) for i in range(count)]
        )
        counts = np.zeros(len(fiber))
        for base_string in compiled:
            key = tuple(base_string)
            self.assertIn(key, fiber, "compiled outside the fiber")
            counts[fiber[key]] += 1
        chi2 = ((counts - count / len(fiber)) ** 2 / (count / len(fiber))).sum()
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


class TestOverlappingKmers(unittest.TestCase):
    """Kmers that can overlap each other, or themselves, are where the wildcard
    constraint stops being per-symbol: whether a wildcard may emit something
    depends on the symbols after it, which may belong to a kmer."""

    def test_wildcard_may_not_complete_a_kmer_with_the_next_kmer(self):
        # A wildcard sitting before CG must not emit A: the parse would take the
        # ACG and never see the CG the super-string asked for.
        vocab = KmerVocabulary(kmers=((0, 1, 2), (1, 2)), base_alphabet_size=4)
        x = vocab.unknown_symbol
        for seed in range(200):
            out = vocab.compile([x, 1], np.random.default_rng(seed))
            self.assertNotEqual(out[0], 0, out)
            self.assertEqual(vocab.parse(out), [x, 1])

    def test_wildcard_may_not_extend_a_self_overlapping_kmer(self):
        # AA overlaps itself, so a wildcard before it cannot emit A either.
        vocab = KmerVocabulary(kmers=((0, 0),), base_alphabet_size=4)
        x = vocab.unknown_symbol
        for seed in range(200):
            out = vocab.compile([x, 0], np.random.default_rng(seed))
            self.assertNotEqual(out[0], 0, out)
            self.assertEqual(vocab.parse(out), [x, 0])

    def test_parse_is_leftmost_when_occurrences_overlap(self):
        # AA occurs at 0 and at 1 in AAA; the leftmost wins and the odd symbol
        # is left over as a wildcard.
        vocab = KmerVocabulary(kmers=((0, 0),), base_alphabet_size=4)
        self.assertEqual(vocab.parse([0, 0, 0]), [0, vocab.unknown_symbol])
        self.assertEqual(vocab.parse([0, 0, 0, 0]), [0, 0])

    def test_roundtrip_with_overlapping_kmers(self):
        for kmers in [
            ((0, 0),),
            ((0, 1, 2), (1, 2)),
            ((0, 1), (1, 0)),
            ((0, 1, 0), (1, 0, 1)),
        ]:
            vocab = KmerVocabulary(kmers=kmers, base_alphabet_size=4)
            rng = np.random.default_rng(0)
            for _ in range(300):
                n = int(rng.integers(1, 10))
                s = [int(rng.integers(vocab.alphabet_size)) for _ in range(n)]
                self.assertEqual(
                    vocab.parse(vocab.compile(s, rng)),
                    vocab.canonicalize(s),
                    f"{kmers} {s}",
                )

    def test_uniform_over_the_fiber_with_a_self_overlapping_kmer(self):
        # Self-overlap makes the legal symbols differ sharply between positions,
        # which is exactly what an unweighted draw would get wrong.
        vocab = KmerVocabulary(kmers=((0, 0),), base_alphabet_size=2)
        x = vocab.unknown_symbol
        s = [x, x, 0, x, x]
        want = vocab.canonicalize(s)
        length = sum(vocab.compiled_length(sym) for sym in s)
        fiber = {}
        for candidate in itertools.product(range(2), repeat=length):
            if vocab.parse(list(candidate)) == want:
                fiber[candidate] = len(fiber)
        self.assertGreater(len(fiber), 1, "need a fiber with room to be non-uniform")

        count = 400 * len(fiber)
        compiled = vocab.compile_many(
            [s] * count, [np.random.default_rng(i) for i in range(count)]
        )
        counts = np.zeros(len(fiber))
        for base_string in compiled:
            self.assertIn(tuple(base_string), fiber, "compiled outside the fiber")
            counts[fiber[tuple(base_string)]] += 1
        expected = count / len(fiber)
        chi2 = ((counts - expected) ** 2 / expected).sum()
        self.assertLess(chi2, 4 * max(len(fiber) - 1, 1))


class TestCompileEdges(unittest.TestCase):
    def setUp(self):
        # kmers starting with symbol 0, so that symbol 0 is the one banned
        # somewhere -- with a leading zero weight, the draw has to skip past it.
        self.vocab = KmerVocabulary(
            kmers=((0, 0),), base_alphabet_size=2, num_wildcards=1
        )
        self.X = self.vocab.unknown_symbol

    def test_a_draw_of_exactly_zero_skips_banned_symbols(self):
        class ZeroDraws:
            def random(self, size):
                return np.zeros(size)

        s = [self.X, self.X, 0]
        out = self.vocab.compile_many([s], [ZeroDraws()])[0]
        self.assertEqual(self.vocab.parse(out), self.vocab.canonicalize(s))

    def test_symbols_outside_the_alphabet_are_refused(self):
        rng = np.random.default_rng(0)
        for bad in (-1, self.vocab.alphabet_size):
            with self.assertRaises(AssertionError):
                self.vocab.compile([bad], rng)


if __name__ == "__main__":
    unittest.main()
