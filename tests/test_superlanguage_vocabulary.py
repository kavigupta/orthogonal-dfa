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


class TestPrefixRelatedKmers(unittest.TestCase):
    """A kmer may be a prefix of another. parse then has to prefer the longer one,
    and compile has to keep what follows a kmer from growing it into that longer
    one -- which is not always possible, so compile is partial here."""

    def setUp(self):
        self.vocab = KmerVocabulary(
            kmers=((0, 1), (0, 1, 2)), base_alphabet_size=4, num_wildcards=2
        )
        self.short, self.long = 0, 1
        self.X = self.vocab.unknown_symbol

    def test_parse_takes_the_longer_kmer(self):
        self.assertEqual(self.vocab.parse([0, 1, 2]), [self.long])
        self.assertEqual(self.vocab.parse([0, 1, 3]), [self.short, self.X])

    def test_parse_does_not_depend_on_kmer_order(self):
        flipped = KmerVocabulary(kmers=((0, 1, 2), (0, 1)), base_alphabet_size=4)
        # same string, same reading, even though the indices are swapped
        self.assertEqual(self.vocab.kmers[self.vocab.parse([0, 1, 2])[0]], (0, 1, 2))
        self.assertEqual(flipped.kmers[flipped.parse([0, 1, 2])[0]], (0, 1, 2))

    def test_wildcard_may_not_grow_the_kmer_it_follows(self):
        # (0,1) then a wildcard: emitting 2 would spell (0,1,2) and lose the short
        # kmer, so the wildcard never takes it.
        for seed in range(300):
            out = self.vocab.compile([self.short, self.X], np.random.default_rng(seed))
            self.assertNotEqual(out[2], 2, out)
            self.assertEqual(
                self.vocab.parse(out), self.vocab.canonicalize([self.short, self.X])
            )

    def test_roundtrip_on_encodable_super_strings(self):
        # Anything parse produces came from a real base string, so it is encodable.
        rng = np.random.default_rng(0)
        for _ in range(400):
            x = rng.integers(4, size=int(rng.integers(1, 16))).tolist()
            s = self.vocab.parse(x)
            self.assertEqual(
                self.vocab.parse(self.vocab.compile(s, rng)),
                self.vocab.canonicalize(s),
            )

    def test_unencodable_super_string_is_refused(self):
        # (0,1) then (2,0) can only spell 0,1,2,0, which reads back as (0,1,2).
        vocab = KmerVocabulary(kmers=((0, 1), (0, 1, 2), (2, 0)), base_alphabet_size=4)
        with self.assertRaises(ValueError):
            vocab.compile([0, 2], np.random.default_rng(0))

    def test_still_uniform_over_the_fiber(self):
        s = [self.X, self.short, self.X, self.X]
        want = self.vocab.canonicalize(s)
        length = sum(self.vocab.compiled_length(sym) for sym in s)
        fiber = {}
        for candidate in itertools.product(range(4), repeat=length):
            if self.vocab.parse(list(candidate)) == want:
                fiber[candidate] = len(fiber)
        self.assertGreater(len(fiber), 1)

        count = 60 * len(fiber)
        compiled = self.vocab.compile_many(
            [s] * count, [np.random.default_rng(i) for i in range(count)]
        )
        counts = np.zeros(len(fiber))
        for base_string in compiled:
            self.assertIn(tuple(base_string), fiber, "compiled outside the fiber")
            counts[fiber[tuple(base_string)]] += 1
        expected = count / len(fiber)
        chi2 = ((counts - expected) ** 2 / expected).sum()
        self.assertLess(chi2, 3 * (len(fiber) - 1))

    def test_compile_succeeds_exactly_on_what_parse_can_produce(self):
        """The encodable super-strings are precisely parse's image: every string
        some base string reads as can be compiled, and every one refused is a
        string no base string reads as. Checked exhaustively on a small alphabet.
        """
        vocab = KmerVocabulary(
            kmers=((0, 1), (0, 1, 2), (2, 0)), base_alphabet_size=3, num_wildcards=1
        )
        rng = np.random.default_rng(0)
        # Super-strings of up to three symbols compile to at most nine base
        # symbols, so enumerating that far settles which of them are reachable.
        reachable = {
            tuple(vocab.parse(list(b)))
            for n in range(10)
            for b in itertools.product(range(3), repeat=n)
        }
        refused = 0
        for n in range(1, 4):
            for s in itertools.product(range(vocab.alphabet_size), repeat=n):
                try:
                    out = vocab.compile(list(s), rng)
                except ValueError:
                    refused += 1
                    self.assertNotIn(s, reachable, f"refused reachable {s}")
                    continue
                self.assertIn(s, reachable, f"compiled unreachable {s}")
                self.assertEqual(vocab.parse(out), vocab.canonicalize(list(s)))
        self.assertGreater(refused, 0, "this vocabulary should have gaps")


if __name__ == "__main__":
    unittest.main()
