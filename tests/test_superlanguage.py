"""The kmers are the stop codons, so all_frames_closed reads exactly what they
carry and a wildcard can never forge one. Its label is then a function of the
super-string alone, which is what lets these assert equality rather than a rate.
"""

import unittest

import numpy as np
from parameterized import parameterized

from orthogonal_dfa.l_star.examples.bernoulli_parity import AllFramesClosedOracle
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.structures import NoiseModel, Oracle, SymmetricBernoulli
from orthogonal_dfa.superlanguage.learn import learn_superlanguage
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

# ACGT base alphabet: A=0, C=1, G=2, T=3.
TAG, TGA, TAA = (3, 0, 2), (3, 2, 0), (3, 0, 0)


class _PredicateOracle(Oracle):
    """Accepts a base string iff predicate(string)."""

    def __init__(self, predicate, alphabet_size=4):
        self._predicate = predicate
        self._alphabet_size = alphabet_size

    @property
    def alphabet_size(self):
        return self._alphabet_size

    def membership_queries(self, strings):
        return np.array([bool(self._predicate(s)) for s in strings], dtype=bool)

    def membership_query(self, string):
        return bool(self._predicate(string))


class _Halt(Exception):
    """Ends a learn once it has queried enough to show what it was configured with."""


class _RecordingOracle(Oracle):
    def __init__(self):
        self.lengths = []

    @property
    def alphabet_size(self):
        return 4

    def membership_queries(self, strings):
        if self.lengths:
            raise _Halt
        self.lengths = [len(s) for s in strings]
        return np.zeros(len(strings), dtype=bool)

    def membership_query(self, string):
        return bool(self.membership_queries([string])[0])


class _RecordingNoise(NoiseModel):
    def __init__(self):
        self.calls = 0

    def apply_noise(self, correct_value, string, seed):
        self.calls += 1
        return correct_value


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


class TestLiftedOracle(unittest.TestCase):
    def setUp(self):
        self.vocab = KmerVocabulary(kmers=(TAG, TGA, TAA), base_alphabet_size=4)
        self.X = self.vocab.unknown_symbol
        self.base = AllFramesClosedOracle(noise_model=SymmetricBernoulli(1.0), seed=0)
        self.oracle = LiftedOracle(self.base, self.vocab, num_compilations=6, seed=0)

    def test_alphabet_size_is_super(self):
        self.assertEqual(self.oracle.alphabet_size, self.vocab.alphabet_size)

    def test_empty_batch(self):
        out = self.oracle.membership_queries([])
        self.assertEqual(out.shape, (0,))
        self.assertEqual(out.dtype, bool)

    def test_determinism(self):
        query = [[0, self.X, 1, self.X, 2], [0, 0, 0], [self.X, self.X]]
        a = self.oracle.membership_queries(query)
        b = self.oracle.membership_queries(query)
        np.testing.assert_array_equal(a, b)

    def test_x_insensitive_label(self):
        # Invertibility makes the vote unanimous, so one compilation settles it.
        one = LiftedOracle(self.base, self.vocab, num_compilations=1, seed=0)
        many = LiftedOracle(self.base, self.vocab, num_compilations=16, seed=0)
        sampler = SuperSampler(self.vocab, 40)
        rng = np.random.default_rng(5)
        strings = [sampler.sample(rng, self.vocab.alphabet_size) for _ in range(200)]
        np.testing.assert_array_equal(
            one.membership_queries(strings), many.membership_queries(strings)
        )

    def test_majority_vote_over_compilations(self):
        # A lone wildcard is uniform over the base alphabet, so three draws in four
        # are non-A and the vote goes that way; the mirrored oracle goes the other.
        vocab = KmerVocabulary(kmers=((0, 1),), base_alphabet_size=4)
        x = vocab.unknown_symbol
        mostly = LiftedOracle(
            _PredicateOracle(lambda s: s[0] != 0), vocab, num_compilations=64, seed=0
        )
        rarely = LiftedOracle(
            _PredicateOracle(lambda s: s[0] == 0), vocab, num_compilations=64, seed=0
        )
        self.assertTrue(mostly.membership_query([x]))
        self.assertFalse(rarely.membership_query([x]))

    def test_the_noise_model_reaches_the_answer(self):
        # p_correct=0 inverts every label, so a noised oracle must contradict a
        # clean one everywhere.
        sampler = SuperSampler(self.vocab, 20)
        rng = np.random.default_rng(7)
        strings = [sampler.sample(rng, self.vocab.alphabet_size) for _ in range(50)]
        clean = LiftedOracle(self.base, self.vocab, num_compilations=1, seed=0)
        flipped = LiftedOracle(
            self.base,
            self.vocab,
            num_compilations=1,
            seed=0,
            noise_model=SymmetricBernoulli(p_correct=0.0),
        )
        np.testing.assert_array_equal(
            flipped.membership_queries(strings), ~clean.membership_queries(strings)
        )

    def test_the_seed_picks_out_a_different_compilation(self):
        # Which base string a wildcard compiles to is the seed's business, so an
        # oracle that reads the fill answers differently under a different one.
        vocab = KmerVocabulary(kmers=((0, 1),), base_alphabet_size=4)
        x = vocab.unknown_symbol
        base = _PredicateOracle(lambda s: s[0] != 0)
        strings = [[x] * n for n in range(1, 40)]
        first = LiftedOracle(base, vocab, num_compilations=1, seed=0)
        second = LiftedOracle(base, vocab, num_compilations=1, seed=1)
        self.assertTrue(
            (
                first.membership_queries(strings) != second.membership_queries(strings)
            ).any()
        )

    def test_wildcard_suffixes_preserve_the_label(self):
        """Wildcard-only suffixes share the empty suffix's column, which is the one
        the suffix-family clustering seeds on.
        """
        wild = range(self.vocab.num_kmers, self.vocab.alphabet_size)
        sampler = SuperSampler(self.vocab, 30)
        rng = np.random.default_rng(11)
        for _ in range(50):
            prefix = sampler.sample(rng, self.vocab.alphabet_size)
            suffix = [int(rng.choice(wild)) for _ in range(20)]
            self.assertEqual(
                self.oracle.membership_query(prefix),
                self.oracle.membership_query(list(prefix) + suffix),
            )

    def test_many_distinct_wildcard_only_suffixes(self):
        # One wildcard would give a single such suffix per length, far under the
        # family size; two give plenty.
        sampler = SuperSampler(self.vocab, 40)
        rng = np.random.default_rng(12)
        seen = set()
        for _ in range(400):
            s = sampler.sample(rng, self.vocab.alphabet_size)
            if all(self.vocab.is_unknown(sym) for sym in s):
                seen.add(tuple(s))
        self.assertGreater(len(seen), 10)

    def test_stop_codons_accept_when_all_frames_closed(self):
        # [TAG, X, TAG, X, TAG] places stops at base positions 0, 4, 8 -- one in
        # each reading frame -- so all frames are closed for every X realization.
        self.assertTrue(self.oracle.membership_query([0, self.X, 0, self.X, 0]))

    def test_stop_codons_reject_when_a_frame_is_open(self):
        # Only frames 0 and 1 get a stop; frame 2 stays open, and X cannot forge one.
        self.assertFalse(self.oracle.membership_query([0, self.X, 0]))

    def test_all_kmer_string_leaves_two_frames_open(self):
        self.assertFalse(self.oracle.membership_query([0, 1, 2]))

    def test_base_alphabet_mismatch_asserts(self):
        with self.assertRaises(AssertionError):
            LiftedOracle(
                _PredicateOracle(lambda s: True, alphabet_size=2),
                self.vocab,
                num_compilations=1,
            )

    def test_needs_at_least_one_compilation(self):
        with self.assertRaises(AssertionError):
            LiftedOracle(self.base, self.vocab, num_compilations=0)


class TestBuildPstWiring(unittest.TestCase):
    """build_pst must thread a custom sampler through to the tracker."""

    def test_super_sampler_flows_through_build_pst(self):
        vocab = KmerVocabulary(kmers=((0, 1),), base_alphabet_size=4)
        sampler = SuperSampler(vocab, length=20)
        base = _PredicateOracle(lambda s: len(s) > 0 and s[0] == 0)

        def oracle_creator(noise_model, seed):
            return LiftedOracle(
                base, vocab, num_compilations=2, seed=seed, noise_model=noise_model
            )

        pst = build_pst(
            oracle_creator,
            min_signal_strength=0.4,
            seed=0,
            sampler=sampler,
        )
        self.assertIs(pst.sampler, sampler)
        # The tracker works over the super alphabet, not the base one.
        self.assertEqual(pst.alphabet_size, vocab.alphabet_size)
        self.assertNotEqual(pst.alphabet_size, vocab.base_alphabet_size)


class TestLearnForwarding(unittest.TestCase):
    """The knobs learn_superlanguage takes are read off the first queries it makes,
    so that a dropped argument shows up here rather than as a slower learn."""

    def _first_batch(self, **kwargs):
        vocab = KmerVocabulary(kmers=(TAG, TGA, TAA), base_alphabet_size=4)
        base, noise = _RecordingOracle(), _RecordingNoise()
        with self.assertRaises(_Halt):
            learn_superlanguage(
                base,
                vocab,
                min_signal_strength=0.3,
                seed=0,
                noise_model=noise,
                **kwargs,
            )
        return base.lengths, noise

    def test_num_symbols_reaches_the_sampler(self):
        # The batch also holds the short prefix-closed core, so the sampled length
        # shows in the longest string: seven super-symbols, each a base symbol or a
        # whole codon. Left at the default it would run to forty.
        lengths, _ = self._first_batch(num_symbols=7)
        self.assertTrue(7 <= max(lengths) <= 21, max(lengths))

    def test_num_compilations_reaches_the_oracle(self):
        lengths, _ = self._first_batch(num_symbols=7, num_compilations=3)
        self.assertEqual(len(lengths) % 3, 0)
        self.assertGreater(len(lengths), 3)

    def test_the_noise_model_reaches_the_tracker(self):
        _, noise = self._first_batch(num_symbols=7)
        self.assertGreater(noise.calls, 0)


class TestLearnSuperlanguage(unittest.TestCase):
    """The superlanguage counterpart of test_no_orf. A stop codon is one
    super-symbol rather than a three-symbol path, so the DFA is smaller than the
    base-alphabet one.
    """

    @parameterized.expand([(signal,) for signal in (0.3, 0.2)])
    def test_learns_all_frames_closed(self, signal):
        vocab = KmerVocabulary(kmers=(TAG, TGA, TAA), base_alphabet_size=4)
        base = AllFramesClosedOracle(noise_model=SymmetricBernoulli(1.0), seed=0)
        dfa, _ = learn_superlanguage(base, vocab, min_signal_strength=signal, seed=0)
        self.assertIsNotNone(dfa)

        oracle = LiftedOracle(base, vocab, num_compilations=1, seed=0)
        sampler = SuperSampler(vocab, 40)
        rng = np.random.default_rng(0x1234)
        strings = [sampler.sample(rng, vocab.alphabet_size) for _ in range(3000)]
        expected = oracle.membership_queries(strings)
        actual = np.array([dfa.accepts_input(s) for s in strings], dtype=bool)
        accuracy = (expected == actual).mean()
        self.assertGreaterEqual(
            accuracy, 0.97, f"signal={signal} accuracy {accuracy:.3f}"
        )


if __name__ == "__main__":
    unittest.main()
