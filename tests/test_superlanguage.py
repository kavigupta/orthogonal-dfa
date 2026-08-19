"""Tests for the superlanguage stack: vocabulary, sampler, lifted oracle, and
the ``build_pst`` wiring.

The lifted-oracle tests use the real :class:`AllFramesClosedOracle` with the
vocabulary's kmers set to the stop codons ``TAG``/``TGA``/``TAA`` -- an oracle
that genuinely reads the features the kmers encode -- and construct super-strings
whose all-frames-closed label is independent of how ``X`` compiles, so the
assertions are deterministic.
"""

import unittest

import numpy as np

from orthogonal_dfa.l_star.examples.bernoulli_parity import AllFramesClosedOracle
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.structures import Oracle, SymmetricBernoulli
from orthogonal_dfa.superlanguage import (
    KmerVocabulary,
    LiftedOracle,
    SuperSampler,
)

# ACGT base alphabet: A=0, C=1, G=2, T=3.
TAG, TGA, TAA = (3, 0, 2), (3, 2, 0), (3, 0, 0)


class _PredicateOracle(Oracle):
    """Base oracle accepting a base string iff ``predicate(string)``."""

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


class TestKmerVocabulary(unittest.TestCase):
    def test_alphabet_shape(self):
        v = KmerVocabulary(kmers=(TAG, TGA), base_alphabet_size=4)
        self.assertEqual(v.num_kmers, 2)
        self.assertEqual(v.alphabet_size, 3)  # two kmers + X
        self.assertEqual(v.unknown_symbol, 2)
        self.assertTrue(v.is_unknown(2))
        self.assertFalse(v.is_unknown(0))

    def test_probabilities_uniform_null(self):
        v = KmerVocabulary(kmers=(TAG, TGA, TAA), base_alphabet_size=4)
        probs = v.probabilities()
        # each length-3 kmer under the uniform null: 4**-3 = 1/64.
        np.testing.assert_allclose(probs[:3], 1 / 64)
        self.assertAlmostEqual(probs[v.unknown_symbol], 1 - 3 / 64)
        self.assertAlmostEqual(probs.sum(), 1.0)

    def test_probabilities_shortest_wins(self):
        # (0,1,2) is preempted by the strictly shorter prefix (0,1): its mass is
        # claimed by the shorter kmer, so it gets 0 and X collects the rest.
        v = KmerVocabulary(kmers=((0, 1), (0, 1, 2)), base_alphabet_size=4)
        probs = v.probabilities()
        self.assertAlmostEqual(probs[0], 1 / 16)  # (0,1): 4**-2
        self.assertEqual(probs[1], 0.0)  # (0,1,2): preempted
        self.assertAlmostEqual(probs[v.unknown_symbol], 1 - 1 / 16)
        self.assertAlmostEqual(probs.sum(), 1.0)

    def test_probabilities_variable_length(self):
        # A short and a long kmer that are not prefix-related: both keep mass.
        v = KmerVocabulary(kmers=((0, 1), (2, 3, 0)), base_alphabet_size=4)
        probs = v.probabilities()
        self.assertAlmostEqual(probs[0], 1 / 16)  # length 2
        self.assertAlmostEqual(probs[1], 1 / 64)  # length 3
        self.assertAlmostEqual(probs[v.unknown_symbol], 1 - 1 / 16 - 1 / 64)

    def test_compile_symbol(self):
        v = KmerVocabulary(kmers=(TAG, TGA), base_alphabet_size=4)
        rng = np.random.default_rng(0)
        self.assertEqual(v.compile_symbol(0, rng), list(TAG))
        self.assertEqual(v.compile_symbol(1, rng), list(TGA))
        # X compiles to exactly one base symbol, in range.
        for seed in range(20):
            out = v.compile_symbol(v.unknown_symbol, np.random.default_rng(seed))
            self.assertEqual(len(out), 1)
            self.assertIn(out[0], range(4))

    def test_compiled_length(self):
        v = KmerVocabulary(kmers=((0, 1), (2, 3, 0)), base_alphabet_size=4)
        self.assertEqual(v.compiled_length(0), 2)
        self.assertEqual(v.compiled_length(1), 3)
        self.assertEqual(v.compiled_length(v.unknown_symbol), 1)

    def test_compile_concatenates(self):
        v = KmerVocabulary(kmers=(TAG, TGA), base_alphabet_size=4)
        rng = np.random.default_rng(0)
        # kmer symbols only -> fully deterministic concatenation.
        self.assertEqual(v.compile([0, 1, 0], rng), list(TAG) + list(TGA) + list(TAG))

    def test_from_corpus_picks_frequent_kmers(self):
        # (0,1,2) appears far more than anything else.
        corpus = [[0, 1, 2, 0, 1, 2, 0, 1, 2]] * 10 + [[3, 3, 3, 3]]
        v = KmerVocabulary.from_corpus(corpus, 4, lengths=(3,), top_n=1)
        self.assertEqual(v.kmers, ((0, 1, 2),))

    def test_from_corpus_is_prefix_free(self):
        corpus = [[0, 1, 2, 3] * 5]
        v = KmerVocabulary.from_corpus(corpus, 4, lengths=(2, 3, 4), top_n=8)
        for i, a in enumerate(v.kmers):
            for b in v.kmers[i + 1 :]:
                m = min(len(a), len(b))
                self.assertNotEqual(
                    tuple(a[:m]), tuple(b[:m]), f"{a} and {b} are prefix-related"
                )
        # prefix-free => every kept kmer has non-zero probability.
        self.assertTrue((v.probabilities()[: v.num_kmers] > 0).all())

    def test_from_corpus_no_prune_keeps_preempted(self):
        corpus = [[0, 1, 2, 0, 1, 2, 0, 1]] * 5
        v = KmerVocabulary.from_corpus(
            corpus, 4, lengths=(2, 3), top_n=10, prune_non_minimal=False
        )
        # (0,1) and (0,1,2) both appear frequently; without pruning both are kept.
        self.assertIn((0, 1), v.kmers)
        self.assertIn((0, 1, 2), v.kmers)

    def test_rejects_bad_kmers(self):
        with self.assertRaises(AssertionError):
            KmerVocabulary(kmers=((0, 9),), base_alphabet_size=4)  # 9 out of range
        with self.assertRaises(AssertionError):
            KmerVocabulary(kmers=((),), base_alphabet_size=4)  # empty kmer
        with self.assertRaises(AssertionError):
            KmerVocabulary(kmers=((0,), (0,)), base_alphabet_size=4)  # duplicate


class TestSuperSampler(unittest.TestCase):
    def setUp(self):
        self.vocab = KmerVocabulary(kmers=(TAG, TGA, TAA), base_alphabet_size=4)
        self.sampler = SuperSampler(self.vocab, length=25)

    def test_length_and_range(self):
        out = self.sampler.sample(np.random.default_rng(0), self.vocab.alphabet_size)
        self.assertEqual(len(out), 25)
        self.assertTrue(all(0 <= s < self.vocab.alphabet_size for s in out))

    def test_length_attribute_is_symbol_count(self):
        # The learner reads sampler.length as the number of super-symbols.
        self.assertEqual(self.sampler.length, 25)

    def test_distribution_matches_probabilities(self):
        sampler = SuperSampler(self.vocab, length=200_000)
        draw = sampler.sample(np.random.default_rng(1), self.vocab.alphabet_size)
        counts = np.bincount(draw, minlength=self.vocab.alphabet_size)
        empirical = counts / counts.sum()
        np.testing.assert_allclose(
            empirical, self.vocab.probabilities(), atol=5e-3
        )

    def test_alphabet_size_mismatch_asserts(self):
        with self.assertRaises(AssertionError):
            self.sampler.sample(np.random.default_rng(0), self.vocab.alphabet_size + 1)


class TestLiftedOracle(unittest.TestCase):
    def setUp(self):
        self.vocab = KmerVocabulary(kmers=(TAG, TGA, TAA), base_alphabet_size=4)
        self.X = self.vocab.unknown_symbol
        # Identity noise: the compilation is the only stochasticity.
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

    def test_stop_codons_accept_when_all_frames_closed(self):
        # [TAG, X, TAG, X, TAG] places stops at base positions 0, 4, 8 -- one in
        # each reading frame -- so all frames are closed for every X realization.
        self.assertTrue(self.oracle.membership_query([0, self.X, 0, self.X, 0]))

    def test_stop_codons_reject_when_a_frame_is_open(self):
        # Only frames 0 and 1 get a stop; frame 2 stays open regardless of X.
        self.assertFalse(self.oracle.membership_query([0, self.X, 0]))

    def test_all_kmer_string_leaves_two_frames_open(self):
        # Back-to-back length-3 kmers keep every stop in frame 0.
        self.assertFalse(self.oracle.membership_query([0, 1, 2]))

    def test_majority_vote_over_compilations(self):
        # Base accepts iff the first base symbol is not A(0): a lone X compiles to
        # a uniform base symbol, so ~3/4 of compilations accept and the vote is
        # accept; the mirrored oracle (accept iff first symbol *is* 0) votes reject.
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

    def test_base_alphabet_mismatch_asserts(self):
        with self.assertRaises(AssertionError):
            LiftedOracle(_PredicateOracle(lambda s: True, alphabet_size=2), self.vocab)

    def test_needs_at_least_one_compilation(self):
        with self.assertRaises(AssertionError):
            LiftedOracle(self.base, self.vocab, num_compilations=0)


class TestBuildPstWiring(unittest.TestCase):
    """``build_pst`` must thread a custom sampler through to the tracker, and the
    whole superlanguage stack must run through calibration without error."""

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
        self.assertEqual(pst.alphabet_size, vocab.alphabet_size)

    def test_default_sampler_is_uniform(self):
        base = _PredicateOracle(lambda s: len(s) > 0 and s[0] == 0)

        def oracle_creator(noise_model, seed):
            vocab = KmerVocabulary(kmers=((0, 1),), base_alphabet_size=4)
            return LiftedOracle(
                base, vocab, num_compilations=2, seed=seed, noise_model=noise_model
            )

        # No sampler passed -> defaults to a UniformSampler over sample_length.
        pst = build_pst(
            oracle_creator, min_signal_strength=0.4, seed=0, sample_length=15
        )
        self.assertIsInstance(pst.sampler, UniformSampler)
        self.assertEqual(pst.sampler.length, 15)


if __name__ == "__main__":
    unittest.main()
