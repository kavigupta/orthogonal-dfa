"""Tests for the superlanguage stack: sampler, lifted oracle, the ``build_pst``
wiring, and end-to-end learning.  The vocabulary itself is covered separately, in
``test_superlanguage_vocabulary``.

The oracle and learning tests use the real :class:`AllFramesClosedOracle` with the
vocabulary's kmers set to the stop codons ``TAG``/``TGA``/``TAA``.  Because compile
is invertible, ``X`` never spells a stop codon, so the all-frames-closed label is a
deterministic (X-insensitive) function of the super-string.
"""

import unittest

import numpy as np
from parameterized import parameterized

from orthogonal_dfa.l_star.examples.bernoulli_parity import AllFramesClosedOracle
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.structures import Oracle, SymmetricBernoulli
from orthogonal_dfa.superlanguage.learn import learn_superlanguage
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary

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
        # The sampler compiles back to a uniform base stream, so the induced base
        # marginal is flat and no super-string contains a spurious structure.
        out = self.sampler.sample(np.random.default_rng(3), self.vocab.alphabet_size)
        # every sampled super-string round-trips (up to which wildcard was used)
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
        # Because compile is invertible, one compilation already gives the final
        # answer -- the vote is unanimous no matter how many compilations, which
        # is why num_compilations defaults to 1.
        one = LiftedOracle(self.base, self.vocab, num_compilations=1, seed=0)
        many = LiftedOracle(self.base, self.vocab, num_compilations=16, seed=0)
        sampler = SuperSampler(self.vocab, 40)
        rng = np.random.default_rng(5)
        strings = [sampler.sample(rng, self.vocab.alphabet_size) for _ in range(200)]
        np.testing.assert_array_equal(
            one.membership_queries(strings), many.membership_queries(strings)
        )

    def test_majority_vote_over_compilations(self):
        """A base oracle that *does* read the wildcard fill is answered by the
        majority of ``num_compilations`` draws, which is what more than one
        compilation buys.  Here a lone wildcard compiles to a uniform base symbol,
        so ~3/4 of the draws are non-A and the vote lands on accept; the mirrored
        oracle votes reject.
        """
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

    def test_wildcard_suffixes_preserve_the_label(self):
        """Appending wildcards cannot create a stop codon, so a wildcard-only
        suffix leaves the all-frames-closed label alone -- these suffixes have the
        same membership column as the empty suffix, which is what the learner's
        suffix-family clustering locks onto.  Several wildcards exist precisely so
        there are many *distinct* such suffixes to fill a family with.
        """
        wild = self.vocab.wildcard_symbols
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
        # With one wildcard there is exactly one wildcard-only suffix per length,
        # too few to fill a suffix family; with two there are plenty.
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
            LiftedOracle(_PredicateOracle(lambda s: True, alphabet_size=2), self.vocab)

    def test_needs_at_least_one_compilation(self):
        with self.assertRaises(AssertionError):
            LiftedOracle(self.base, self.vocab, num_compilations=0)


class TestBuildPstWiring(unittest.TestCase):
    """``build_pst`` must thread a custom sampler through to the tracker."""

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


class TestLearnSuperlanguage(unittest.TestCase):
    """End-to-end: learn a DFA over the stop-codon superlanguage against the real
    all-frames-closed oracle -- the superlanguage analogue of ``test_no_orf``.

    The induced DFA is smaller than the base-alphabet one (a stop codon is a
    single super-symbol rather than a three-symbol path).
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
