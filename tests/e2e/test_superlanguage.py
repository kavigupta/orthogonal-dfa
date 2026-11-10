import unittest

import numpy as np
from parameterized import parameterized

from orthogonal_dfa.l_star.examples.bernoulli_parity import AllFramesClosedOracle
from orthogonal_dfa.l_star.learn import DEFAULT_MAX_COVERAGE_ERROR
from orthogonal_dfa.l_star.tracker import RecordingTracker
from orthogonal_dfa.superlanguage.learn import learn_superlanguage
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary
from tests.lstar_common import (
    assert_rounds_accept_preserving,
    assert_searches_end_inside_the_proofs_budget,
)

# ACGT base alphabet: A=0, C=1, G=2, T=3.
TAG, TGA, TAA = (3, 0, 2), (3, 2, 0), (3, 0, 0)


class TestLearnSuperlanguage(unittest.TestCase):
    """The superlanguage counterpart of test_no_orf. A stop codon is one
    super-symbol rather than a three-symbol path, so the DFA is smaller than the
    base-alphabet one.
    """

    @parameterized.expand([(signal,) for signal in (0.3, 0.2)])
    def test_learns_all_frames_closed(self, signal):
        vocab = KmerVocabulary(kmers=(TAG, TGA, TAA), base_alphabet_size=4)
        base = AllFramesClosedOracle()
        tracker = RecordingTracker()
        dfa = learn_superlanguage(
            base, vocab, min_signal_strength=signal, seed=0, tracker=tracker
        )
        self.assertIsNotNone(dfa)

        oracle = LiftedOracle(base, vocab, seed=0)
        # Every family the clustering produced, not just the DFA it ended on.
        assert_rounds_accept_preserving(
            tracker.classifiers, oracle.target_dfa(), DEFAULT_MAX_COVERAGE_ERROR
        )
        assert_searches_end_inside_the_proofs_budget(tracker.endings)
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
