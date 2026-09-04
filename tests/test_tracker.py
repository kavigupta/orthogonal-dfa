import pickle
import unittest

from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliRegex
from orthogonal_dfa.l_star.learn import learn_dfa
from orthogonal_dfa.l_star.structures import NoisyOracle
from orthogonal_dfa.l_star.tracker import RecordingTracker


class _OrderedRecorder(RecordingTracker):
    """A recorder that also keeps ``(callback, round_index)`` in fired order."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def on_family_resolved(self, vs, boundary, round_index):
        super().on_family_resolved(vs, boundary, round_index)
        self.calls.append(("family", round_index))

    def on_round_classified(self, classifier, round_index):
        super().on_round_classified(classifier, round_index)
        self.calls.append(("classified", round_index))

    def on_initial_dfa_found(self, dfa, tree, round_index):
        super().on_initial_dfa_found(dfa, tree, round_index)
        self.calls.append(("initial", round_index))

    def on_consistency_estimated(self, consistency, round_index):
        super().on_consistency_estimated(consistency, round_index)
        self.calls.append(("consistency", round_index))

    def on_corrected_dfa_found(self, dfa, round_index):
        super().on_corrected_dfa_found(dfa, round_index)
        self.calls.append(("corrected", round_index))


def _learn(tracker):
    oracle_creator = lambda noise_model, seed: NoisyOracle(
        BernoulliRegex(regex=r".*1010101.*"), noise_model, seed
    )
    return learn_dfa(oracle_creator, min_signal_strength=0.3, seed=0, tracker=tracker)


class TestSynthesisTracker(unittest.TestCase):
    """One learning run, shared: the assertions are all about what it reported."""

    @classmethod
    def setUpClass(cls):
        cls.tracker = _OrderedRecorder()
        cls.dfa = _learn(cls.tracker)

    def test_learns_without_a_tracker(self):
        self.assertIsNotNone(_learn(None))

    def test_every_round_reports_once(self):
        rounds = 1 + max(r for _, r in self.tracker.calls)
        for name in ("family", "classified", "initial", "consistency"):
            fired = [r for called, r in self.tracker.calls if called == name]
            self.assertEqual(fired, list(range(rounds)), name)

    def test_correction_comes_after_every_round(self):
        # The corrected DFA is the run's, not a round's.
        self.assertEqual(
            [c for c in self.tracker.calls if c[0] == "corrected"],
            [("corrected", self.tracker.corrected[1])],
        )
        self.assertEqual(self.tracker.calls[-1][0], "corrected")

    def test_a_round_reports_in_the_order_it_works(self):
        first = [
            name
            for name, r in self.tracker.calls
            if r == 0 and name != "corrected"  # which may also name round 0
        ]
        self.assertEqual(first, ["family", "classified", "initial", "consistency"])

    def test_corrected_names_the_most_consistent_round(self):
        consistency = self.tracker.consistency
        best = max(range(len(consistency)), key=consistency.__getitem__)
        self.assertEqual(self.tracker.corrected[1], best)

    def test_streams_agree_on_the_round_count(self):
        rounds = len(self.tracker.consistency)
        self.assertEqual(len(self.tracker.families), rounds)
        self.assertEqual(len(self.tracker.classifiers), rounds)
        self.assertEqual(len(self.tracker.hypotheses), rounds)

    def test_recording_tracker_pickles(self):
        reloaded = pickle.loads(pickle.dumps(self.tracker))
        self.assertEqual(len(reloaded.classifiers), len(self.tracker.classifiers))
        self.assertEqual(reloaded.consistency, self.tracker.consistency)
        self.assertEqual(len(reloaded.hypotheses), len(self.tracker.hypotheses))
