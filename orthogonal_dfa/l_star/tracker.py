"""Callbacks over a synthesis run, for a caller that wants the intermediates.

The loop never reads back what it hands over, and each round builds its own
family, tree and DFA, so what a tracker keeps is not mutated by the rounds that
follow -- it can be pickled and analysed after the run.
"""


class SynthesisTracker:
    """No-op; override the rounds' artefacts you care about."""

    def on_family_resolved(self, suffixes, boundary, round_index):
        """The family this round clustered, and the accept rate it re-estimated
        as the boundary between accept and reject."""

    def on_round_classified(self, classifier, round_index):
        """How that family cut the round's representative prefixes."""

    def on_initial_dfa_found(self, dfa, tree, round_index):
        """The round's hypothesis, as discovery labelled it."""

    def on_consistency_estimated(self, consistency, round_index):
        """How far the round's DFA and tree agreed on fresh samples."""

    def on_corrected_dfa_found(self, dfa, round_index):
        """The hypothesis the run settled on, denoised.  Fired once, by
        ``do_counterexample_driven_synthesis`` -- a caller driving
        ``counterexample_driven_synthesis`` itself never sees it."""


class RecordingTracker(SynthesisTracker):
    """Keeps every artefact: one entry per round in round order, plus the single
    ``corrected`` pair.

    ``corrected[1]`` indexes back into the per-round streams, so each append
    asserts its position. A round reporting one artefact but not another would
    otherwise shift every later index silently.
    """

    def __init__(self):
        self.families = []
        self.classifiers = []
        self.hypotheses = []
        self.consistency = []
        self.corrected = None

    def on_family_resolved(self, suffixes, boundary, round_index):
        assert len(self.families) == round_index
        self.families.append((suffixes, boundary))

    def on_round_classified(self, classifier, round_index):
        assert len(self.classifiers) == round_index
        self.classifiers.append(classifier)

    def on_initial_dfa_found(self, dfa, tree, round_index):
        assert len(self.hypotheses) == round_index
        self.hypotheses.append((dfa, tree))

    def on_consistency_estimated(self, consistency, round_index):
        assert len(self.consistency) == round_index
        self.consistency.append(consistency)

    def on_corrected_dfa_found(self, dfa, round_index):
        self.corrected = (dfa, round_index)
