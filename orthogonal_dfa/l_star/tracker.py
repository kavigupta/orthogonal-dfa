"""Callbacks over a synthesis run, for a caller that wants the intermediates.

Each round hands over what it made as it makes them.  The loop never reads them
back, so a tracker is free to keep them, pickle them, or ignore them; and each
round builds its own family, tree and DFA, so what a tracker keeps is not
mutated by the rounds that follow.
"""


class SynthesisTracker:
    """No-op; override the rounds' artefacts you care about."""

    def on_family_resolved(self, vs, boundary, round_index):
        """The suffix rows this round clustered, and the boundary they read at."""

    def on_round_classified(self, classifier, round_index):
        """How that family cut the round's representative prefixes."""

    def on_initial_dfa_found(self, dfa, tree, round_index):
        """The round's hypothesis, as discovery labelled it."""

    def on_consistency_estimated(self, consistency, round_index):
        """How far the round's DFA and tree agreed on fresh samples."""

    def on_corrected_dfa_found(self, dfa, round_index):
        """The hypothesis the run settled on, denoised.  Fires once, naming the
        round it came from -- which is the most consistent round, not the last."""


class RecordingTracker(SynthesisTracker):
    """Keeps every artefact, one entry per round in round order."""

    def __init__(self):
        self.families = []
        self.classifiers = []
        self.hypotheses = []
        self.consistency = []
        self.corrected = None

    def on_family_resolved(self, vs, boundary, round_index):
        self.families.append((vs, boundary))

    def on_round_classified(self, classifier, round_index):
        self.classifiers.append(classifier)

    def on_initial_dfa_found(self, dfa, tree, round_index):
        self.hypotheses.append((dfa, tree))

    def on_consistency_estimated(self, consistency, round_index):
        self.consistency.append(consistency)

    def on_corrected_dfa_found(self, dfa, round_index):
        self.corrected = (dfa, round_index)
