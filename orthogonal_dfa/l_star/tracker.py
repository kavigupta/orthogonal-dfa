"""Callbacks over a synthesis run, for a caller that wants the intermediates.

The loop never reads back what it hands over, and each round builds its own
family, tree and DFA, so what a tracker keeps is not mutated by the rounds that
follow -- it can be pickled and analysed after the run.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RoundSummary:
    """A round's table and family sizes and its search's cost, as of the end of
    that search."""

    prefixes: int
    representative_prefixes: int
    suffixes: int
    family_size: int
    probes: int
    ending: str
    hypothesis_states: int
    hypothesis_edges: int
    sampling_seconds: float
    probing_seconds: float


@dataclass(frozen=True)
class RestartSummary:
    """What a recorder keeps of a restart; ``search_score`` is the best score the
    search stopped on, ``None`` where it stopped on none."""

    restart: int
    ending: object
    dfa: object
    pick_score: Optional[float]
    search_score: Optional[float]
    probes: int


class SynthesisTracker:
    """No-op; override the rounds' artefacts you care about.

    One instance of this class is the default argument of `learn_dfa` and
    `learn_superlanguage`, so every untracked run shares it: it has to stay
    stateless."""

    def on_family_resolved(self, suffixes, boundary, fnr, round_index):
        """The family this round clustered, the accept rate it re-estimated as
        the boundary between accept and reject, and the family's FNR."""

    def on_round_classified(self, classifier, round_index):
        """How that family cut the round's representative prefixes."""

    # pylint: disable=too-many-positional-arguments
    def on_restart_ended(self, restart, search, ending, dfa, pick_score, round_index):
        """One of the round's counterexample searches, with the DFA it found and
        that DFA's held-out phi on the pick sample (``None`` where there is no
        sample).  ``search`` holds closures, so keep what you need of it."""

    def on_search_ended(self, ending, round_index):
        """How the round's counterexample search ended."""

    def on_initial_dfa_found(self, dfa, tree, round_index):
        """The round's hypothesis, as discovery labelled it."""

    def on_consistency_estimated(self, consistency, round_index):
        """How far the round's DFA and tree agreed on fresh samples."""

    def on_certificate_decided(self, certified, round_index):
        """Whether the certificate passed the round's denoised DFA; asked only
        of rounds that reached the consistency target."""

    def on_round_summarized(self, summary, round_index):
        """The round's `RoundSummary`."""

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
        self.fnrs = []
        #: round -> its restarts, in order.
        self.restarts = []
        self.classifiers = []
        self.endings = []
        self.hypotheses = []
        self.consistency = []
        #: round -> whether it was certified, for the rounds that were asked.
        self.certified = {}
        self.summaries = []
        self.corrected = None

    def on_family_resolved(self, suffixes, boundary, fnr, round_index):
        assert len(self.families) == round_index
        self.families.append((suffixes, boundary))
        self.fnrs.append(fnr)

    def on_round_classified(self, classifier, round_index):
        assert len(self.classifiers) == round_index
        self.classifiers.append(classifier)

    # pylint: disable=too-many-positional-arguments
    def on_restart_ended(self, restart, search, ending, dfa, pick_score, round_index):
        if restart == 0:
            assert len(self.restarts) == round_index
            self.restarts.append([])
        assert len(self.restarts[round_index]) == restart
        score_patience = search.score_patience
        self.restarts[round_index].append(
            RestartSummary(
                restart,
                ending,
                dfa,
                pick_score,
                None if score_patience is None else score_patience.best_score,
                search.probes,
            )
        )

    def on_search_ended(self, ending, round_index):
        assert len(self.endings) == round_index
        self.endings.append(ending)

    def on_initial_dfa_found(self, dfa, tree, round_index):
        assert len(self.hypotheses) == round_index
        self.hypotheses.append((dfa, tree))

    def on_consistency_estimated(self, consistency, round_index):
        assert len(self.consistency) == round_index
        self.consistency.append(consistency)

    def on_certificate_decided(self, certified, round_index):
        self.certified[round_index] = certified

    def on_round_summarized(self, summary, round_index):
        assert len(self.summaries) == round_index
        self.summaries.append(summary)

    def on_corrected_dfa_found(self, dfa, round_index):
        self.corrected = (dfa, round_index)
