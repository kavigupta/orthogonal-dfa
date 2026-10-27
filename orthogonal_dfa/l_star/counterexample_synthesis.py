"""
Counterexample-driven synthesis: the E-L* learner loop.

Each round searches a suffix family and runs the counterexample search against it; what
the round harvests joins the next round's pool.
"""

import time
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from automata.fa.dfa import DFA

from .certificate import certifies, look_level
from .cluster import sample_suffix_family
from .counterexample_config import counterexample_search_config
from .counterexample_search import CounterexampleSearch
from .dfa_utils import held_out_phi, labelled_sample
from .lstar import denoise_accept_labels, estimate_agreement_rate
from .mask_table import UNIFORM
from .midfix_tree import MidfixTree, oracle_decider
from .preconditions import start_length
from .prefix_populations import PoolState
from .prefix_sources import HarvestSource, Replay, UniformSource, harvest_reader
from .probe_batching import warmed_draws
from .progress import counter
from .suffix_family import SuffixFamily
from .tracker import RoundSummary, SynthesisTracker


@dataclass
class RoundClassifier:
    """One synthesis round's empty-seeded family, as it classifies that round's
    representative prefixes -- the round's attempt at the accept-preserving cut.
    ``votes[i]`` is prefix ``prefixes[i]``'s accept-rate over the family.

    The thresholds are the prefix/suffix tracker's own, so the cut here is the one
    synthesis made. ``calibrated[i]`` marks prefixes of the sampler length -- the
    population the family was clustered on. Off-length prefixes (boundary strings)
    reach the family off its calibration, so a consumer checking the recorded cut
    should restrict to the calibrated ones."""

    prefixes: List[bytes]
    votes: np.ndarray
    accept_thresh: float
    reject_thresh: float
    calibrated: np.ndarray

    @property
    def accept(self) -> np.ndarray:
        return self.votes >= self.accept_thresh

    @property
    def reject(self) -> np.ndarray:
        return self.votes < self.reject_thresh

    @property
    def decisive(self) -> np.ndarray:
        return self.accept | self.reject


def _round_classifier(pst, vs) -> RoundClassifier:
    mask = pst.table.representative
    prefixes = [p for p, keep in zip(pst.table.prefixes, mask) if keep]
    calibrated = np.array([len(p) == pst.sampler.length for p in prefixes], dtype=bool)
    return RoundClassifier(
        prefixes,
        pst.compute_decision(vs, mask),
        pst.accept_thresh,
        pst.reject_thresh,
        calibrated,
    )


#: P(some round certifies a DFA whose error is over certified_error), where the
#: signal is stated exactly.
CERTIFICATE_ALPHA = 1e-3


def _accumulate_harvest(strings, read, state, wanted) -> int:
    """Take up to ``wanted`` of the harvested ``strings`` ``state`` does not
    already hold, returning how many.

    Sorted then shuffled with a fixed rng, so the cap picks the same unbiased
    sample every run.
    """
    taken = sorted(set(strings) - state.seen)
    np.random.default_rng(0).shuffle(taken)
    for string in taken[:wanted]:
        state.take(string, read)
    return min(wanted, len(taken))


def _boundary_source(pst, state, *, acc_threshold) -> None:
    """Hands the round's boundary population a source that draws more the way
    its strings were found, proved only when a family search first asks it for
    more."""
    if state.harvesting is None:
        return
    state.sources[state.harvesting] = HarvestSource(
        Counter(state.harvest_reads),
        pst.rng,
        known=state.seen,
        acc_threshold=acc_threshold,
        # Its search walked deeper because its reads repeated, so they may run out.
        retirable=any(
            read.round.k > read.round.config.k for read in state.harvest_reads
        ),
    )


def _publish_pool(pst, state) -> int:
    """Put the round's populations in the table, returning how many of its
    prefixes are representative.

    Ends the round: the next one names a boundary population of its own.
    """
    # Retired before it is redefined, so a mid-round top-up's prefixes do not
    # outlive the round that bought them.
    for label in state.published - state.held.keys():
        pst.table.drop_population(label)
    for label, prefixes in ((UNIFORM, state.uniform), *state.held.items()):
        pst.table.drop_population(label)
        if prefixes:
            pst.table.add_prefixes(sorted(set(prefixes)), population=label)
    state.published = set(state.held)
    state.close_harvest()
    return int(pst.table.representative.sum())


#: Consecutive rounds with no progress. See `_StallDetector` for more details.
STALL_PATIENCE = 2

#: Rounds a run keeps going after the certificate first refuses a round at the
#: consistency target.
CERTIFICATE_PATIENCE = 5

#: The tightest the FNR limit gets, as a share of where it starts.
FNR_FLOOR = 0.25


class _StallDetector:
    """Stops a run whose rounds have stopped handing the next one new strings or a
    tighter family."""

    def __init__(self, patience: int):
        self._patience = patience
        self._stalled = 0

    def stalled(self, *, progressed: bool) -> bool:
        self._stalled = 0 if progressed else self._stalled + 1
        return self._stalled >= self._patience


class UncertifiedResult(UserWarning):
    """Synthesis returned a DFA the certificate did not pass."""


@dataclass
class BestRound:
    """The certified round's hypothesis, or else the best by ``score`` where rounds
    are scored and the most consistent one where they are not. Rounds
    are not monotone -- rebuilding the representative pool re-clusters, so a
    later family can classify worse -- so the run keeps this rather than the last
    round's. The boundary comes with it because denoising reads the labels
    against it."""

    consistency: float = -1.0
    score: Optional[float] = None
    dfa: Optional[DFA] = None
    tree: Optional[MidfixTree] = None
    boundary: Optional[float] = None
    round_index: Optional[int] = None
    #: The denoised DFA the certificate passed, if one did.
    certified: Optional[DFA] = None

    def consider(
        self, *, consistency, score, dfa, tree, boundary, round_index, certified
    ):
        rank = consistency if score is None else score
        held = self.consistency if self.score is None else self.score
        if (certified is not None, rank) > (self.certified is not None, held):
            self.consistency, self.score = consistency, score
            self.dfa, self.tree = dfa, tree
            self.boundary, self.round_index = boundary, round_index
            self.certified = certified


def _certified(pst, dfa, *, index, tracker):
    """denoise_accept_labels(dfa) if the certificate passes it, else None; and the
    state it blames."""
    output = denoise_accept_labels(pst, dfa)
    # Spread over the rounds, whichever of them reach the certificate.
    verdict = certifies(pst, output, alpha=look_level(CERTIFICATE_ALPHA, index))
    tracker.on_certificate_decided(verdict.certified, index)
    if verdict.certified:
        print(f"[round {index}] certified; stopping synthesis")
        return output, None
    print(f"[round {index}] at target, not certified; blames {verdict.blamed}")
    return None, verdict.blamed


def _next_harvest(pst, search, ending, blamed):
    """The strings the next family search is shown, and what draws more of them:
    a success the certificate refused shows the blamed state's undecided reads."""
    draw = UniformSource(pst).draw
    if blamed is None or ending.harvest:
        return ending.harvest, Replay(search, harvest_reader(ending), draw)
    leaf = search.hypothesis.tree.path_of(blamed)

    def at_leaf(by_edge):
        return [z for c in range(pst.alphabet_size) for z in by_edge.get((leaf, c), [])]

    return at_leaf(search.counts.edge_harvest), Replay(
        search, lambda charged: at_leaf(charged.undecided_by_edge), draw
    )


def _uncertified_too_long(index, uncertified_since) -> bool:
    if uncertified_since is None or index - uncertified_since < CERTIFICATE_PATIENCE:
        return False
    print(
        f"[round {index}] no hypothesis the certificate passes in "
        f"{CERTIFICATE_PATIENCE} "
        "rounds since the certificate first failed; stopping synthesis"
    )
    return True


def _counterexample_search(pst, vs, *, acc_threshold):
    """The round's search and how it ended."""
    family = SuffixFamily(pst, vs)
    search = CounterexampleSearch(
        counterexample_search_config(acc_threshold, pst.config.acceptable_fnr),
        lambda z: family.is_accept(z, b""),
        MidfixTree([pst.table.suffix(i) for i in vs]),
        max_k=start_length(pst.sampler.length),
    )
    with counter(search.config.max_probes, "Probing") as pbar:

        def draw():
            pbar.update(1)
            return pst.sampler.sample(pst.rng, pst.alphabet_size)

        ending = search.run(warmed_draws(search, draw, family.read_many))
    return search, ending


def _best_search(pst, vs, *, acc_threshold, index, selection):
    """The round's search, how it ended, and its DFA: of ``counterexample_restarts``
    searches on independent probe streams, the one with the best held-out phi."""
    restarts = pst.config.counterexample_restarts
    if restarts == 1:
        search, ending = _counterexample_search(pst, vs, acc_threshold=acc_threshold)
        return search, ending, _hypothesis_dfa(pst, search.hypothesis)
    runs = []
    for i, rng in enumerate(pst.rng.spawn(restarts)):
        pst.rng = rng
        search, ending = _counterexample_search(pst, vs, acc_threshold=acc_threshold)
        dfa = _hypothesis_dfa(pst, search.hypothesis)
        phi = held_out_phi(dfa, *selection)
        print(f"[round {index}] restart {i}: {ending.kind}, held-out phi {phi:.3f}")
        runs.append((phi, rng, search, ending, dfa))
    # The run goes on from the kept search's stream, as if it were the only one.
    _, pst.rng, search, ending, dfa = max(runs, key=lambda run: run[0])
    return search, ending, dfa


def _round_score(pst, dfa, selection, index) -> Optional[float]:
    """What ``BestRound`` ranks an uncertified round on in place of consistency."""
    if not pst.config.rank_rounds_by_held_out:
        return None
    score = held_out_phi(dfa, *selection)
    print(f"[round {index}] held-out phi: {score:.4f}")
    return score


#: Strings in each half of the held-out sample.  At 2,000 the DFA picked on it
#: scored 0.06 above its phi on fresh strings.
SELECTION_HALF = 8000


def _selection_sample(pst):
    """The held-out ``(fit, read)`` pair, or ``None`` where nothing is scored."""
    config = pst.config
    if config.counterexample_restarts == 1 and not config.rank_rounds_by_held_out:
        return None
    # A child stream, so scoring leaves the run's own draws as they were.
    rng = pst.rng.spawn(1)[0]
    return tuple(
        labelled_sample(pst.oracle, pst.sampler, pst.alphabet_size, SELECTION_HALF, rng)
        for _ in range(2)
    )


def _hypothesis_dfa(pst, hypothesis):
    boundary = pst.decision_boundary
    decide, _ = oracle_decider(
        pst.oracle, hypothesis.tree.base_family, boundary, boundary
    )
    initial = hypothesis.tree.classify(b"", decide)
    return hypothesis.to_dfa(pst.alphabet_size, 0 if initial is None else initial)


def _round_summary(pst, vs, search, ending, seconds) -> RoundSummary:
    """``seconds`` is the round's ``(sampling, probing)`` time."""
    tree = search.hypothesis.tree
    assert tree.num_states >= 2
    return RoundSummary(
        prefixes=pst.num_prefixes,
        representative_prefixes=int(pst.table.representative.sum()),
        suffixes=pst.table.num_suffixes,
        family_size=len(vs),
        probes=search.probes,
        ending=ending.kind,
        hypothesis_states=tree.num_states,
        hypothesis_edges=len(search.hypothesis.edges),
        sampling_seconds=seconds[0],
        probing_seconds=seconds[1],
    )


def counterexample_driven_synthesis(
    pst,
    *,
    acc_threshold: float,
    tracker: SynthesisTracker,
    max_rounds: Optional[int] = None,
    indecisive_fraction: float = 0.1,
    min_indecisive: int = 200,
) -> BestRound:
    """Rounds until a hypothesis is certified, the pool stalls, the rounds since
    the first refusal run out of patience, or max_rounds of them have run.
    Only a caller driving the loop itself can set that cap; `learn_dfa` does not
    forward one."""
    # The cap is read once a round has run, so a round always runs.
    assert max_rounds is None or max_rounds >= 1, max_rounds
    # Kept across rounds: the FNR gate resolves the chain one state per round, so
    # earlier rounds' boundary strings keep the family honest about the whole
    # chain (they turn decisive once their state is resolved).
    uniform = [
        p for p, keep in zip(pst.table.prefixes, pst.table.representative) if keep
    ]
    state = PoolState(uniform, pst.rng.bytes(16))
    stall = _StallDetector(STALL_PATIENCE)
    best = BestRound()
    selection = _selection_sample(pst)
    # Round of the first refusal; CERTIFICATE_PATIENCE counts from it.
    uncertified_since = None
    index = 0
    while True:
        print(f"[round {index}] starting with {pst.num_prefixes} prefixes")
        started = time.monotonic()
        vs, boundary = sample_suffix_family(pst, pst.table.intern_suffix(b""), state)
        pst.decision_boundary = boundary
        tracker.on_family_resolved([pst.table.suffix(i) for i in vs], boundary, index)
        classifier = _round_classifier(pst, vs)
        tracker.on_round_classified(classifier, index)
        sampled = time.monotonic()
        search, ending, dfa = _best_search(
            pst, vs, acc_threshold=acc_threshold, index=index, selection=selection
        )
        summary = _round_summary(
            pst, vs, search, ending, (sampled - started, time.monotonic() - sampled)
        )
        tracker.on_search_ended(ending, index)
        tracker.on_round_summarized(summary, index)
        dt = search.hypothesis.tree
        print(
            f"[round {index}] {summary.ending} after {summary.probes} probes: "
            f"{summary.hypothesis_states} states, {summary.hypothesis_edges} edges "
            f"learned, over a family of {summary.family_size} suffixes "
            f"({summary.sampling_seconds:.1f}s sampling, "
            f"{summary.probing_seconds:.1f}s probing)"
        )
        tracker.on_initial_dfa_found(dfa, dt, index)
        print(dfa)
        true_acc = estimate_agreement_rate(
            pst,
            pst.sampler,
            pst.oracle,
            dt,
            dfa,
            num_samples=2000,
            acc_threshold=acc_threshold,
        )
        print(f"[round {index}] DFA/DT consistency on fresh samples: {true_acc:.4f}")
        tracker.on_consistency_estimated(true_acc, index)
        # Only a round that would otherwise return is worth the certificate's reads.
        output, blamed = None, None
        if true_acc >= acc_threshold:
            output, blamed = _certified(pst, dfa, index=index, tracker=tracker)
            uncertified_since = (
                index if uncertified_since is None else uncertified_since
            )
        best.consider(
            consistency=true_acc,
            score=_round_score(pst, dfa, selection, index),
            dfa=dfa,
            tree=dt,
            boundary=pst.decision_boundary,
            round_index=index,
            certified=output,
        )
        if output is not None:
            return best
        if _uncertified_too_long(index, uncertified_since):
            return best
        # Everything below readies the next round.
        if max_rounds is not None and index + 1 >= max_rounds:
            print(f"[round {index}] ran the {max_rounds} rounds asked for")
            return best
        target = max(int(indecisive_fraction * pst.num_prefixes), min_indecisive)
        strings, provenance = _next_harvest(pst, search, ending, blamed)
        taken = _accumulate_harvest(strings, provenance, state, target)
        halved = taken == 0 and pst.fnr_limit / 2 >= FNR_FLOOR * pst.config.fnr_limit
        if halved:
            pst.fnr_limit /= 2
            print(f"[round {index}] FNR limit now {pst.fnr_limit:.4f}")
        # Rounds after a refusal have their own patience, so they are not
        # weighed for a stall.
        if uncertified_since is None and stall.stalled(progressed=taken > 0 or halved):
            print(
                f"[round {index}] no new strings harvested in {STALL_PATIENCE} "
                "rounds; stopping synthesis"
            )
            return best
        _boundary_source(pst, state, acc_threshold=acc_threshold)
        pool = _publish_pool(pst, state)
        print(
            f"[round {index}] pool now {pool} representative prefixes, "
            f"{len(state.seen)} boundary strings harvested so far"
        )
        index += 1


def do_counterexample_driven_synthesis(
    pst, *, acc_threshold: float, tracker: SynthesisTracker
) -> Optional[DFA]:
    best = counterexample_driven_synthesis(
        pst, acc_threshold=acc_threshold, tracker=tracker
    )
    if best.dfa is None:
        return None
    pst.decision_boundary = best.boundary
    dfa = best.certified
    if dfa is None:
        warnings.warn(
            f"no round was certified; returning round {best.round_index}'s DFA",
            UncertifiedResult,
        )
        dfa = denoise_accept_labels(pst, best.dfa)
    tracker.on_corrected_dfa_found(dfa, best.round_index)
    return dfa
