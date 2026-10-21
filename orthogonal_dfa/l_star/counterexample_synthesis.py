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
from .lstar import denoise_accept_labels, estimate_agreement_rate
from .mask_table import UNIFORM
from .midfix_tree import MidfixTree, oracle_decider
from .preconditions import start_length
from .prefix_populations import PoolState
from .prefix_sources import (
    HarvestSource,
    Replay,
    UniformSource,
    aim_at,
    harvest_reader,
    state_source,
)
from .probe_batching import warmed_draws
from .progress import counter, track
from .round_end import HARVESTS, SUCCESS
from .suffix_family import SuffixFamily
from .tracker import SynthesisTracker


@dataclass
class RoundClassifier:
    """One synthesis round's empty-seeded family, as it classifies that round's
    representative prefixes -- the round's attempt at the accept-preserving cut.
    ``votes[i]`` is prefix ``prefixes[i]``'s accept-rate over the family.

    The thresholds are the prefix/suffix tracker's own, so the cut here is the one
    synthesis made. ``calibrated[i]`` marks prefixes of the sampler length -- the
    population the family was clustered on. Off-length prefixes (boundary strings,
    per-state samples) reach the family off its calibration, so a consumer checking
    the recorded cut should restrict to the calibrated ones."""

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


def _family_too_loose(ending, true_acc, acc_threshold, taken) -> bool:
    return ending.kind in HARVESTS or (
        ending.kind == SUCCESS and true_acc >= acc_threshold and taken == 0
    )


def _per_state_members(pst, hypothesis, dfa, state, per_state) -> None:
    """``("state", leaf) -> members``, ``per_state`` of them resting at each
    state that has a source."""
    state.retire("state")
    for leaf in track(
        range(hypothesis.tree.num_states), "Drawing each state's prefixes"
    ):
        aim = aim_at(pst, dfa, leaf)
        if aim is None:
            # Out of reach rather than short: too few strings of the sampler's
            # length arrive here to draw from, so no round is going to fill it
            # and this one is not waiting on a draw.
            continue
        path = hypothesis.tree.path_of(leaf)
        source = state_source(lambda s, path=path: hypothesis.sift(s) == path, aim)
        if source is None:
            continue
        state.hold(("state", leaf), source, per_state)


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
    """Stops a run whose rounds have stopped handing the next one new strings."""

    def __init__(self, patience: int):
        self._patience = patience
        self._stalled = 0

    def stalled(self, *, progressed: bool) -> bool:
        self._stalled = 0 if progressed else self._stalled + 1
        return self._stalled >= self._patience


#: Representative strings drawn per DFA state.  Every round draws this many
#: afresh through the state's source and replaces the last round's, so the
#: population does not accumulate across rounds.
PER_STATE = 50


class UncertifiedResult(UserWarning):
    """Synthesis returned a DFA the certificate did not pass."""


@dataclass
class BestRound:
    """The certified round's hypothesis, or else the most consistent one. Rounds
    are not monotone -- rebuilding the representative pool re-clusters, so a
    later family can classify worse -- so the run keeps this rather than the last
    round's. The boundary comes with it because denoising reads the labels
    against it."""

    consistency: float = -1.0
    dfa: Optional[DFA] = None
    tree: Optional[MidfixTree] = None
    boundary: Optional[float] = None
    round_index: Optional[int] = None
    #: The denoised DFA the certificate passed, if one did.
    certified: Optional[DFA] = None

    def consider(self, *, consistency, dfa, tree, boundary, round_index, certified):
        if (certified is not None, consistency) > (
            self.certified is not None,
            self.consistency,
        ):
            self.consistency = consistency
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


#: Start checks one round runs.  One resume from the check's failures is
#: expected to mend the start; a second covers the resumed probes' splits
#: disturbing it again; a start failing past that is left to the next family.
CHECK_ATTEMPTS = 3


def _searched_and_checked(pst, vs, *, acc_threshold, k, index):
    """The round's search, how it ended, its DFA, and the DFA's agreement with
    the tree from ε.  Probes walked from k > 0 never run the DFA from ε, so a
    search that succeeds while that agreement falls short probes the strings
    it disagreed on from ε and resumes."""
    family = SuffixFamily(pst, vs)
    search = CounterexampleSearch(
        counterexample_search_config(acc_threshold, pst.config.acceptable_fnr, k),
        lambda z: family.is_accept(z, b""),
        MidfixTree([pst.table.suffix(i) for i in vs]),
    )
    for attempt in range(1, CHECK_ATTEMPTS + 1):
        ending = _run(pst, search, family)
        dfa = _hypothesis_dfa(pst, search.hypothesis)
        true_acc, disagreeing = estimate_agreement_rate(
            pst,
            pst.sampler,
            pst.oracle,
            search.hypothesis.tree,
            dfa,
            num_samples=2000,
            acc_threshold=acc_threshold,
        )
        print(
            f"[round {index}] DFA/DT consistency on fresh samples: {true_acc:.4f} "
            f"after {ending.kind}, k={k}, check {attempt}"
        )
        if (
            k == 0
            or ending.kind != SUCCESS
            or true_acc >= acc_threshold
            or attempt == CHECK_ATTEMPTS
        ):
            break
        version = search.hypothesis.version
        search.settle(disagreeing)
        if search.hypothesis.version == version:
            break
        print(
            f"[round {index}] {len(disagreeing)} start-check failures probed "
            "from ε; resuming the search"
        )
    return search, ending, dfa, true_acc


def _run(pst, search, family):
    with counter(search.config.max_probes, "Probing") as pbar:

        def draw():
            pbar.update(1)
            return pst.sampler.sample(pst.rng, pst.alphabet_size)

        return search.run(warmed_draws(search, draw, family.read_many))


def _walk_deeper(pst, state, k, harvested, strings, *, taken, index) -> int:
    """Double k, from 0 to 1 and at most start_length(L), after a round that
    took no new strings or repeated one of harvested.  Reaching
    start_length(L) drops every population holding a shorter string."""
    deepest = start_length(pst.sampler.length)
    repeats = not harvested.isdisjoint(strings)
    harvested.update(strings)
    if k >= deepest or (taken > 0 and not repeats):
        return k
    deeper = min(max(2 * k, 1), deepest)
    print(f"[round {index}] k now {deeper}")
    if deeper == deepest:
        _drop_shallow_populations(state, deepest, index=index)
    return deeper


def _drop_shallow_populations(state, depth, *, index) -> None:
    shallow = [
        label
        for label, strings in state.held.items()
        if any(len(string) < depth for string in strings)
    ]
    dropped = 0
    for label in shallow:
        strings = state.held.pop(label)
        dropped += len(strings)
        state.seen.difference_update(strings)
        state.seen.difference_update(state.reserves.pop(label, ()))
        state.sources.pop(label, None)
    if state.harvesting in shallow:
        state.close_harvest()
    print(
        f"[round {index}] dropped {len(shallow)} populations, {dropped} strings, "
        f"holding strings shorter than {depth}"
    )


def _hypothesis_dfa(pst, hypothesis):
    boundary = pst.decision_boundary
    decide, _ = oracle_decider(
        pst.oracle, hypothesis.tree.base_family, boundary, boundary
    )
    initial = hypothesis.tree.classify(b"", decide)
    return hypothesis.to_dfa(pst.alphabet_size, 0 if initial is None else initial)


def counterexample_driven_synthesis(
    pst,
    *,
    acc_threshold: float,
    tracker: SynthesisTracker,
    initial_k: int,
    max_rounds: Optional[int] = None,
    per_state: int = PER_STATE,
    indecisive_fraction: float = 0.1,
    min_indecisive: int = 200,
) -> BestRound:
    """Rounds until a hypothesis is certified, the pool stalls, the rounds since
    the first refusal run out of patience, or max_rounds of them have run.
    Only a caller driving the loop itself can set that cap; `learn_dfa` does not
    forward one."""
    # The cap is read at the foot of the body, so a round always runs.
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
    # Round of the first refusal; CERTIFICATE_PATIENCE counts from it.
    uncertified_since = None
    k = initial_k
    harvested = set()
    index = 0
    while True:
        print(f"[round {index}] starting with {pst.num_prefixes} prefixes, k={k}")
        started = time.monotonic()
        vs, boundary = sample_suffix_family(pst, pst.table.intern_suffix(b""), state)
        pst.decision_boundary = boundary
        tracker.on_family_resolved([pst.table.suffix(i) for i in vs], boundary, index)
        classifier = _round_classifier(pst, vs)
        tracker.on_round_classified(classifier, index)
        sampled = time.monotonic()
        search, ending, dfa, true_acc = _searched_and_checked(
            pst, vs, acc_threshold=acc_threshold, k=k, index=index
        )
        hypothesis, dt = search.hypothesis, search.hypothesis.tree
        tracker.on_search_ended(ending, index)
        print(
            f"[round {index}] {ending.kind} after {search.probes} probes: "
            f"{dt.num_states} states, {len(hypothesis.edges)} edges learned, over a "
            f"family of {len(vs)} suffixes ({sampled - started:.1f}s sampling, "
            f"{time.monotonic() - sampled:.1f}s probing)"
        )
        assert dt.num_states >= 2
        tracker.on_initial_dfa_found(dfa, dt, index)
        print(dfa)
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
        strings, provenance = _next_harvest(pst, search, ending, blamed)
        taken = _accumulate_harvest(
            strings,
            provenance,
            state,
            max(int(indecisive_fraction * pst.num_prefixes), min_indecisive),
        )
        if (
            _family_too_loose(ending, true_acc, acc_threshold, taken)
            and pst.fnr_limit / 2 >= FNR_FLOOR * pst.config.fnr_limit
        ):
            pst.fnr_limit /= 2
            print(f"[round {index}] FNR limit now {pst.fnr_limit:.4f}")
        _per_state_members(pst, hypothesis, dfa, state, per_state)
        deeper = _walk_deeper(
            pst, state, k, harvested, strings, taken=taken, index=index
        )
        # Rounds after a refusal have their own patience, and a round that
        # walks deeper is progress, so neither is weighed for a stall.
        if (
            deeper == k
            and uncertified_since is None
            and stall.stalled(progressed=taken > 0)
        ):
            print(
                f"[round {index}] no new strings harvested in {STALL_PATIENCE} "
                "rounds; stopping synthesis"
            )
            return best
        k = deeper
        _boundary_source(pst, state, acc_threshold=acc_threshold)
        pool = _publish_pool(pst, state)
        print(
            f"[round {index}] pool now {pool} representative prefixes, "
            f"{len(state.seen)} boundary strings harvested so far"
        )
        index += 1
        if max_rounds is not None and index >= max_rounds:
            print(f"[round {index - 1}] ran the {max_rounds} rounds asked for")
            return best


def do_counterexample_driven_synthesis(
    pst, *, acc_threshold: float, tracker: SynthesisTracker, initial_k: int
) -> Optional[DFA]:
    best = counterexample_driven_synthesis(
        pst, acc_threshold=acc_threshold, tracker=tracker, initial_k=initial_k
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
