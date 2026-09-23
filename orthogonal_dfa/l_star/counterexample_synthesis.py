"""
Counterexample-driven synthesis: the E-L* learner loop.

Each round builds a DFA from the current prefix pool and splits it in place on
DFA-vs-tree disagreements (the counterexample pass).

When the estimate still falls short, the representative pool is rebuilt to add
    - boundary strings the family could not place
    - per-state balanced sample

These drive the suffix-family FNR gate to re-cluster and resolve them
in the next round.
"""

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import scipy.stats
from automata.fa.dfa import DFA

from .cluster import sample_suffix_family
from .dfa_utils import (
    count_paths_to_state,
    sample_string_reaching_state,
    uniform_weights,
)
from .lstar import denoise_accept_labels, estimate_agreement_rate
from .mask_table import UNIFORM
from .midfix_tree import MidfixTree
from .preconditions import DEFAULT_MIN_COVERAGE
from .prefix_populations import PoolState
from .prefix_sources import BoundarySource, aim_at, state_source
from .progress import track
from .split_evidence import MEMBERS_TO_RULE_OUT_A_SPLIT
from .tracker import SynthesisTracker
from .transition_resolver import TransitionResolver


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


#: Probes drawn per counterexample pass.
COUNTEREXAMPLE_PROBES = 4000

#: Candidates weighed against a leaf that does not read as one class.  Each
#: costs every member read against the whole family behind it, so this is the
#: scan's price where the cheap read has already found something.
SPLIT_CANDIDATE_PATIENCE = 24

#: Rate at which the cheap read may call a single-class leaf mixed.  A false one
#: costs the candidates above; missing a real one costs the merge.
SPLIT_SCAN_ALPHA = 1e-3

#: How far a leaf's accept rate must sit inside the two pure rates before the read
#: above is believed.
#:
#: The rates it is compared against are `decision_boundary +- min_signal_strength`, and
#: the boundary is re-estimated every round -- it moves by a percent or two between
#: them.  The count of prefixes grows with the leaf's share, so the largest leaves are
#: read precisely enough to resolve far below that drift, and a pure leaf then reads
#: many sigma off a rate that is itself wrong.  Under this much, the deviation says the
#: boundary is mis-estimated rather than the leaf is mixed.
#:
#: A minority of share `w` moves the rate by `2 * min_signal_strength * w`, so this is a
#: floor on the minority worth another round rather than on the rate alone.
SPLIT_SCAN_MIN_DEVIATION = 0.05


def _split_until_settled(pst, resolver, vs, best, *, index, acc_threshold):
    """Split every leaf the split test will take, re-reading the hypothesis each
    time so the round returns the split one."""
    (
        dfa,
        dt,
    ) = resolver.to_dfa_and_tree()
    merged = []
    true_acc = None
    for _ in range(STALL_PATIENCE):
        reached, merged = split_unreached_leaves(resolver, pst, vs, dfa)
        if not reached:
            break
        dfa, dt = resolver.to_dfa_and_tree()
        true_acc = estimate_agreement_rate(
            pst,
            pst.sampler,
            pst.oracle,
            dt,
            dfa,
            num_samples=2000,
            acc_threshold=acc_threshold,
        )
        print(
            f"[round {index}] the split test reached {reached} leaf(s) the "
            f"counterexample pass could not: {dt.num_states} states, "
            f"consistency {true_acc:.4f}"
        )
    if true_acc is not None:
        best.consider(
            consistency=true_acc,
            dfa=dfa,
            tree=dt,
            boundary=pst.decision_boundary,
            round_index=index,
            merged=bool(merged),
        )
    return merged


def scan_prefixes(signal, leaf_share, boundary, min_coverage=DEFAULT_MIN_COVERAGE):
    """Prefixes to read at a leaf for a class of ``min_coverage`` mass to show
    in its accept rate.

    A minority of share `w` within the leaf moves the rate by `2 * signal * w`
    against a standard error of `sqrt(p(1-p)/n)`.  Such a class is a
    `min_coverage / leaf_share` share of the leaf, so the count grows with the
    square of that share: the leaves worth many prefixes are the large ones a
    rare class can hide in, and a flat share to resolve would miss it in exactly
    those.  One query each, so a leaf holding most of the sampler costs tens of
    thousands and the rest cost far less.
    """
    z = scipy.stats.norm.isf(SPLIT_SCAN_ALPHA / 2)
    p = boundary + signal
    want = min_coverage / max(leaf_share, min_coverage)
    return int(np.ceil(p * (1 - p) * (z / (2 * signal * want)) ** 2))


def reads_as_one_class(pst, dfa, state, alpha=SPLIT_SCAN_ALPHA):
    """Whether a state's own prefixes read as a single class at the empty suffix.

    The prefixes are aimed straight at the state off the path counts, so none
    are drawn to be discarded and none are read against anything but themselves
    -- one query each, where a candidate distinguisher costs every member read
    against the whole family behind it.  This sees a class merged with one of
    the opposite label and not two that share one; the split test would catch
    those, but not at a price every run can pay.
    """
    signal = pst.config.min_signal_strength
    weights = uniform_weights(dfa)
    reaching = count_paths_to_state(dfa, state, pst.sampler.length, weights)
    space = pst.alphabet_size**pst.sampler.length
    share = reaching[pst.sampler.length][dfa.initial_state] / space
    wanted = scan_prefixes(signal, share, pst.decision_boundary)
    drawn = (
        sample_string_reaching_state(dfa, reaching, pst.rng, weights)
        for _ in range(wanted)
    )
    pool = [p for p in drawn if p is not None]
    if len(pool) < MEMBERS_TO_RULE_OUT_A_SPLIT:
        return True
    hits = int(sum(pst.table.memo.membership_queries(pool)))
    n = len(pool)
    accept_rate = pst.decision_boundary + signal
    reject_rate = pst.decision_boundary - signal
    inside = min(accept_rate - hits / n, hits / n - reject_rate)
    if inside < SPLIT_SCAN_MIN_DEVIATION:
        return True
    not_accept = scipy.stats.binom.cdf(hits, n, accept_rate)
    not_reject = scipy.stats.binom.sf(hits - 1, n, reject_rate)
    return not (not_accept < alpha / 2 and not_reject < alpha / 2)


def split_unreached_leaves(resolver, pst, vs, dfa) -> Tuple[int, List[int]]:
    """Put the leaves that do not read as one class to the split test with the
    round's own suffixes, returning how many split and which of the rest the
    gate flagged.

    The counterexample pass proposes a distinguisher only where the tree and the
    DFA disagree, so a leaf they agree about is never weighed at all -- and a
    leaf holding two classes the family votes the same way is exactly that.
    The split test itself is unchanged: it groups on one half of the family and
    scores on the disjoint other, so a leaf holding one class cannot be split by
    the noise that grouped it.
    """
    candidates = [v for v in (pst.table.suffix(i) for i in vs) if v]
    split = 0
    merged = []
    for leaf in list(range(resolver.num_states)):
        if leaf not in dfa.states or reads_as_one_class(pst, dfa, leaf):
            continue
        distinguisher = resolver.splits.first_split(
            resolver.splits.scan_members(leaf),
            candidates,
            patience=SPLIT_CANDIDATE_PATIENCE,
        )
        if distinguisher is None:
            merged.append(leaf)
            continue
        resolver.split_on(leaf, distinguisher)
        split += 1
    return split, merged


def _default_patience(acc_threshold: float) -> int:
    """Consecutive clean probes that end a counterexample pass: seeing this many
    in a row is a ``<= 0.05`` event if the disagreement rate were still at the
    tolerated ``1 - acc_threshold``.

    A perfect-accuracy target tolerates no disagreement, so no finite clean run
    rules it out -- never early-stop, run the whole probe budget."""
    if acc_threshold >= 1:
        return COUNTEREXAMPLE_PROBES
    return math.ceil(math.log(0.05) / math.log(acc_threshold))


def _accumulate_indecisive(resolver, state, wanted) -> int:
    """Take up to ``wanted`` of the round's boundary strings ``state`` does not
    already hold, returning how many.

    Sorted then shuffled with a fixed rng, so the cap picks the same unbiased
    sample every run.
    """
    taken = sorted(resolver.indecisive - state.seen)
    np.random.default_rng(0).shuffle(taken)
    for string in taken[:wanted]:
        state.seen.add(string)
        state.harvest().append(string)
    return min(wanted, len(taken))


def _per_state_members(pst, resolver, dfa, state, per_state) -> None:
    """``("state", leaf) -> members``, ``per_state`` of them resting at each
    state that has a source."""
    state.retire_states()
    for leaf in track(range(resolver.num_states), "Drawing each state's prefixes"):
        aim = aim_at(pst, dfa, leaf)
        if aim is None:
            # Out of reach rather than short: too few strings of the sampler's
            # length arrive here to draw from, so no round is going to fill it
            # and this one is not waiting on a draw.
            continue
        source = state_source(resolver, leaf, aim, wanted=per_state)
        if source is None:
            continue
        state.held[("state", leaf)] = sorted(source.draw() for _ in range(per_state))
        state.sources[("state", leaf)] = source


def _top_up_boundary(pst, resolver, dfa, state, wanted) -> None:
    """Probe for up to ``wanted`` more boundary strings, keeping what the yield
    test turned up even when the source fails it.

    A round with its fill already still leaves the population a source, unproved:
    otherwise the only population the counterexample pass fills for free is the
    one a later round has nothing to draw with.
    """
    source = BoundarySource(pst, resolver.sifter, dfa.transitions, known=state.seen)
    if wanted > 0:
        # Not `has_sufficient_yield`: same probes, but the verdict is not kept.
        drawing = source.worth_drawing()
        found = source.found()
        if drawing:
            found += [source.draw() for _ in range(wanted - len(found))]
        for string in found[:wanted]:
            state.seen.add(string)
            state.harvest().append(string)
    if state.harvesting is not None:
        state.sources[state.harvesting] = source


def _aimed_at(pst, resolver, dfa) -> set:
    """The leaves the round aims at, which are the ones its aims settle strings
    into -- `state_source` proves a leaf's yield by aiming at it, so a leaf
    whose yield comes out too low has still been filled by the proving.
    """
    return {
        leaf
        for leaf in range(resolver.num_states)
        if aim_at(pst, dfa, leaf) is not None
    }


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
    state.harvesting = None
    return int(pst.table.representative.sum())


#: Consecutive rounds with no progress. See `_StallDetector` for more details.
STALL_PATIENCE = 2


class _StallDetector:
    """Stops a run that has started repeating itself. We consider a round stalled if

    1. There are no new states
    2. (Internal) accuracy has not increased
    3. No distinguisher the tree can propose still splits a state

    Deliberately fairly restrictive, so we can have a low Patience before
    exiting the loop.
    """

    def __init__(self, patience: int):
        self._patience = patience
        self._states = 0
        self._stalled = 0

    def stalled(self, *, states: int, improved: bool, settled) -> bool:
        progressed = states > self._states or improved or not settled()
        self._stalled = 0 if progressed else self._stalled + 1
        self._states = states
        return self._stalled >= self._patience


#: Representative strings drawn per DFA state.  Every round draws this many
#: afresh through the state's source and replaces the last round's, so the
#: population does not accumulate across rounds.  Over
#: `MEMBERS_TO_RULE_OUT_A_SPLIT` with room to spare, since the draws the family
#: cannot place are not among the ones that rule a split out.
PER_STATE = 50


@dataclass
class BestRound:
    """The most consistent round's hypothesis. Rounds are not monotone --
    rebuilding the representative pool re-clusters, so a later family can
    classify worse -- so the run keeps this rather than the last round's. The
    boundary comes with it because denoising reads the labels against it."""

    consistency: float = -1.0
    #: Whether a leaf of this hypothesis still read as more than one class.
    merged: bool = True
    dfa: Optional[DFA] = None
    tree: Optional[MidfixTree] = None
    boundary: Optional[float] = None
    round_index: Optional[int] = None

    def consider(self, *, consistency, dfa, tree, boundary, round_index, merged=False):
        # A merge is what consistency scores best on, so the score cannot be the whole
        # ranking: a hypothesis a leaf check caught is behind every one it did not.
        if (not merged, consistency) > (not self.merged, self.consistency):
            self.consistency = consistency
            self.merged = merged
            self.dfa, self.tree = dfa, tree
            self.boundary, self.round_index = boundary, round_index


def counterexample_driven_synthesis(
    pst,
    *,
    acc_threshold: float,
    tracker: SynthesisTracker,
    max_rounds: Optional[int] = None,
    per_state: int = PER_STATE,
    indecisive_fraction: float = 0.1,
    min_indecisive: int = 200,
) -> BestRound:
    """Rounds until the hypothesis is consistent enough, the pool stalls, or
    ``max_rounds`` of them have run.  Only a caller driving the loop itself can
    set that cap; `learn_dfa` does not forward one."""
    # The cap is read at the foot of the body, so a round always runs.
    assert max_rounds is None or max_rounds >= 1, max_rounds
    patience = _default_patience(acc_threshold)
    # Kept across rounds: the FNR gate resolves the chain one state per round, so
    # earlier rounds' boundary strings keep the family honest about the whole
    # chain (they turn decisive once their state is resolved).
    uniform = [
        p for p, keep in zip(pst.table.prefixes, pst.table.representative) if keep
    ]
    state = PoolState(uniform)
    stall = _StallDetector(STALL_PATIENCE)
    best = BestRound()
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
        resolver = TransitionResolver(pst, vs)
        resolver.close_edges()
        resolver.counterexample_pass(
            max_probes=COUNTEREXAMPLE_PROBES, patience=patience
        )
        dfa, dt = resolver.to_dfa_and_tree()
        print(
            f"[round {index}] resolved {dt.num_states} states over a family of "
            f"{len(vs)} suffixes ({sampled - started:.1f}s sampling, "
            f"{time.monotonic() - sampled:.1f}s resolving)"
        )
        assert dt.num_states >= 2
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
        if true_acc >= acc_threshold:
            merged = _split_until_settled(
                pst,
                resolver,
                vs,
                best,
                index=index,
                acc_threshold=acc_threshold,
            )
            best.consider(
                consistency=true_acc,
                dfa=dfa,
                tree=dt,
                boundary=pst.decision_boundary,
                round_index=index,
                merged=bool(merged),
            )
            if not merged:
                print(
                    f"[round {index}] reached the target DFA/DT consistency of "
                    f"{acc_threshold:.4f}; stopping synthesis"
                )
                return best
            print(
                f"[round {index}] consistency {true_acc:.4f} is at target, but "
                f"leaf(s) {merged} read as more than one class and no candidate "
                f"cut them; carrying on"
            )
            dfa, dt = resolver.to_dfa_and_tree()
        else:
            best.consider(
                consistency=true_acc,
                dfa=dfa,
                tree=dt,
                boundary=pst.decision_boundary,
                round_index=index,
            )
        target = max(int(indecisive_fraction * pst.num_prefixes), min_indecisive)
        taken = _accumulate_indecisive(resolver, state, target)
        _per_state_members(pst, resolver, dfa, state, per_state)
        # Asked after the aims, which are what fill the leaves it reads.  A
        # leaf nothing aims at is not one the round waits on.
        if stall.stalled(
            states=dt.num_states,
            improved=best.round_index == index,
            settled=lambda: resolver.splits.nothing_left_to_split(
                _aimed_at(pst, resolver, dfa)
            ),
        ):
            print(
                f"[round {index}] no progress ({dt.num_states} states) in "
                f"{STALL_PATIENCE} rounds -- pool churning without resolving; "
                "stopping synthesis"
            )
            return best
        # Last, so what the draws and the check strand lands in the pool the
        # round they were found rather than the round after.
        taken += _accumulate_indecisive(resolver, state, target - taken)
        _top_up_boundary(pst, resolver, dfa, state, target - taken)
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
    pst, *, acc_threshold: float, tracker: SynthesisTracker
) -> Optional[DFA]:
    best = counterexample_driven_synthesis(
        pst, acc_threshold=acc_threshold, tracker=tracker
    )
    if best.dfa is None:
        return None
    pst.decision_boundary = best.boundary
    dfa = denoise_accept_labels(pst, best.dfa)
    tracker.on_corrected_dfa_found(dfa, best.round_index)
    return dfa
