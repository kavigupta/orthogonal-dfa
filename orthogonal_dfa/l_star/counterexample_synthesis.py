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
from typing import List, Optional

import numpy as np
from automata.fa.dfa import DFA

from .cluster import sample_suffix_family
from .lstar import denoise_accept_labels, estimate_agreement_rate
from .mask_table import UNIFORM
from .midfix_tree import MidfixTree
from .prefix_sources import (
    BoundarySource,
    UniformSource,
    aim_at,
    draw_many,
    state_source,
)
from .progress import track
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


class _PoolState:
    """The pool state carried across rounds: the initial uniform sample (kept in
    the representative set every round so global calibration stays anchored to the
    sampling distribution even if the per-state sample is skewed), and the
    populations the rounds have made, with a ``seen`` set to dedup the boundary
    strings across them."""

    def __init__(self, uniform):
        self.uniform = list(uniform)
        self.held = {}
        #: What draws more of each population, for the round that asks.
        self.sources = {}
        self.seen = set()
        #: Boundary populations named so far, which is what numbers them.
        self.named = 0
        #: The one this round is filling, or None before it strands anything.
        self.harvesting = None
        #: Labels the table holds, so a round retires what it does not renew.
        self.published = set()

    def retire_states(self) -> None:
        """Forget last round's state populations: this round's states are the
        ones there are, and a leaf it does not have is not one to go on
        publishing."""
        for stale in [label for label in self.held if label[0] == "state"]:
            self.held.pop(stale)
            self.sources.pop(stale, None)

    def harvest(self) -> list:
        """This round's boundary population, named on the first string to reach
        it."""
        if self.harvesting is None:
            self.named += 1
            self.harvesting = ("boundary", self.named)
            self.held[self.harvesting] = []
        return self.held[self.harvesting]


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
    test turned up even when the source fails it."""
    if wanted <= 0:
        return
    source = BoundarySource(pst, resolver.sifter, dfa.transitions, known=state.seen)
    worth_drawing = source.has_sufficient_yield()
    found = source.found()
    if worth_drawing:
        found += [source.draw() for _ in range(wanted - len(found))]
    for string in found[:wanted]:
        state.seen.add(string)
        state.harvest().append(string)
    if state.harvesting is not None:
        state.sources[state.harvesting] = source


def grow_population(pst, state, label) -> bool:
    """Draw more prefixes for one population, saying whether it could.

    A population nothing draws for any more is retired here, table and all: a
    rate the round cannot answer is not one to hold a family to.
    """
    if label == UNIFORM:
        pst.sample_more_prefixes()
        return True
    source = state.sources.get(label)
    if source is None or not source.worth_drawing():
        # Forgotten, not held aside, so a later round that strands one of these
        # again can pool it behind a source that does draw.
        state.seen.difference_update(state.held.pop(label, ()))
        state.sources.pop(label, None)
        pst.table.drop_population(label)
        return False
    # As many as the uniform draw this stands in for adds: growing the
    # population a refusal names is a substitution, not a token.
    drawn = [source.draw() for _ in range(pst.config.num_addtl_prefixes)]
    state.held.setdefault(label, []).extend(drawn)
    pst.table.add_prefixes(sorted(set(drawn)), population=label)
    return True


class _Populations:
    """What the next round's family search may ask of this round's populations:
    more prefixes for one of them, or prefixes to read the split on."""

    def __init__(self, pst, state):
        self._pst = pst
        self._state = state

    def labels(self) -> list:
        """The populations a family is read over: this round's, and the uniform
        pool the table keeps across rounds."""
        return [UNIFORM, *self._state.held]

    def grow(self, label) -> bool:
        return grow_population(self._pst, self._state, label)

    def for_split(self, label, wanted: int) -> list:
        """Prefixes for one population, to read the split on and not to keep.

        Empty where nothing draws for it: no say in the split rather than a
        split held up.  `grow` is what retires it.
        """
        source = (
            UniformSource(self._pst)
            if label == UNIFORM
            else self._state.sources.get(label)
        )
        # Already proven, or not drawn from: proving one costs a round's worth
        # of probes, which is not what a handful of prefixes to read a split on
        # is worth.  A population the round never proved sits this one out.
        if source is None or not source.proven:
            return []
        return draw_many(source, wanted)


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
    dfa: Optional[DFA] = None
    tree: Optional[MidfixTree] = None
    boundary: Optional[float] = None
    round_index: Optional[int] = None

    def consider(self, *, consistency, dfa, tree, boundary, round_index):
        if consistency > self.consistency:
            self.consistency = consistency
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
    state = _PoolState(uniform)
    stall = _StallDetector(STALL_PATIENCE)
    best = BestRound()
    index = 0
    while True:
        print(f"[round {index}] starting with {pst.num_prefixes} prefixes")
        started = time.monotonic()
        vs, boundary = sample_suffix_family(
            pst, pst.table.intern_suffix(b""), _Populations(pst, state)
        )
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
        best.consider(
            consistency=true_acc,
            dfa=dfa,
            tree=dt,
            boundary=pst.decision_boundary,
            round_index=index,
        )
        if true_acc >= acc_threshold:
            print(
                f"[round {index}] reached the target DFA/DT consistency of "
                f"{acc_threshold:.4f}; stopping synthesis"
            )
            return best
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
