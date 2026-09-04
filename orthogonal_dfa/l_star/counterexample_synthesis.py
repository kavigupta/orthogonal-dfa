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

import itertools
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from automata.fa.dfa import DFA

from .cluster import sample_suffix_family
from .dfa_utils import (
    count_paths_to_state,
    sample_string_reaching_state,
    uniform_weights,
)
from .lstar import denoise_accept_labels, estimate_agreement_rate
from .midfix_tree import MidfixTree
from .progress import track
from .transition_resolver import TransitionResolver


@dataclass
class RoundClassifier:
    """One synthesis round's empty-seeded family, as it classifies that round's
    representative prefixes -- the round's attempt at the accept-preserving cut.
    ``votes[i]`` is prefix ``prefixes[i]``'s accept-rate over the family.

    The thresholds are the tracker's own, so the cut recorded here is the one
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


def _take_indecisive(resolver, target):
    """
    Take up to target of the round's boundary strings.

    The set is sorted then shuffled with a fixed rng, so the
    cap picks the same unbiased sample every run.
    """
    ordered = sorted(resolver.indecisive)
    np.random.default_rng(0).shuffle(ordered)
    return ordered[:target]


class _PoolState:
    """The pool state carried across rounds: the initial uniform sample (kept in
    the representative set every round so global calibration stays anchored to the
    sampling distribution even if the per-state sample is skewed), the accumulated
    boundary strings (with a ``seen`` set to dedup them), and last round's sample."""

    def __init__(self, uniform):
        self.uniform = list(uniform)
        self.accumulated = []
        self.seen = set()
        self.sampled = []


#: Rounds of topping a short state up before it is left short.  Each one draws
#: and sifts, so this is the give-up: a state the tree will not place strings at
#: does not get an unbounded number of tries.
TOP_UP_ROUNDS = 3


def _short_states(resolver, per_state):
    """``state -> members`` for every state, and which of them are still short.

    The members are whatever the population has already placed at that state's
    leaf, which is the only evidence that a string reaches it: the hypothesis
    says where a string *should* land, the tree says where it does.
    """
    held = {}
    for leaf in range(resolver.num_states):
        path = resolver.tree.path_of(leaf)
        if path is None:
            continue
        held[leaf] = resolver.population.members(path, per_state)
    return held, [s for s, m in held.items() if len(m) < per_state]


def _aimed_at(dfa, state, count, *, length, weights, rng):
    """``count`` strings the hypothesis says reach ``state``; empty where it says
    none do at this length."""
    counts = count_paths_to_state(dfa, state, length, uniform_weights(dfa))
    if counts[length][dfa.initial_state] == 0:
        return []
    mass = count_paths_to_state(dfa, state, length, weights)
    drawn = (
        sample_string_reaching_state(dfa, mass, rng, weights) for _ in range(count)
    )
    return [d for d in drawn if d is not None]


def _per_state_members(pst, resolver, dfa, per_state):
    """``state -> members`` up to ``per_state`` of them resting at each state,
    topped up until they are there or the tries run out, and whether every
    state got its full complement.

    Aiming a string at a state is a guess the hypothesis makes; the tree is what
    settles where it goes.  So candidates are pushed through the population and
    counted where they land, which is also what makes a state that stays short
    worth knowing about -- it is one the round cannot reach rather than one the
    sampler was unlucky about.
    """
    length = pst.sampler.length
    weights = pst.sampler.symbol_weights(pst.alphabet_size)
    held, short = _short_states(resolver, per_state)
    for attempt in range(TOP_UP_ROUNDS):
        if not short:
            break
        desc = f"Topping up short states (try {attempt + 1}/{TOP_UP_ROUNDS})"
        for leaf in track(short, desc):
            wanted = per_state - len(held[leaf])
            # Aimed first, and at the last attempt drawn plainly: a state the
            # hypothesis has the wrong shape for is one it cannot aim at.
            fresh = (
                _aimed_at(
                    dfa, leaf, wanted, length=length, weights=weights, rng=pst.rng
                )
                if attempt < TOP_UP_ROUNDS - 1
                else [
                    pst.sampler.sample(pst.rng, alphabet_size=pst.alphabet_size)
                    for _ in range(wanted)
                ]
            )
            for f in fresh:
                resolver.population.add(f)
        held, short = _short_states(resolver, per_state)
    return held, not short


def _grow_representative_pool(
    pst,
    resolver,
    dfa,
    state,
    *,
    indecisive_fraction,
    min_indecisive,
    per_state,
):
    """Rebuild the pool, returning its size and whether every state filled."""
    target = max(int(indecisive_fraction * pst.num_prefixes), min_indecisive)
    for t in _take_indecisive(resolver, target):
        if t not in state.seen:
            state.seen.add(t)
            state.accumulated.append(t)
    by_state, every_state_full = _per_state_members(pst, resolver, dfa, per_state)
    state.sampled = sorted({m for members in by_state.values() for m in members})
    # Each population is redefined here, so each is retired first: last round's
    # states are not this round's, and the prefixes a mid-round top-up bought
    # the family search are not the pool's to keep.
    for population, prefixes in (
        ("uniform", state.uniform),
        ("boundary", state.accumulated),
        ("state", state.sampled),
    ):
        pst.table.drop_population(population)
        if prefixes:
            pst.table.add_prefixes(sorted(set(prefixes)), population=population)
    return int(pst.table.representative.sum()), every_state_full


def tree_is_saturated(resolver, every_state_full) -> bool:
    """Whether this round's prefixes had nothing left to say.

    A state the round could not fill is one more prefixes would say more about.
    Past that every node has to come out settled (see `Decisions`), each on its
    own evidence, so that one node still straddling its midfix keeps the round
    open however clean the rest are.
    """
    return every_state_full and resolver.decisions.every_node_settled()


#: Consecutive rounds with no progress. See `_StallDetector` for more details.
STALL_PATIENCE = 2


class _StallDetector:
    """Stops a run that has started repeating itself. We consider a round stalled if

    1. There are no new states
    2. (Internal) accuracy has not increased
    3. The tree is saturated (see `tree_is_saturated`)

    This catches a situation where the fixed-length probes can't find any information
    about transient states.

    Deliberately fairly restrictive, so we can have a low Patience before
    exiting the loop.
    """

    def __init__(self, patience: int):
        self._patience = patience
        self._states = 0
        self._stalled = 0

    def stalled(self, *, states: int, improved: bool, saturated: bool) -> bool:
        progressed = states > self._states or improved or not saturated
        self._stalled = 0 if progressed else self._stalled + 1
        self._states = states
        return self._stalled >= self._patience


#: Target number of representative strings per DFA state.  Each round tops the
#: sampled pool up to this per state (see ``per_state_sample``); states the
#: original prefixes already cover need no top-up, so the pool converges rather
#: than growing every round.
PER_STATE = 20


def counterexample_driven_synthesis(
    pst,
    *,
    acc_threshold: float,
    per_state: int = PER_STATE,
    indecisive_fraction: float = 0.1,
    min_indecisive: int = 200,
):
    patience = _default_patience(acc_threshold)
    # Kept across rounds: the FNR gate resolves the chain one state per round, so
    # earlier rounds' boundary strings keep the family honest about the whole
    # chain (they turn decisive once their state is resolved).
    uniform = [
        p for p, keep in zip(pst.table.prefixes, pst.table.representative) if keep
    ]
    state = _PoolState(uniform)
    stall = _StallDetector(STALL_PATIENCE)
    best_acc = -1.0
    for index in itertools.count():
        print(f"[round {index}] starting with {pst.num_prefixes} prefixes")
        started = time.monotonic()
        vs, boundary = sample_suffix_family(pst, pst.table.intern_suffix(b""))
        pst.decision_boundary = boundary
        classifier = _round_classifier(pst, vs)
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
        if true_acc >= acc_threshold:
            print(
                f"[round {index}] reached the target DFA/DT consistency of "
                f"{acc_threshold:.4f}; stopping synthesis"
            )
            yield dfa, dt, true_acc, pst.decision_boundary, classifier
            return
        pool, every_state_full = _grow_representative_pool(
            pst,
            resolver,
            dfa,
            state,
            indecisive_fraction=indecisive_fraction,
            min_indecisive=min_indecisive,
            per_state=per_state,
        )
        print(
            f"[round {index}] pool now {pool} representative prefixes, "
            f"{len(state.accumulated)} boundary strings harvested so far"
        )
        improved = true_acc > best_acc
        best_acc = max(best_acc, true_acc)
        if stall.stalled(
            states=dt.num_states,
            improved=improved,
            saturated=tree_is_saturated(resolver, every_state_full),
        ):
            print(
                f"[round {index}] no progress ({dt.num_states} states) in "
                f"{STALL_PATIENCE} rounds -- pool churning without resolving; "
                "stopping synthesis"
            )
            yield dfa, dt, true_acc, pst.decision_boundary, classifier
            return
        yield dfa, dt, true_acc, pst.decision_boundary, classifier


def do_counterexample_driven_synthesis(
    pst, *, acc_threshold: float
) -> Tuple[Optional[DFA], Optional[MidfixTree], List[RoundClassifier]]:
    # Rounds are not monotone -- rebuilding the representative pool re-clusters,
    # so a later family can classify worse -- so keep the most accurate
    # hypothesis, not the last. The boundary is kept with it because denoising
    # reads the tree against it.
    best_acc, best_dfa, best_dt, best_boundary = -1.0, None, None, None
    classifiers = []
    for dfa, dt, true_acc, boundary, classifier in counterexample_driven_synthesis(
        pst, acc_threshold=acc_threshold
    ):
        classifiers.append(classifier)
        if true_acc > best_acc:
            best_acc, best_dfa, best_dt, best_boundary = true_acc, dfa, dt, boundary
    dfa, dt = best_dfa, best_dt
    if dfa is not None:
        pst.decision_boundary = best_boundary
        dfa = denoise_accept_labels(pst, dfa)
    return dfa, dt, classifiers
