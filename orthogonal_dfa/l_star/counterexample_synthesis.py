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
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from automata.fa.dfa import DFA

from .cluster import limit_is_expressible, sample_suffix_family
from .lstar import denoise_accept_labels, estimate_agreement_rate
from .midfix_tree import MidfixTree
from .prefix_sources import (
    WANTED,
    IndecisiveSource,
    StateSource,
    UniformSource,
    collect,
    draw_for_split,
)
from .statistics import binomial_side_of_boundary
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


def classify_pool(pst, tree, *, accept, reject):
    """
    Classify every prefix in the pool to its leaf (or -1 if undecided), from
    the cached mask matrix. Uses accept and reject thresholds.
    """

    def decide_columns(midfix):
        decision = pst.compute_decision_from_strings(tree.suffixes(midfix))
        return decision >= accept, decision < reject

    return tree.classify_pool(pst.num_prefixes, decide_columns)


def _take_indecisive(resolver, target):
    """
    Take up to target of the round's boundary strings.

    The set is sorted then shuffled with a fixed rng, so the
    cap picks the same unbiased sample every run.
    """
    ordered = sorted(resolver.indecisive)
    np.random.default_rng(0).shuffle(ordered)
    return ordered[:target]


class Pools:
    """The prefix populations, as a source apiece rather than a list apiece.

    A round hands the next one these.  What a later round needs more of it draws
    more of -- the gate wanting enough prefixes to read a rate over is a draw and
    not a special case -- and a source that cannot fill its population has that
    population dropped, which is what there is to say about a state nothing
    reaches.
    """

    def __init__(self, pst):
        self._pst = pst
        #: Boundary pools sealed so far, which is what names them.
        self._sealed = 0
        #: What this round could not place, offered to the pool it will become.
        self._harvest = IndecisiveSource()
        self._sources = {"baseline": UniformSource(pst)}
        #: One per round that produced any: the strings that round could not
        #: place, kept as they were.
        self._boundaries = {}
        self.held = {}

    def offer_indecisive(self, string) -> None:
        self._harvest.offer(string)

    def rebuild(self, resolver, dfa) -> None:
        """Take the sources this round defines, and fill what they can fill.

        Each round's unplaceable strings become a population of their own and
        stay one.  Merged into a single pool they would be whichever state was
        worst last round and no other, so a family made decisive on that state
        could let the states an earlier round settled come undone -- the pool
        that settled them having been spent.  Kept apart, every one of them goes
        on being a rate the family has to meet.
        """
        # Sorted: a set of bytes iterates in hash order, which python varies per
        # process, and which of these reach a boundary population decides what
        # the next family is made to resolve.
        for string in sorted(resolver.indecisive):
            self._harvest.offer(string)
        self.seal_ready_harvest()

        self._sources = {"baseline": UniformSource(self._pst)}
        for leaf in range(resolver.num_states):
            source = StateSource(
                self._pst, resolver, dfa, leaf, sink=self._harvest.offer
            )
            self._sources[source.label] = source
        self.held = dict(self._boundaries)
        for label, source in self._sources.items():
            got = collect(source)
            if got is not None:
                self.held[label] = got
        self.publish()

    def seal_ready_harvest(self) -> None:
        """Make a pool of the buffered harvest, once there is enough of it.

        Under ``1 / fnr_limit`` strings the gate could admit the pool only with
        nothing straddling at all, which is a bar no sampling clears and which
        nothing can grow it past -- a sealed pool has no source.  So they keep
        instead, and buffering across rounds is what the harvest is a source for.
        """
        if limit_is_expressible(len(self._harvest), self._pst.config.fnr_limit):
            self._sealed += 1
            self._boundaries[("boundary", self._sealed)] = self._harvest.take()

    @property
    def sealed_pools(self) -> int:
        """Boundary pools made so far, each a round's worth of harvest."""
        return len(self._boundaries)

    @property
    def boundary_strings(self) -> int:
        """Unplaceable strings held, sealed into pools or still buffering.

        Counts both because the stall detector asks whether the round harvested
        anything, and a round whose harvest was too small to seal still found
        something.
        """
        return sum(len(pool) for pool in self._boundaries.values()) + len(self._harvest)

    def for_split(self, label, wanted: int):
        """Prefixes for one population, to read the split on and not to keep."""
        source = self._sources.get(label)
        return draw_for_split(source, wanted) if source is not None else []

    def more(self, label, wanted: int) -> bool:
        """Draw ``wanted`` further prefixes for one population.  Says whether it
        could: a source that has stopped delivering ends its population.

        A sealed pool of unplaceable strings has no source and never grows -- it
        is the strings a round could not place, all of them, and asking for more
        is asking the wrong question.  Saying so is what stops the caller asking
        again."""
        source = self._sources.get(label)
        if source is None:
            return False
        drawn = collect(source, wanted=wanted)
        if drawn is None:
            self.held.pop(label, None)
            self.publish()
            return False
        self.held[label] = self.held.get(label, []) + drawn
        self.publish()
        return True

    def publish(self) -> None:
        """Install the populations as the representative set."""
        representative, strata = [], []
        for label, prefixes in self.held.items():
            representative.extend(prefixes)
            strata.extend([label] * len(prefixes))
        fresh = sorted(
            {p for p in representative if not self._pst.table.contains_prefix(p)}
        )
        if fresh:
            self._pst.table.add_prefixes(fresh)
        self._pst.table.set_representative(representative, strata)


class _PoolAccess:
    """What the next round's family search may ask of this round's populations:
    more prefixes for one of them, or prefixes to read the split on."""

    def __init__(self, pools):
        self._pools = pools

    def __call__(self, label):
        return self._pools.more(label, WANTED)

    def for_split(self, label, wanted):
        return self._pools.for_split(label, wanted)


def uncoverable_access_strings(pst, tree):
    """Access strings the hypothesis cannot resolve and can never be covered.

    The short prefix-closed core is the set of access strings, it reaches
    every state, including transient ones that a fixed-length prefix sampler
    never lands on.

    We can use this to detect when the underlying DFA is not learnable in
    our model. Specifically, when a state in the access strings is not also
    reached by any representative (longer) prefix. This prevents us from
    averaging across multiple prefixes to get a representative set for this state
    implying that the state is only reached by a small number of strings
    overall.
    """
    prefixes = list(pst.table.prefixes)
    # Coverage is measured against the stable non-core (sampled) prefixes, not the
    # representative set, the driver re-scopes representative to focus clustering,
    # which must not narrow what counts as "covered".
    sampled = pst.table.noncore
    fam = pst.table.fully_observed()
    if len(fam) == 0 or not sampled.any():
        return []

    eta = 0.5 - pst.config.min_signal_strength
    # Two prefixes at the same state agree on every suffix up to independent
    # per-cell noise, so their expected mask-disagreement rate is 2*eta*(1-eta).
    same_state_rate = 2 * eta * (1 - eta)
    n = len(fam)

    repr_masks = pst.table.observed_masks(fam, sampled).T  # [n_sampled, n_fam]
    leaves = classify_pool(
        pst, tree, accept=pst.accept_thresh, reject=pst.reject_thresh
    )
    potentially_problematic = np.flatnonzero(
        (~sampled) & (leaves == -1)
    )  # only unclassifiable core prefixes
    flagged = []
    for i in potentially_problematic:
        col = np.zeros(len(prefixes), dtype=bool)
        col[i] = True
        mask_i = pst.table.observed_masks(fam, col).T[0]
        # get the nearest and see if it's too far away to be a sibling.  If so, this prefix is problematic.
        nearest = int((repr_masks != mask_i).sum(1).min())
        if binomial_side_of_boundary(nearest, n, same_state_rate, failure_prob=0.01):
            flagged.append((prefixes[i], nearest / n))
    return flagged


#: Consecutive rounds with no progress. See `_StallDetector` for more details.
STALL_PATIENCE = 2


class _StallDetector:
    """Stops a run that has started repeating itself. We consider a round stalled if

    1. There are no new states
    2. (Internal) accuracy has not increased
    3. No new boundary strings have been harvested

    This catches a situation where the fixed-length probes can't find any information
    about transient states.

    Deliberately fairly restrictive, so we can have a low Patience before
    exiting the loop.
    """

    def __init__(self, patience: int):
        self._patience = patience
        self._states = 0
        self._boundary_strings = 0
        self._stalled = 0

    def stalled(self, *, states: int, improved: bool, boundary_strings: int) -> bool:
        progressed = (
            states > self._states
            or improved
            or boundary_strings > self._boundary_strings
        )
        self._stalled = 0 if progressed else self._stalled + 1
        self._states, self._boundary_strings = states, boundary_strings
        return self._stalled >= self._patience


def counterexample_driven_synthesis(pst, *, acc_threshold: float):
    patience = _default_patience(acc_threshold)
    pools = Pools(pst)
    stall = _StallDetector(STALL_PATIENCE)
    best_acc = -1.0
    # No previous round, so the first family search has no populations to reach
    # back for and answers everything by drawing uniformly.
    grow = None
    while True:
        print(f"Starting synthesis iteration with {pst.num_prefixes} prefixes")
        vs, boundary = sample_suffix_family(pst, pst.table.intern_suffix(b""), grow)
        pst.decision_boundary = boundary
        classifier = _round_classifier(pst, vs)
        resolver = TransitionResolver(pst, vs)
        resolver.close_edges()
        resolver.counterexample_pass(
            max_probes=COUNTEREXAMPLE_PROBES, patience=patience
        )
        dfa, dt = resolver.to_dfa_and_tree()
        print(f"Resolved DFA with {dt.num_states} states")
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
        print(f"Estimated DFA accuracy on fresh samples: {true_acc:.4f}")
        if true_acc >= acc_threshold:
            print(f"Achieved desired accuracy of {acc_threshold}; stopping synthesis")
            yield dfa, dt, true_acc, pst.decision_boundary, classifier
            return
        uncoverable = uncoverable_access_strings(pst, dt)
        if uncoverable:
            examples = ", ".join(
                "".join(map(str, p)) or "eps" for p, _ in uncoverable[:5]
            )
            print(
                f"Stopping synthesis: {len(uncoverable)} access string(s) reach "
                f"states no sampled prefix can cover at length "
                f"{pst.sampler.length} (e.g. {examples}); the target is not "
                f"learnable with this prefix sampler."
            )
            yield dfa, dt, true_acc, pst.decision_boundary, classifier
            return
        pools.rebuild(resolver, dfa)
        # Handed to the next round's family search, which is the thing that finds
        # out a population needs more prefixes than it holds.
        grow = _PoolAccess(pools)
        improved = true_acc > best_acc
        best_acc = max(best_acc, true_acc)
        if stall.stalled(
            states=dt.num_states,
            improved=improved,
            boundary_strings=pools.boundary_strings,
        ):
            print(
                f"No progress ({dt.num_states} states) in {STALL_PATIENCE} rounds "
                "-- pool churning without resolving; stopping synthesis"
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
