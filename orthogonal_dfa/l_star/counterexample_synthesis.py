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
from typing import Dict, List, Optional

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


#: Prefixes a population is asked for.
WANTED = 100


class Pools:
    """The prefix populations, a source apiece, which a round hands the next one.

    What a later round needs more of it draws more of; a population nothing
    draws for any more is retired.
    """

    def __init__(self, pst):
        self._pst = pst
        #: Boundary pools sealed so far, which is what names them.
        self._sealed = 0
        #: This round's unplaceable strings, in the order they arrived.
        self._harvest: Dict[bytes, None] = {}
        #: Every string some pool holds, so no two pools hold the same one.
        self._pooled: set = set()
        self._sources = {UNIFORM: UniformSource(pst)}
        #: One per round that produced any: the strings that round could not
        #: place, and the source that can find that round more of them.
        self._boundaries = {}
        self._boundary_sources = {}
        #: This round's populations, ``label -> prefixes``.
        self.held = {}
        #: Labels the table holds, so a round can retire what it does not renew.
        self._published: set = set()

    def offer_indecisive(self, string) -> None:
        """Hold ``string`` for the pool this round will make, unless an earlier
        round already made one of it."""
        if string not in self._pooled:
            self._harvest[string] = None

    def rebuild(self, resolver, dfa) -> None:
        """Take the sources this round defines, and fill what they can fill.

        Each round's unplaceable strings stay a population of their own: merged,
        the ones an earlier round settled would be spent on whichever state was
        worst last round, and could come undone.
        """
        # Sorted: a set of bytes iterates in hash order, which python varies per
        # process, and which of these reach a boundary population decides what
        # the next family is made to resolve.
        for string in sorted(resolver.indecisive):
            self.offer_indecisive(string)

        self._sources = {UNIFORM: UniformSource(self._pst)}
        states = []
        for leaf in track(range(resolver.num_states), "Drawing each state's prefixes"):
            aim = aim_at(self._pst, dfa, leaf)
            if aim is None:
                # Out of reach rather than short: no string of the sampler's
                # length arrives here, so the round is not waiting on a draw.
                continue
            source = state_source(resolver, leaf, aim, wanted=WANTED)
            if source is None:
                continue
            self._sources[("state", leaf)] = source
            states.append(("state", leaf))
        # No population for the uniform source: the table keeps its pool across
        # rounds rather than remaking it at this size every round.
        collected = {}
        for label in states:
            collected[label] = draw_many(self._sources[label], WANTED)
        # After the state sources: an aimed draw is one of the places a string
        # turns out to be unplaceable, and those belong to this round's pool.
        self.pool_the_harvest(resolver, dfa)
        self._sources.update(self._boundary_sources)
        self.held = dict(self._boundaries)
        self.held.update(collected)
        self.publish()

    def pool_the_harvest(self, resolver, dfa) -> None:
        """Make a pool of this round's harvest, and a source to grow it with.

        Whether that source still finds anything is asked when something asks it
        to draw: these strings are ones a round could not place however few more
        there are to find.
        """
        if not self._harvest:
            return
        self._sealed += 1
        label = ("boundary", self._sealed)
        self._boundary_sources[label] = BoundarySource(
            self._pst,
            resolver.sifter,
            dfa.transitions,
            known=self._pooled | self._harvest.keys(),
        )
        self._boundaries[label] = list(self._harvest)
        self._pooled.update(self._harvest)
        self._harvest = {}

    @property
    def boundary_strings(self) -> int:
        """Unplaceable strings held, pooled or still buffering."""
        return sum(len(pool) for pool in self._boundaries.values()) + len(self._harvest)

    def labels(self) -> list:
        """The populations a family is read over: this round's, and the uniform
        pool the table keeps across rounds."""
        return [UNIFORM, *self.held]

    def for_split(self, label, wanted: int):
        """Prefixes for one population, to read the split on and not to keep.

        Empty where nothing draws for it any more: no say in the split rather
        than a split held up.  `more` is what retires it.
        """
        source = self._sources.get(label)
        if source is None or not source.worth_drawing():
            return []
        return draw_many(source, wanted)

    def more(self, label) -> bool:
        """Draw more prefixes for one population, saying whether it could.

        A population nothing draws for any more is retired here, table and all:
        a rate the round cannot answer is not one to hold a family to.
        """
        source = self._sources.get(label)
        if source is None or not source.worth_drawing():
            self.held.pop(label, None)
            # Forgotten, not held aside, so a later round that strands one of
            # these again can pool it behind a source that does draw.
            self._pooled.difference_update(self._boundaries.pop(label, ()))
            self._pst.table.drop_population(label)
            return False
        drawn = draw_many(source, WANTED)
        # Extended, not rebound: a boundary pool's list is the one `_boundaries`
        # holds, which is what the next round republishes it from.  And a
        # population this round did not define is not one it retires either, so
        # the uniform pool grows without joining what `publish` resets.
        if label in self.held:
            self.held[label].extend(drawn)
        self._pst.table.add_prefixes(sorted(set(drawn)), population=label)
        return True

    def publish(self) -> None:
        """Install this round's populations, retiring the ones it replaces."""
        for label in self._published - self.held.keys():
            self._pst.table.drop_population(label)
        for label, prefixes in self.held.items():
            self._pst.table.drop_population(label)
            if prefixes:
                self._pst.table.add_prefixes(sorted(set(prefixes)), population=label)
        self._published = set(self.held)


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
) -> BestRound:
    """Rounds until the hypothesis is consistent enough, the pool stalls, or
    ``max_rounds`` of them have run.  Only a caller driving the loop itself can
    set that cap; `learn_dfa` does not forward one."""
    # The cap is read at the foot of the body, so a round always runs.
    assert max_rounds is None or max_rounds >= 1, max_rounds
    patience = _default_patience(acc_threshold)
    pools = Pools(pst)
    stall = _StallDetector(STALL_PATIENCE)
    best = BestRound()
    index = 0
    while True:
        print(f"[round {index}] starting with {pst.num_prefixes} prefixes")
        started = time.monotonic()
        vs, boundary = sample_suffix_family(pst, pst.table.intern_suffix(b""), pools)
        pst.decision_boundary = boundary
        tracker.on_family_resolved([pst.table.suffix(i) for i in vs], boundary, index)
        tracker.on_round_classified(_round_classifier(pst, vs), index)
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
        pools.rebuild(resolver, dfa)
        print(
            f"[round {index}] pool now {pst.num_prefixes} prefixes over "
            f"{len(pools.held)} populations, {pools.boundary_strings} boundary "
            f"strings harvested so far"
        )
        # Asked after the rebuild, whose aims are what fill the leaves it
        # reads.
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
        # The check strands strings of its own reading every leaf; they belong
        # to the harvest rather than dying with this round's resolver.
        for string in sorted(resolver.indecisive):
            pools.offer_indecisive(string)
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
