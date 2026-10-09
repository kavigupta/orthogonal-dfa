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
import warnings
from dataclasses import dataclass
from functools import partial
from typing import List, Optional

import numpy as np
from automata.fa.dfa import DFA

from .certificate import certifies, look_level
from .cluster import sample_suffix_family
from .lstar import denoise_accept_labels
from .mask_table import UNIFORM
from .midfix_tree import MidfixTree
from .preconditions import start_length
from .prefix_populations import PoolState
from .prefix_sources import HarvestSource, MidfixSource, aim_at, state_source
from .progress import track
from .tracker import SynthesisTracker
from .transition_resolver import MIN_PROBES, PAIR_TRIP, STOPPED, TransitionResolver


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


def _default_patience(acc_threshold: float) -> int:
    """Consecutive clean probes that end a counterexample pass: seeing this many
    in a row is a ``<= 0.05`` event if the disagreement rate were still at the
    tolerated ``1 - acc_threshold``.

    A perfect-accuracy target tolerates no disagreement, so no finite clean run
    rules it out -- wait out ``MIN_PROBES`` of them."""
    if acc_threshold >= 1:
        return MIN_PROBES
    return math.ceil(math.log(0.05) / math.log(acc_threshold))


def _accumulate_indecisive(resolver, state, wanted) -> int:
    """Take up to ``wanted`` of the round's boundary strings ``state`` does not
    already hold, returning how many.

    Sorted then shuffled with a fixed rng, so the cap picks the same unbiased
    sample every run.
    """
    taken = sorted(set(resolver.indecisive) - state.seen)
    np.random.default_rng(0).shuffle(taken)
    for string in taken[:wanted]:
        state.take(string)
    return min(wanted, len(taken))


def _per_state_members(pst, resolver, dfa, state, per_state) -> None:
    """``("state", leaf) -> members``, ``per_state`` of them resting at each
    state that has a source."""
    state.retire("state")
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
        state.hold(("state", leaf), source, per_state)


def _read_round(resolver, certificate, *, patience, acc_threshold, index):
    """The pass, then the gate on the hypothesis it leaves, then the certificate
    on one the gate passes.  A refusal by either is sampled (see
    ``refusal_sample``) and goes back to the pass with the edges its sample's
    searches ended at, but those given up on, as its first probes, until the
    certificate passes, a refusal sample ends at no such edge, or the round's
    probes run out.  Returns the last reading, its DFA, and the certified DFA if
    any."""
    first = []
    while True:
        resolver.counterexample_pass(patience=patience, first=first)
        gate = resolver.read_fresh(acc_threshold=acc_threshold)
        dfa = resolver.to_dfa_and_tree(gate.start)[0]
        if gate.passed:
            output = certificate.certify(dfa, index=index)
            if output is not None:
                return gate, dfa, output
            gate = resolver.refusal_sample(gate)
        if not gate.disagreements or resolver.probed >= resolver.probe_budget(patience):
            return gate, dfa, None
        first = gate.disagreements


def _hold_ends(pst, state, ends, count) -> None:
    """``(end, m)`` -> ``count`` of the sampler's draws, cut to the start length
    for ``"start"`` or whole for ``"end"``, followed by the midfix m: what the
    next family reads at node m when it sifts a walk's start or a probe whole."""
    lengths = {"start": start_length(pst.sampler.length), "end": pst.sampler.length}
    for end in lengths:
        state.retire(end)
    for end, midfix in ends:
        state.hold((end, midfix), MidfixSource(pst, lengths[end], midfix), count)


def _hold_harvests(pst, resolver, gate, state, *, per_state, acc_threshold):
    """Hold what the gate's refusal sample's outcomes and the pass's stopped
    guards left, a population per kind grown by replaying that reading, and the
    start and end populations at the midfixes the sample's ends stopped at."""
    _hold_ends(pst, state, gate.ends, per_state)
    for kind, found in {**gate.harvests, STOPPED: resolver.stopped}.items():
        if found:
            source = HarvestSource(
                partial(resolver.replay, gate, kind),
                known=state.seen,
                acc_threshold=acc_threshold,
            )
            state.hold_found(kind, found, source)


def _halve(pst, gate) -> bool:
    """Halve the FNR limit on a refusal whose sample came down to pairs too
    often, or read through holding nothing and meeting no edge to rerun.  Says
    whether it halved."""
    halve = gate.fired is not None and (
        PAIR_TRIP in gate.fired or not (gate.fired or gate.disagreements)
    )
    if halve:
        pst.fnr_limit /= 2
    return halve


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
    state.close_harvest()
    return int(pst.table.representative.sum())


#: Consecutive rounds with no progress. See `_StallDetector` for more details.
STALL_PATIENCE = 2

#: Rounds a run keeps going after the certificate first refuses a round at the
#: consistency target.
CERTIFICATE_PATIENCE = 5


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


class _Certificate:
    """The certificate, its alpha spread over every attempt in the run."""

    def __init__(self, pst, tracker):
        self.pst, self.tracker = pst, tracker
        self.attempts = 0
        #: Round of the first attempt; CERTIFICATE_PATIENCE counts from it.
        self.first_round = None

    def certify(self, dfa, *, index):
        """denoise_accept_labels(dfa) if the certificate passes it, else None."""
        if self.first_round is None:
            self.first_round = index
        output = denoise_accept_labels(self.pst, dfa)
        alpha = look_level(CERTIFICATE_ALPHA, self.attempts)
        self.attempts += 1
        verdict = certifies(self.pst, output, alpha=alpha)
        self.tracker.on_certificate_decided(verdict.certified, index)
        if verdict.certified:
            print(f"[round {index}] certified; stopping synthesis")
            return output
        print(f"[round {index}] at target, not certified; blames {verdict.blamed}")
        return None


def _uncertified_too_long(index, uncertified_since) -> bool:
    if uncertified_since is None or index - uncertified_since < CERTIFICATE_PATIENCE:
        return False
    print(
        f"[round {index}] no hypothesis the certificate passes in "
        f"{CERTIFICATE_PATIENCE} "
        "rounds since the certificate first failed; stopping synthesis"
    )
    return True


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
    """Rounds until a hypothesis is certified, the pool stalls, the rounds since
    the first refusal run out of patience, or max_rounds of them have run.
    Only a caller driving the loop itself can set that cap; `learn_dfa` does not
    forward one."""
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
    certificate = _Certificate(pst, tracker)
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
        gate, dfa, output = _read_round(
            resolver,
            certificate,
            patience=patience,
            acc_threshold=acc_threshold,
            index=index,
        )
        true_acc = gate.agreement
        dt = resolver.tree
        print(
            f"[round {index}] resolved {dt.num_states} states over a family of "
            f"{len(vs)} suffixes ({sampled - started:.1f}s sampling, "
            f"{time.monotonic() - sampled:.1f}s resolving)"
        )
        assert dt.num_states >= 2
        tracker.on_initial_dfa_found(dfa, dt, index)
        print(dfa)
        print(f"[round {index}] DFA/DT consistency on fresh samples: {true_acc:.4f}")
        tracker.on_consistency_estimated(true_acc, index)
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
        if _uncertified_too_long(index, certificate.first_round):
            return best
        if _halve(pst, gate):
            print(f"[round {index}] FNR limit now {pst.fnr_limit:.4f}")
        _hold_harvests(
            pst, resolver, gate, state, per_state=per_state, acc_threshold=acc_threshold
        )
        target = max(int(indecisive_fraction * pst.num_prefixes), min_indecisive)
        taken = _accumulate_indecisive(resolver, state, target)
        _per_state_members(pst, resolver, dfa, state, per_state)
        # Asked after the aims, which are what fill the leaves it reads.  A
        # leaf nothing aims at is not one the round waits on.  Rounds after a
        # refusal have their own patience, so they are not weighed for a stall.
        if certificate.first_round is None and stall.stalled(
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
        _accumulate_indecisive(resolver, state, target - taken)
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
    dfa = best.certified
    if dfa is None:
        warnings.warn(
            f"no round was certified; returning round {best.round_index}'s DFA",
            UncertifiedResult,
        )
        dfa = denoise_accept_labels(pst, best.dfa)
    tracker.on_corrected_dfa_found(dfa, best.round_index)
    return dfa
