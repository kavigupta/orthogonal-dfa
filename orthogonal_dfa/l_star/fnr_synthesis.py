"""Drive the direct-L* learner in rounds, forcing the family to resolve boundary
states.

One round learns a hypothesis from one suffix family.  When that family cannot
place a string (``sift -> None``) the string is a *boundary* string -- measured
to be, cleanly, an indecisive boundary state -- and adding it to the
representative pool makes ``sample_suffix_family`` see a high FNR over it and
re-cluster to a family that classifies it decisively, dropping the "completing"
suffixes that were diluting it.  So each round hands the next one the strings it
could not handle, and the chain resolves a state at a time.

The learner itself knows nothing about any of this; it is one way of driving it.
"""

import math
from typing import List, Optional, Set, Tuple

from automata.fa.dfa import DFA

from .cluster import sample_suffix_family
from .dfa_utils import count_paths_to_state, sample_string_reaching_state
from .direct_lstar import DirectLStarLearner
from .split_evidence import DEFAULT_SPLIT_MISS_RATE
from .structures import DecisionTree, TriPredicate


def _curated_pool(dfa, rng, length: int, per_state: int) -> List[List[int]]:
    """A state-balanced sample: up to ``per_state`` distinct length-``length``
    strings reaching *each* DFA state (via the path-counting sampler).  This is
    the representative population for the next round -- clean and balanced across
    states, rather than the accumulated sift scratch."""
    pool: List[List[int]] = []
    for state in sorted(dfa.states):
        counts = count_paths_to_state(dfa, state, length)
        reachable = counts[length][dfa.initial_state]
        if reachable == 0:
            continue
        seen = set()
        for _ in range(per_state * 5):
            if len(seen) >= min(per_state, reachable):
                break
            seen.add(tuple(sample_string_reaching_state(dfa, counts, rng)))
        pool.extend(list(s) for s in seen)
    return pool


def _take_indecisive(learner, target: int) -> List[List[int]]:
    """Up to ``target`` of the boundary strings the learner bumped into while
    building the DFA (``learner.indecisive``) -- no separate search; these arise
    naturally from transition resolution and consistency checking."""
    return [list(t) for t in list(learner.indecisive)[:target]]


def _grow_representative_pool(
    pst,
    learner,
    dfa,
    accumulated: List[List[int]],
    seen: Set,
    *,
    indecisive_fraction: float,
    min_indecisive: int,
    per_state: int,
) -> None:
    """Accumulate this round's boundary strings (capped) into ``accumulated`` /
    ``seen``, then rebuild the table's representative set as those boundary
    strings (which drive the FNR gate) plus a capped per-state balanced sample
    (bounded coverage for the consistency check)."""
    target = max(int(indecisive_fraction * pst.num_prefixes), min_indecisive)
    for t in _take_indecisive(learner, target):
        key = tuple(t)
        if key not in seen:
            seen.add(key)
            accumulated.append(t)
    curated = _curated_pool(dfa, pst.rng, pst.sampler.length, per_state)
    representative = accumulated + curated
    fresh = [p for p in representative if not pst.table.contains_prefix(p)]
    if fresh:
        pst.table.add_prefixes(fresh)
    pst.table.set_representative(representative)


class _Best:
    """The most accurate hypothesis seen so far.  Rounds are not monotone -- a
    later family can classify worse -- so the run returns its best, not its last."""

    def __init__(self):
        self.accuracy = -1.0
        self.dfa = None
        self.dt = None
        self.boundary = 0.0

    def offer(self, accuracy, dfa, dt, boundary) -> None:
        if accuracy > self.accuracy:
            self.accuracy, self.dfa, self.dt, self.boundary = (
                accuracy,
                dfa,
                dt,
                boundary,
            )


class _StallDetector:
    """Stops a run that has started repeating itself.

    Some targets are unlearnable with a fixed-length prefix sampler (transient
    states it never lands on, #128).  The FNR gate then finds no new boundary
    strings and the round is a byte-for-byte repeat of the last, so the remaining
    rounds would burn queries for nothing.  Two consecutive rounds with no new
    states, no accuracy gain and no new boundary strings confirm the fixpoint."""

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


def _discover(pst, vs, *, max_probes: int, patience: int, split_fpr, split_miss_rate):
    """One round: close the hypothesis, hunt counterexamples, close it again.

    The discovery pass samples fresh strings and splits on DFA-vs-tree
    disagreements (the equivalence-oracle role).  Its binary search homes in on
    the errors, which sit at boundary states, so it *also* harvests the boundary
    (sift -> None) strings that feed the FNR gate -- densely and targeted, so a
    probe need not end at the boundary.  This subsumes the separate consistency
    check (pool verification + boundary sweep): dropping it brings the substring
    case to E-L* query parity with no loss of convergence across seeds."""
    learner = DirectLStarLearner(
        pst, vs, split_fpr=split_fpr, split_miss_rate=split_miss_rate
    )
    learner.init_worklist()
    learner.run_worklist()
    learner.counterexample_pass(
        max_probes=max_probes, patience=patience, boundary_target=10**9
    )
    learner.run_worklist()
    dfa, dt = learner.to_dfa_and_tree()
    return learner, dfa, dt


def _estimate_accuracy(pst, dfa, dt, acc_threshold: float) -> float:
    """Agreement between the exported DFA and the tree read decisively -- the
    termination test.  The tree is re-read with both thresholds at the decision
    boundary so it answers every string rather than abstaining."""
    from .lstar import estimate_agreement_rate

    boundary = pst.decision_boundary
    decisive = dt.map_over_predicates(lambda p, b=boundary: TriPredicate(p.vs, b, b))
    return estimate_agreement_rate(
        pst,
        pst.sampler,
        pst.oracle,
        decisive,
        dfa,
        num_samples=2000,
        acc_threshold=acc_threshold,
    )


def _default_patience(acc_threshold: float) -> int:
    """Consecutive clean probes that end a discovery pass.

    If the DFA-vs-tree disagreement rate were still at the tolerated level
    ``eps = 1 - acc_threshold``, seeing k clean probes in a row has probability
    ``acc_threshold ** k``, so ``k = ceil(ln(alpha) / ln(acc_threshold))`` makes
    stopping early a ``<= alpha`` event.  This is a cost knob -- the outer
    estimate and the next round both verify -- so a modest alpha suffices (149 at
    acc_threshold 0.98, ~300 at 0.99)."""
    return math.ceil(math.log(0.05) / math.log(acc_threshold))


def synthesize_direct_lstar_fnr(
    pst,
    *,
    acc_threshold: float,
    per_state: int = 60,
    indecisive_fraction: float = 0.1,
    min_indecisive: int = 200,
    max_rounds: int = 20,
    counterexample_probes: int = 4000,
    counterexample_patience: Optional[int] = None,
    stall_patience: int = 2,
    split_fpr: Optional[float] = None,
    split_miss_rate: float = DEFAULT_SPLIT_MISS_RATE,
) -> Tuple[DFA, DecisionTree]:
    """Learn a DFA, forcing the suffix family to resolve boundary states.

    Each round, the strings the family cannot classify (``sift -> None`` --
    measured to be, cleanly, the indecisive boundary states) are added to the
    *representative* pool.  ``sample_suffix_family`` then sees a high FNR over
    them and re-clusters to a family that classifies them decisively, dropping
    the "completing" suffixes that were diluting them, so the next round can
    place them and split."""
    if counterexample_patience is None:
        counterexample_patience = _default_patience(acc_threshold)

    first_round = True
    best = _Best()
    stall = _StallDetector(stall_patience)
    # Kept across rounds: the FNR gate resolves the chain one state per round, so
    # earlier rounds' indecisives keep the family honest about the whole chain
    # (they turn decisive once their state is resolved).
    accumulated: List[List[int]] = []
    seen: Set = set()

    for round_idx in range(max_rounds):
        prior_best = best.accuracy
        vs, boundary = sample_suffix_family(
            pst, pst.table.intern_suffix([]), first_round=first_round
        )
        pst.decision_boundary = boundary
        first_round = False

        learner, dfa, dt = _discover(
            pst,
            vs,
            max_probes=counterexample_probes,
            patience=counterexample_patience,
            split_fpr=split_fpr,
            split_miss_rate=split_miss_rate,
        )
        true_acc = _estimate_accuracy(pst, dfa, dt, acc_threshold)
        best.offer(true_acc, dfa, dt, pst.decision_boundary)
        if true_acc >= acc_threshold:
            print(
                f"[direct-lstar/fnr] round {round_idx}: converged, "
                f"{learner.num_states} states"
            )
            break

        _grow_representative_pool(
            pst,
            learner,
            dfa,
            accumulated,
            seen,
            indecisive_fraction=indecisive_fraction,
            min_indecisive=min_indecisive,
            per_state=per_state,
        )
        print(
            f"[direct-lstar/fnr] round {round_idx}: {learner.num_states} states, "
            f"est {true_acc:.3f}, {len(accumulated)} accumulated indecisive, "
            f"{int(pst.table.representative.sum())} rep / {pst.num_prefixes} total"
        )
        if stall.stalled(
            states=learner.num_states,
            improved=true_acc > prior_best + 1e-9,
            boundary_strings=len(accumulated),
        ):
            print(
                f"[direct-lstar/fnr] round {round_idx}: no progress "
                f"({learner.num_states} states) -- target unresolvable with this "
                "sampler, stopping"
            )
            break

    # The structural labeling (leaves on the root's accept side) can flip a
    # low-support state under noise, especially asymmetric noise; a resample plus
    # binomial test per reachable state corrects it.  Same step the resolver
    # pipeline applies at the end.
    from .lstar import denoise_accept_labels

    pst.decision_boundary = best.boundary
    return denoise_accept_labels(pst, best.dfa), best.dt
