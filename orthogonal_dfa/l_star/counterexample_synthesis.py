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
import os
import pickle
from typing import List, Optional, Set, Tuple

import numpy as np
from automata.fa.dfa import DFA

from .cluster import sample_suffix_family
from .dfa_utils import per_state_sample
from .transition_resolver import TransitionResolver
from .midfix_tree import MidfixTree
from .statistics import binomial_side_of_boundary


def _dump_round(dump_dir, *, round_idx, dfa, dt, est, decision_boundary):
    """Opt-in per-round snapshot (active only when ``DLSTAR_DUMP_DIR`` is set), for
    offline round-by-round analysis -- e.g. measuring phi(round DFA, oracle) to see
    the shatter -> collapse trajectory.  Dumps only picklable structure: the class
    DFA and discrimination tree (midfixes are int-lists) plus the decision boundary,
    NOT the learner (it holds pst -> oracle -> the SpliceAI model and the membership
    cache, which are huge / unpicklable).  A dump failure never kills the run."""
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"round_{round_idx:02d}.pkl")
    state = dict(
        round=round_idx,
        est=est,
        num_states=len(dfa.states),
        dfa=dfa,
        dt=dt,
        tree=dt,
        final_states=sorted(dfa.final_states),
        decision_boundary=decision_boundary,
    )
    try:
        with open(path, "wb") as f:
            pickle.dump(state, f)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # never let a dump failure kill the run
        with open(path, "wb") as f:
            pickle.dump(
                dict(round=round_idx, est=est, num_states=len(dfa.states),
                     dump_error=repr(exc)),
                f,
            )
        print(f"[dump] round {round_idx}: full-state dump failed ({exc!r}); "
              "wrote metadata-only fallback", flush=True)


def classify_pool(pst, tree, *, accept, reject):
    """
    Classify every prefix in the pool to its leaf (or -1 if undecided), from
    the cached mask matrix. Uses accept and reject thresholds.
    """

    def decide_columns(midfix):
        decision = pst.compute_decision_from_strings(tree.suffixes(midfix))
        return decision >= accept, decision < reject

    return tree.classify_pool(pst.num_prefixes, decide_columns)


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
            flagged.append((list(prefixes[i]), nearest / n))
    return flagged


def _take_indecisive(learner, target: int) -> List[List[int]]:
    """Up to ``target`` of the boundary strings the learner bumped into while
    building the DFA.  No separate search for them: resolving an edge sifts
    ``member + symbol``, and the ones the family cannot place are exactly these.

    The set is sorted then shuffled with a fixed rng, so the cap picks the same
    unbiased sample every run rather than an arbitrary iteration-order slice."""
    ordered = sorted(tuple(b) for b in learner.indecisive)
    np.random.default_rng(0).shuffle(ordered)
    return [list(t) for t in ordered[:target]]


class _PoolState:
    """The pool state carried across rounds: the initial uniform sample (kept in
    the representative set every round so global calibration stays anchored to the
    sampling distribution even if the per-state sample is skewed), the accumulated
    boundary strings (with a ``seen`` set to dedup them), and last round's sample."""

    def __init__(self, baseline):
        self.baseline = [list(p) for p in baseline]
        self.accumulated: List[List[int]] = []
        self.seen: Set = set()
        self.sampled: List[List[int]] = []


def _grow_representative_pool(
    pst,
    learner,
    dfa,
    state,
    *,
    indecisive_fraction: float,
    min_indecisive: int,
    per_state: int,
) -> None:
    """Accumulate this round's boundary strings (capped) into ``state``, then
    rebuild the table's representative set as those boundary strings -- which
    drive the FNR gate -- plus a capped per-state balanced sample, so the
    population stays spread across the states."""
    target = max(int(indecisive_fraction * pst.num_prefixes), min_indecisive)
    for t in _take_indecisive(learner, target):
        key = tuple(t)
        if key not in state.seen:
            state.seen.add(key)
            state.accumulated.append(t)
    # Feed last round's sample back in so per_state_sample tops each state up to
    # per_state rather than adding a fresh per_state every round; the sample then
    # converges to per_state-per-state instead of building up.
    state.sampled = per_state_sample(
        dfa, pst.rng, pst.sampler.length, per_state, existing=state.sampled
    )
    representative = state.baseline + state.accumulated + state.sampled
    fresh = [
        list(p)
        for p in sorted(
            set(tuple(p) for p in representative if not pst.table.contains_prefix(p))
        )
    ]
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

    A fixed-length prefix sampler cannot reach every target's transient states,
    and when it cannot, the FNR gate finds no new boundary strings and the round
    repeats the last one exactly.  Two consecutive rounds with no new states, no
    accuracy gain and no new boundary strings confirm that fixpoint, so the
    remaining rounds are not spent on it."""

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


def _discover(pst, vs, *, max_probes: int, patience: int):
    """One round: close the hypothesis, hunt counterexamples, close it again.

    The discovery pass samples fresh strings and splits on DFA-vs-tree
    disagreements (the equivalence-oracle role).  Its binary search homes in on
    the errors, which sit at boundary states, so it *also* harvests the boundary
    (sift -> None) strings that feed the FNR gate -- densely and targeted, so a
    probe need not end at the boundary.  One pass therefore both finds the splits
    and gathers what the next round's family must resolve."""
    learner = TransitionResolver(pst, vs)
    learner.close_edges()
    learner.counterexample_pass(max_probes=max_probes, patience=patience)
    learner.close_edges()
    dfa, dt = learner.to_dfa_and_tree()
    return learner, dfa, dt


def _estimate_accuracy(pst, dfa, tree, acc_threshold: float) -> float:
    """Agreement between the exported DFA and the tree read decisively -- the
    termination test.  ``estimate_agreement_rate`` re-reads the tree with both
    thresholds at the decision boundary so it answers every string rather than
    abstaining."""
    from .lstar import estimate_agreement_rate

    return estimate_agreement_rate(
        pst,
        pst.sampler,
        pst.oracle,
        tree,
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
    per_state: int = 20,
    indecisive_fraction: float = 0.1,
    min_indecisive: int = 200,
    max_rounds: int = 20,
    counterexample_probes: int = 4000,
    counterexample_patience: Optional[int] = None,
    stall_patience: int = 2,
) -> Tuple[DFA, MidfixTree]:
    """Learn a DFA, forcing the suffix family to resolve boundary states.

    Each round, the strings the family cannot classify (``sift -> None`` --
    measured to be, cleanly, the indecisive boundary states) are added to the
    *representative* pool.  ``sample_suffix_family`` then sees a high FNR over
    them and re-clusters to a family that classifies them decisively, dropping
    the "completing" suffixes that were diluting them, so the next round can
    place them and split."""
    if counterexample_patience is None:
        counterexample_patience = _default_patience(acc_threshold)

    best = _Best()
    stall = _StallDetector(stall_patience)
    # Kept across rounds: the FNR gate resolves the chain one state per round, so
    # earlier rounds' indecisives keep the family honest about the whole chain
    # (they turn decisive once their state is resolved).
    baseline = [
        p for p, keep in zip(pst.table.prefixes, pst.table.representative) if keep
    ]
    state = _PoolState(baseline)

    for round_idx in range(max_rounds):
        prior_best = best.accuracy
        vs, boundary = sample_suffix_family(pst, pst.table.intern_suffix([]))
        pst.decision_boundary = boundary

        learner, dfa, dt = _discover(
            pst,
            vs,
            max_probes=counterexample_probes,
            patience=counterexample_patience,
        )
        true_acc = _estimate_accuracy(pst, dfa, dt, acc_threshold)
        best.offer(true_acc, dfa, dt, pst.decision_boundary)
        _dump_dir = os.environ.get("DLSTAR_DUMP_DIR")
        if _dump_dir:
            _dump_round(
                _dump_dir,
                round_idx=round_idx,
                dfa=dfa,
                dt=dt,
                est=true_acc,
                decision_boundary=pst.decision_boundary,
            )
        if true_acc >= acc_threshold:
            print(
                f"[direct-lstar/fnr] round {round_idx}: converged, "
                f"{learner.num_states} states"
            )
            break

        uncoverable = uncoverable_access_strings(pst, dt)
        if uncoverable:
            examples = ", ".join(
                "".join(map(str, p)) or "eps" for p, _ in uncoverable[:5]
            )
            print(
                f"[direct-lstar/fnr] round {round_idx}: {len(uncoverable)} access "
                f"string(s) reach states no sampled prefix can cover at length "
                f"{pst.sampler.length} (e.g. {examples}); target not learnable with "
                "this sampler, stopping"
            )
            break

        _grow_representative_pool(
            pst,
            learner,
            dfa,
            state,
            indecisive_fraction=indecisive_fraction,
            min_indecisive=min_indecisive,
            per_state=per_state,
        )
        print(
            f"[direct-lstar/fnr] round {round_idx}: {learner.num_states} states, "
            f"est {true_acc:.3f}, {len(state.accumulated)} accumulated indecisive, "
            f"{int(pst.table.representative.sum())} rep / {pst.num_prefixes} total"
        )
        if stall.stalled(
            states=learner.num_states,
            improved=true_acc > prior_best + 1e-9,
            boundary_strings=len(state.accumulated),
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
