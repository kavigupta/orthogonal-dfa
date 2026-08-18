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

import numpy as np
from automata.fa.dfa import DFA

from .dfa_utils import per_state_sample
from .lstar import denoise_accept_labels, estimate_agreement_rate
from .statistics import binomial_side_of_boundary
from .transition_resolver import TransitionResolver

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
    ordered = sorted(tuple(b) for b in resolver.indecisive)
    np.random.default_rng(0).shuffle(ordered)
    return [list(t) for t in ordered[:target]]


class _PoolState:
    """The pool state carried across rounds: the initial uniform sample (kept in
    the representative set every round so global calibration stays anchored to the
    sampling distribution even if the per-state sample is skewed), the accumulated
    boundary strings (with a ``seen`` set to dedup them), and last round's sample."""

    def __init__(self, baseline):
        self.baseline = [list(p) for p in baseline]
        self.accumulated = []
        self.seen = set()
        self.sampled = []


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
    target = max(int(indecisive_fraction * pst.num_prefixes), min_indecisive)
    for t in _take_indecisive(resolver, target):
        key = tuple(t)
        if key not in state.seen:
            state.seen.add(key)
            state.accumulated.append(t)
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
    invariance_threshold=None,
):
    patience = _default_patience(acc_threshold)
    # Kept across rounds: the FNR gate resolves the chain one state per round, so
    # earlier rounds' boundary strings keep the family honest about the whole
    # chain (they turn decisive once their state is resolved).
    baseline = [
        p for p, keep in zip(pst.table.prefixes, pst.table.representative) if keep
    ]
    state = _PoolState(baseline)
    stall = _StallDetector(STALL_PATIENCE)
    best_acc = -1.0
    while True:
        print(f"Starting synthesis iteration with {pst.num_prefixes} prefixes")
        resolver = TransitionResolver(pst, invariance_threshold=invariance_threshold)
        resolver.build()
        resolver.counterexample_pass(
            max_probes=COUNTEREXAMPLE_PROBES, patience=patience
        )
        dfa, dt = resolver.export()
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
            yield dfa, dt, true_acc, pst.decision_boundary
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
            yield dfa, dt, true_acc, pst.decision_boundary
            return
        _grow_representative_pool(
            pst,
            resolver,
            dfa,
            state,
            indecisive_fraction=indecisive_fraction,
            min_indecisive=min_indecisive,
            per_state=per_state,
        )
        improved = true_acc > best_acc
        best_acc = max(best_acc, true_acc)
        if stall.stalled(
            states=dt.num_states,
            improved=improved,
            boundary_strings=len(state.accumulated),
        ):
            print(
                f"No progress ({dt.num_states} states) in {STALL_PATIENCE} rounds "
                "-- pool churning without resolving; stopping synthesis"
            )
            yield dfa, dt, true_acc, pst.decision_boundary
            return
        yield dfa, dt, true_acc, pst.decision_boundary


def do_counterexample_driven_synthesis(
    pst, *, acc_threshold: float, invariance_threshold=None
) -> DFA:
    # Rounds are not monotone -- rebuilding the representative pool re-clusters,
    # so a later family can classify worse -- so keep the most accurate
    # hypothesis, not the last. The boundary is kept with it because denoising
    # reads the tree against it.
    best_acc, best_dfa, best_dt, best_boundary = -1.0, None, None, None
    for dfa, dt, true_acc, boundary in counterexample_driven_synthesis(
        pst, acc_threshold=acc_threshold, invariance_threshold=invariance_threshold
    ):
        if true_acc > best_acc:
            best_acc, best_dfa, best_dt, best_boundary = true_acc, dfa, dt, boundary
    dfa, dt = best_dfa, best_dt
    if dfa is not None:
        pst.decision_boundary = best_boundary
        dfa = denoise_accept_labels(pst, dfa)
    return dfa, dt
