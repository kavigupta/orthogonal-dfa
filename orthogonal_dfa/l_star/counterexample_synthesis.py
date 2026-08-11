"""
Counterexample-driven synthesis: the E-L* learner loop.

Resolves a DFA from the current prefix pool (``resolve_dfa``), estimates its
accuracy, and when it falls short generates counterexample prefixes and enriches
under-represented leaves before resolving again -- until the estimate clears the
threshold or the search dries up.  The classification and accuracy primitives it
uses live in ``lstar``.
"""

import copy
import warnings

import numpy as np
import tqdm.auto as tqdm
from automata.fa.dfa import DFA

from .lstar import (
    _oracle_classify,
    denoise_accept_labels,
    estimate_agreement_rate,
    locate_incorrect_point,
)
from .statistics import binomial_side_of_boundary, counterexample_search_exhausted
from .transition_resolver import resolve_dfa


def classify_pool(pst, tree, *, accept, reject):
    """
    Classify every prefix in the pool to its leaf (or -1 if undecided), from
    the cached mask matrix. Uses accept and reject thresholds.
    """

    def decide_columns(midfix):
        decision = pst.compute_decision_from_strings(tree.suffixes(midfix))
        return decision >= accept, decision < reject

    return tree.classify_pool(pst.num_prefixes, decide_columns)


def add_counterexample_prefixes(pst, dt, dfa, count):
    results = generate_counterexamples(
        pst,
        pst.sampler,
        pst.oracle,
        dt,
        dfa,
        count=count,
    )
    if results:
        pst.table.add_prefixes(results)
    return results


#: Draws allowed per counterexample search, per prefix asked for.
SAMPLES_PER_COUNTEREXAMPLE = 50


def counterexample_sample_budget(count: int) -> int:
    return SAMPLES_PER_COUNTEREXAMPLE * count


def generate_counterexamples(pst, us, oracle, tree, dfa, *, count):
    boundary = pst.decision_boundary
    # The counterexample pipeline classifies strings many times: ~log2(string_len)
    # binary search steps + 2 decisive checks, each traversing the full tree.  A
    # false positive just adds an uninformative prefix (harmless), so we can
    # tolerate a much higher overall error rate than state discovery (which uses
    # decision_rule_fpr).  We use 0.2 as the whole-pipeline budget and union-bound
    # over all node-level decisions.
    from .statistics import compute_suffix_size_counterexample_gen as _compute_sfx

    counterexample_fpr = 0.2
    string_len = pst.sampler.length
    num_classifications = 2 + int(np.ceil(np.log2(string_len)))
    num_node_decisions = num_classifications * tree.depth
    effective_p = 0.5 + pst.config.min_signal_strength
    per_node_budget = counterexample_fpr / max(num_node_decisions, 1)
    scaled_suffix_size = _compute_sfx(per_node_budget, effective_p)
    # Both are decisive, reduced is more efficient
    reduced, _ = _oracle_classify(
        tree, oracle, accept=boundary, reject=boundary, suffix_limit=scaled_suffix_size
    )
    decisive, _ = _oracle_classify(tree, oracle, accept=boundary, reject=boundary)
    pbar = tqdm.tqdm(total=count)
    additional_prefixes = []
    num_samples = 0
    max_samples = counterexample_sample_budget(count)
    while True:
        num_samples += 1
        x = us.sample(pst.rng, pst.alphabet_size)
        y = us.sample(pst.rng, pst.alphabet_size)
        s0 = reduced(x)
        prefix, sym = locate_incorrect_point(
            reduced,
            dfa,
            x,
            y,
            s0=s0,
            # Skip the endpoint classification when x itself is unclassifiable --
            # locate_incorrect_point returns on s0 without reading s_end.
            s_end=(reduced(x + y) if s0 is not None else None),
        )
        if counterexample_search_exhausted(
            len(additional_prefixes), num_samples, count, max_samples
        ):
            warnings.warn(
                f"Counterexample search yielded {len(additional_prefixes)}/{count}"
                f" prefixes in {num_samples} samples; the decision tree and the"
                f" DFA disagree too rarely to reach {count} within"
                f" {max_samples} samples"
            )
            pbar.close()
            return additional_prefixes
        if prefix is None:
            continue
        if prefix in additional_prefixes or pst.table.contains_prefix(prefix):
            continue
        state_1 = decisive(prefix)
        state_2 = dfa.transitions[state_1][sym]
        if state_2 == decisive(prefix + [sym]):
            continue
        additional_prefixes.append(prefix)
        pbar.update()
        if len(additional_prefixes) >= count:
            pbar.close()
            return additional_prefixes


def enrich_underrepresented_leaves(pst, tree, *, count):
    """
    Sample random length-L prefixes routed (via the decisive tree) to leaves
    whose current population is below the median.  This rebalances the PST
    so that the next suffix-family clustering has enough signal to pick
    suffixes that shatter under-represented leaves.

    See docs/counterexample_poor_case_findings.md: the
    `test_another_countexample_poor_case` failure was caused by a single
    ground-truth state receiving only ~1.5% of uniform random prefixes,
    which left the suffix-family clustering unable to find discriminating
    suffixes for that state.
    """
    boundary = pst.decision_boundary
    decisive, _ = _oracle_classify(tree, pst.oracle, accept=boundary, reject=boundary)
    # Classify every existing prefix through the decisive tree directly from the cached
    # mask matrix instead of re-querying the oracle once per prefix: all these
    # prefix x suffix pairs are already in corresponding_masks.  -1 marks undecided.
    leaves = classify_pool(pst, tree, accept=boundary, reject=boundary)
    leaf_counts = {}
    for leaf in leaves.tolist():
        if leaf < 0:
            continue
        leaf_counts[leaf] = leaf_counts.get(leaf, 0) + 1
    if not leaf_counts:
        return []
    counts = sorted(leaf_counts.values())
    median = counts[len(counts) // 2]
    target_leaves = {leaf for leaf, c in leaf_counts.items() if c <= median}
    print(
        f"Leaf populations: {sorted(leaf_counts.items())}; enriching leaves"
        f" {sorted(target_leaves)} (median={median})"
    )

    seen = {tuple(p) for p in pst.table.prefixes}
    new_prefixes = []
    max_attempts = count * 200
    attempts = 0
    pbar = tqdm.tqdm(total=count, desc="Enriching under-represented leaves")
    while len(new_prefixes) < count and attempts < max_attempts:
        attempts += 1
        p = pst.sampler.sample(pst.rng, pst.alphabet_size)
        t = tuple(p)
        if t in seen:
            continue
        leaf = decisive(p)
        if leaf is None or leaf not in target_leaves:
            continue
        new_prefixes.append(p)
        seen.add(t)
        pbar.update()
    pbar.close()
    if new_prefixes:
        pst.table.add_prefixes(new_prefixes)
    return new_prefixes


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
    rep = pst.table.representative
    fam = pst.table.fully_observed()
    if len(fam) == 0 or not rep.any():
        return []

    eta = 0.5 - pst.config.min_signal_strength
    # Two prefixes at the same state agree on every suffix up to independent
    # per-cell noise, so their expected mask-disagreement rate is 2*eta*(1-eta).
    same_state_rate = 2 * eta * (1 - eta)
    n = len(fam)

    repr_masks = pst.table.observed_masks(fam, rep).T  # [n_repr, n_fam]
    leaves = classify_pool(
        pst, tree, accept=pst.accept_thresh, reject=pst.reject_thresh
    )
    potentially_problematic = np.flatnonzero(
        (~rep) & (leaves == -1)
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


def counterexample_driven_synthesis(
    pst, *, additional_counterexamples: int, acc_threshold: float
):
    while True:
        print(f"Starting synthesis iteration with {pst.num_prefixes} prefixes")
        while True:
            dfa, dt = resolve_dfa(pst)
            print(f"Resolved DFA with {dt.num_states} states")
            if dt.num_states > 1:
                break
            pst.sample_more_prefixes()
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
            yield dfa, dt, None
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
            yield dfa, dt, None
            return
        ce = add_counterexample_prefixes(pst, dt, dfa, additional_counterexamples)
        enriched = enrich_underrepresented_leaves(
            pst, dt, count=additional_counterexamples
        )
        if not ce and not enriched:
            print(
                "Neither counterexample search nor leaf enrichment found"
                " new prefixes; stopping synthesis"
            )
            yield dfa, dt, None
            return
        yield dfa, dt, copy.deepcopy(pst)


def do_counterexample_driven_synthesis(
    pst, *, additional_counterexamples: int, acc_threshold: float
) -> DFA:
    dfa = dt = None
    for dfa, dt, _ in counterexample_driven_synthesis(
        pst,
        additional_counterexamples=additional_counterexamples,
        acc_threshold=acc_threshold,
    ):
        pass
    if dfa is not None:
        dfa = denoise_accept_labels(pst, dfa)
    return dfa, dt
