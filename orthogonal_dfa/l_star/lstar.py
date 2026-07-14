"""
Key Challenges:

Splitting on a criterion does not exclude the possibility that the same criterion could come up again. We need
a way to ensure that if a decision is made, the same decision will not be made later. Possible fix: require
a full set of classifier strings. Not sure why this would work, but maybe it will.

Maybe one thing we could do is have "confident" classifications during the creation, like just drop everything
in the classification between 40% and 60%. This way, we have much greater confidence that we won't find the same
thing twice, and therefore have lower thresholds otherwise.

Things to work on:

Evidence thresholds need some work. Currently there's the possibiliy of p-hacking. We need to do multiple comparisons.


"""

import copy
import warnings
from collections import defaultdict

import numpy as np
import tqdm.auto as tqdm
from automata.fa.dfa import DFA

from .dfa_utils import (
    count_paths_to_state,
    final_states_all_initial,
    sample_string_reaching_state,
    states_intermediate,
)
from .state_discovery import discover_states
from .statistics import (
    binomial_side_of_boundary,
    split_detection_population,
    unlikely_this_many_agreements,
)
from .structures import DecisionTree, DecisionTreeLeafNode, TriPredicate


def classify_states_with_decision_tree(pst, dt: DecisionTree):
    if isinstance(dt, DecisionTreeLeafNode):
        return np.full(len(pst.prefixes), dt.state_idx)
    results = np.full(len(pst.prefixes), -1)
    # Classify straight from the cached prefix x suffix mask matrix
    # (compute_decision_from_strings reads corresponding_masks), applying each node's
    # OWN thresholds rather than the PST's current margins.  For a discovery-time tree
    # the predicate thresholds equal the margins in effect here, so existing callers are
    # unchanged; for a decisive tree (accept==reject==boundary) this reproduces
    # dt.classify(prefix, oracle) for every prefix without re-querying the oracle.
    decision = pst.compute_decision_from_strings(dt.predicate.vs)
    rej = decision < dt.predicate.reject_threshold
    acc = decision >= dt.predicate.accept_threshold
    results[rej] = classify_states_with_decision_tree(pst, dt.by_rejection[0])[rej]
    results[acc] = classify_states_with_decision_tree(pst, dt.by_rejection[1])[acc]
    return results


def compute_transition_matrix(pst, dt: DecisionTree) -> np.ndarray:
    states = classify_states_with_decision_tree(pst, dt)
    states_after_c = [
        classify_states_with_decision_tree(
            pst,
            dt.map_over_predicates(
                lambda p, c=c: TriPredicate(
                    [[c] + x for x in p.vs], p.accept_threshold, p.reject_threshold
                )
            ),
        )
        for c in range(pst.alphabet_size)
    ]
    num_states = dt.num_states
    transitions = np.zeros((num_states, pst.alphabet_size, num_states), dtype=int)
    for c, states_c in enumerate(states_after_c):
        valid = states_c >= 0
        np.add.at(
            transitions,
            (states[valid], c, states_c[valid]),
            1,
        )
    print("Transition matrix:")
    print(transitions)

    return transitions.argmax(-1)


def optimal_dfa(pst, dt: DecisionTree):
    transitions = compute_transition_matrix(pst, dt)
    num_states = dt.num_states

    accepting_states = set(dt.by_rejection[1].collect_states())

    dfas = [
        DFA(
            states=set(range(num_states)),
            input_symbols=set(range(pst.alphabet_size)),
            transitions={
                s: {sym: transitions[s, sym] for sym in range(pst.alphabet_size)}
                for s in range(num_states)
            },
            initial_state=initial_state,
            final_states=accepting_states,
        )
        for initial_state in range(num_states)
    ]

    # Compare DFA state assignments against decision tree state assignments
    dt_states = classify_states_with_decision_tree(pst, dt)
    confident = dt_states >= 0
    dt_states = dt_states[confident]

    dfa_states = final_states_all_initial(
        transitions, [pre for pre, is_conf in zip(pst.prefixes, confident) if is_conf]
    )
    success_rates = (dfa_states == dt_states).mean(1)
    best_idx = np.argmax(success_rates)
    print(
        f"Best DFA has success rate on 'correct' states {success_rates[best_idx]:.4f}"
    )
    disagreements = np.where((dfa_states[best_idx] != dt_states))[0]
    print(f"Disagreements on indices {disagreements} out of {len(dt_states)}")
    for idx in disagreements:
        print(idx)
        print(
            f"  prefix {pst.prefixes[idx]} classified as {dt_states[idx]} by DT, "
            f"but DFA has final state {dfa_states[best_idx, idx]}"
        )
    return dfas[best_idx]


def denoise_accept_labels(pst, dfa, *, max_samples=200):
    """Recompute each reachable state's accept/reject label from fresh oracle samples.

    Discovery can noise-flip a low-support reject state to accept, leaking ~2% false
    positives (see ``TestLStarBimodalReproducer``). For each state we sample distinct
    length-``pst.sampler.length`` strings that reach it (the standard path-counting DFA
    sampler) and query the oracle, flipping the label only when a binomial test of the
    accept rate lands significantly on one side of ``pst.decision_boundary``. Correct
    labels never reach significance on the wrong side, so only noise-flips get corrected;
    a state that can't decide within ``max_samples`` distinct strings keeps its discovery
    label. Labels change, transitions don't.
    """
    length = pst.sampler.length

    def relabel(state):
        # True=accept, False=reject, None=undecided (keep the discovery label).
        counts = count_paths_to_state(dfa, state, length)
        cap = min(max_samples, counts[length][dfa.initial_state])
        seen, accepts = set(), 0
        while len(seen) < cap:
            string = sample_string_reaching_state(dfa, counts, pst.rng)
            if tuple(string) in seen:
                continue  # need distinct strings for independent oracle draws
            seen.add(tuple(string))
            accepts += int(pst.oracle.membership_query(string))
            decision = binomial_side_of_boundary(
                accepts, len(seen), pst.decision_boundary
            )
            if decision is not None:
                return decision
        return None

    label = {
        states_intermediate(dfa.initial_state, prefix, dfa)[-1]: None
        for prefix in pst.prefixes
    }
    label = {state: relabel(state) for state in label}

    def is_final(s):
        # Decided states use the new label; the rest keep the discovery label.
        return s in dfa.final_states if label.get(s) is None else label[s]

    new_final = {s for s in dfa.states if is_final(s)}
    if new_final == set(dfa.final_states):
        return dfa
    print(f"Denoised accept labels: {sorted(dfa.final_states)} -> {sorted(new_final)}")
    return DFA(
        states=set(dfa.states),
        input_symbols=set(dfa.input_symbols),
        transitions={s: dict(dfa.transitions[s]) for s in dfa.states},
        initial_state=dfa.initial_state,
        final_states=new_final,
        allow_partial=False,
    )


def add_counterexample_prefixes(pst, dt, dfa, count, *, expected_acc):
    results = generate_counterexamples(
        pst,
        pst.sampler,
        pst.oracle,
        dt,
        dfa,
        count=count,
        expected_acc=expected_acc,
    )
    if results:
        pst.add_prefixes(results)
    return results


def locate_incorrect_point(oracle, dt, dfa, x, y, *, s0):
    # ``s0`` is ``dt.classify(x, oracle)``, passed in by the caller: callers that hold
    # ``x`` fixed across many calls (e.g. estimate_agreement_rate, where x is always
    # the empty prefix) compute it once instead of re-querying the oracle every call.
    if s0 is None:
        return None, "could not classify initial state"
    dfa_states_each = states_intermediate(s0, y, dfa)
    if dt.classify(x + y, oracle) == dfa_states_each[-1]:
        return None, "no inconsistency"
    correct_idx = 0
    incorrect_idx = len(y)
    # binary search for first incorrect index
    while correct_idx < incorrect_idx - 1:
        mid_idx = (correct_idx + incorrect_idx) // 2
        dt_state = dt.classify(x + y[: mid_idx + 1], oracle)
        if dt_state is None:
            return None, "could not classify state during binary search"
        if dt_state == dfa_states_each[mid_idx + 1]:
            correct_idx = mid_idx
        else:
            incorrect_idx = mid_idx
    return x + y[: correct_idx + 1], y[correct_idx + 1]


def generate_counterexamples(pst, us, oracle, dt, dfa, *, count, expected_acc):
    boundary = pst.decision_boundary
    # The counterexample pipeline classifies strings many times: ~log2(string_len)
    # binary search steps + 2 decisive checks, each traversing the full DT.  A
    # false positive just adds an uninformative prefix (harmless), so we can
    # tolerate a much higher overall error rate than state discovery (which uses
    # decision_rule_fpr).  We use 0.2 as the whole-pipeline budget and union-bound
    # over all node-level decisions.
    from .statistics import compute_suffix_size_counterexample_gen as _compute_sfx

    counterexample_fpr = 0.2
    string_len = pst.sampler.length
    num_classifications = 2 + int(np.ceil(np.log2(string_len)))
    num_node_decisions = num_classifications * dt.depth
    effective_p = 0.5 + pst.config.min_signal_strength
    per_node_budget = counterexample_fpr / max(num_node_decisions, 1)
    scaled_suffix_size = _compute_sfx(per_node_budget, effective_p)
    dt_with_reduced_predicates = dt.map_over_predicates(
        lambda p: TriPredicate(p.vs[:scaled_suffix_size], boundary, boundary)
    )
    dt_with_decisive_predicates = dt.map_over_predicates(
        lambda p: TriPredicate(p.vs, boundary, boundary)
    )
    pbar = tqdm.tqdm(total=count)
    additional_prefixes = []
    num_samples = 0
    num_agreements = 0
    while True:
        num_samples += 1
        x = us.sample(pst.rng, pst.alphabet_size)
        y = us.sample(pst.rng, pst.alphabet_size)
        prefix, sym = locate_incorrect_point(
            oracle,
            dt_with_reduced_predicates,
            dfa,
            x,
            y,
            s0=dt_with_reduced_predicates.classify(x, oracle),
        )
        if prefix is None:
            num_agreements += sym == "no inconsistency"
            if unlikely_this_many_agreements(num_agreements, num_samples, expected_acc):
                warnings.warn(
                    f"Observed {num_agreements} 'no inconsistency' results in"
                    f" {num_samples} samples, which is unlikely given expected"
                    f" accuracy {expected_acc:.3f}; stopping counterexample"
                    f" search early with {len(additional_prefixes)}/{count}"
                    f" prefixes"
                )
                pbar.close()
                return additional_prefixes
            continue
        if prefix in additional_prefixes or prefix in pst.prefixes:
            continue
        state_1 = dt_with_decisive_predicates.classify(prefix, oracle)
        state_2 = dfa.transitions[state_1][sym]
        if state_2 == dt_with_decisive_predicates.classify(prefix + [sym], oracle):
            continue
        additional_prefixes.append(prefix)
        pbar.update()
        if len(additional_prefixes) >= count:
            pbar.close()
            return additional_prefixes


def estimate_agreement_rate(
    pst, us, oracle, dt_decisive, dfa, *, num_samples, acc_threshold
):
    """
    Estimate the DFA's true agreement rate with the DT on fresh random strings,
    starting from the empty prefix (so the DFA simulates from its actual
    initial_state).  Classification failures are excluded from the denominator.

    Sampling stops early as soon as a one-sided binomial test is confident which
    side of *acc_threshold* the true rate lies on.  The estimate is consumed only
    to decide ``true_acc >= acc_threshold`` (the termination test) and as a loose
    ``expected_acc`` guard for counterexample search, so settling that decision is
    all the precision required; the rate is almost always far from the threshold
    (e.g. 0.2 or 0.8 vs 0.98), in which case a few dozen samples suffice instead
    of the full budget.  Sampling is still capped at *num_samples*, so this never
    costs more than the fixed-budget pass.
    """
    # Minimum trials before the sequential test can fire: at acc_threshold near 1
    # the "above" tail cannot clear alpha with only a handful of samples anyway,
    # and this guards against an unlucky early run of (dis)agreements.
    min_valid = 30
    agreements = 0
    valid = 0
    # Every sample classifies from the empty prefix, so dt_decisive.classify([]) is
    # constant across the loop; compute it once instead of re-querying the oracle on
    # each sample.  On multi-iteration benchmarks this empty-prefix reclassification
    # was ~24% of all oracle queries (it recurs on up to num_samples draws per call).
    s0 = dt_decisive.classify([], oracle)
    for _ in range(num_samples):
        y = us.sample(pst.rng, pst.alphabet_size)
        prefix, reason = locate_incorrect_point(oracle, dt_decisive, dfa, [], y, s0=s0)
        if prefix is None and reason == "no inconsistency":
            agreements += 1
            valid += 1
        elif prefix is not None:
            valid += 1
        else:
            # Could-not-classify samples leave the decision unchanged; don't test.
            continue
        if (
            valid >= min_valid
            and binomial_side_of_boundary(agreements, valid, acc_threshold) is not None
        ):
            break
    return agreements / valid if valid else 0.0


def prune_overrepresented_leaves(pst, dt_decisive):
    """Drop excess prefixes from over-represented leaves so already-saturated states
    stop accumulating redundant probes that every new suffix must be queried against.

    Each state is capped at the per-state prefix population its split test actually
    needs (:func:`split_detection_population`), derived from the search parameters --
    the point past which more prefixes in a state change no discovery decision.
    Leaves above the cap are trimmed to it by dropping a random subset.  Core
    (non-representative) and undecided prefixes are never dropped.  Returns the number
    of prefixes dropped.
    """
    cap = split_detection_population(
        pst.config.decision_rule_fpr,
        pst.config.split_pval,
        pst.config.min_acc_rej,
    )
    leaves = classify_states_with_decision_tree(pst, dt_decisive)
    rep = pst.representative
    by_leaf = defaultdict(list)
    for i, leaf in enumerate(leaves.tolist()):
        if leaf >= 0 and rep[i]:
            by_leaf[leaf].append(i)
    drop = []
    for idxs in by_leaf.values():
        if len(idxs) > cap:
            drop.extend(
                pst.rng.choice(idxs, size=len(idxs) - cap, replace=False).tolist()
            )
    if not drop:
        return 0
    keep_mask = np.ones(len(pst.prefixes), dtype=bool)
    keep_mask[drop] = False
    print(
        f"Pruning {len(drop)} prefixes; capping leaves at {cap} (per-state split population)"
    )
    pst.prune_prefixes(keep_mask)
    return len(drop)


def enrich_underrepresented_leaves(pst, dt_decisive, *, count):
    """
    Sample random length-L prefixes routed (via the decisive DT) to leaves
    whose current population is below the median.  This rebalances the PST
    so that the next suffix-family clustering has enough signal to pick
    suffixes that shatter under-represented leaves.

    See docs/counterexample_poor_case_findings.md: the
    `test_another_countexample_poor_case` failure was caused by a single
    ground-truth state receiving only ~1.5% of uniform random prefixes,
    which left the suffix-family clustering unable to find discriminating
    suffixes for that state.
    """
    # Classify every existing prefix through the decisive tree directly from the cached
    # mask matrix instead of re-querying the oracle once per prefix: all these
    # prefix x suffix pairs are already in corresponding_masks.  -1 marks undecided.
    leaves = classify_states_with_decision_tree(pst, dt_decisive)
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

    seen = {tuple(p) for p in pst.prefixes}
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
        leaf = dt_decisive.classify(p, pst.oracle)
        if leaf is None or leaf not in target_leaves:
            continue
        new_prefixes.append(p)
        seen.add(t)
        pbar.update()
    pbar.close()
    if new_prefixes:
        pst.add_prefixes(new_prefixes)
    return new_prefixes


def counterexample_driven_synthesis(
    pst, *, additional_counterexamples: int, acc_threshold: float
):
    first_round = True
    while True:
        print(f"Starting synthesis iteration with {pst.num_prefixes} prefixes")
        while True:
            dt = discover_states(pst, first_round=first_round)
            first_round = False
            print(f"Extracted flat decision tree with {dt.num_states} states")
            if dt.num_states > 1:
                break
            pst.sample_more_prefixes()
        dfa = optimal_dfa(pst, dt)
        print("DFA found!")
        print(dfa)
        boundary = pst.decision_boundary
        dt_decisive = dt.map_over_predicates(
            lambda p: TriPredicate(p.vs, boundary, boundary)
        )
        true_acc = estimate_agreement_rate(
            pst,
            pst.sampler,
            pst.oracle,
            dt_decisive,
            dfa,
            num_samples=2000,
            acc_threshold=acc_threshold,
        )
        print(f"Estimated DFA accuracy on fresh samples: {true_acc:.4f}")
        if true_acc >= acc_threshold:
            print(f"Achieved desired accuracy of {acc_threshold}; stopping synthesis")
            yield dfa, dt, None
            return
        ce = add_counterexample_prefixes(
            pst, dt, dfa, additional_counterexamples, expected_acc=true_acc
        )
        enriched = enrich_underrepresented_leaves(
            pst, dt_decisive, count=additional_counterexamples
        )
        if not ce and not enriched:
            print(
                "Neither counterexample search nor leaf enrichment found"
                " new prefixes; stopping synthesis"
            )
            yield dfa, dt, None
            return
        if pst.config.prune_saturated_leaves:
            prune_overrepresented_leaves(pst, dt_decisive)
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
