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

import numpy as np
import tqdm.auto as tqdm
from automata.fa.dfa import DFA

from .dfa_utils import (
    count_paths_to_state,
    sample_string_reaching_state,
    states_intermediate,
)
from .midfix_tree import oracle_decider
from .statistics import binomial_side_of_boundary, counterexample_search_exhausted
from .transition_resolver import resolve_dfa


class _TreeClassifier:
    """Classifies fresh strings against a :class:`MidfixTree` through the oracle.

    Bundles the tree with a decision rule -- thresholds and, optionally, a shorter
    ``suffix_limit`` of the base family for a cheaper noise-tolerant read -- so a
    caller can hold "the decisive classifier" or "the reduced classifier" the way
    it used to hold a predicate-rewritten copy of the tree."""

    def __init__(self, tree, oracle, *, accept, reject, suffix_limit=None):
        self.tree = tree
        base = tree.base_family if suffix_limit is None else tree.base_family[:suffix_limit]
        self._decide, self._decide_level = oracle_decider(oracle, base, accept, reject)

    def classify(self, seq):
        return self.tree.classify(seq, self._decide)

    def classify_many(self, seqs):
        return self.tree.classify_many(seqs, self._decide_level)


def classify_pool(pst, tree, *, accept, reject):
    """Classify every prefix in the pool to its leaf (or -1 if undecided) straight
    from the cached prefix x suffix mask matrix, no oracle queries.  ``accept`` /
    ``reject`` are the thresholds every node is read at: the PST's margins for a
    discovery-time tree, or the boundary for a decisive read."""

    def decide_columns(midfix):
        decision = pst.compute_decision_from_strings(tree.suffixes(midfix))
        return decision >= accept, decision < reject

    return tree.classify_pool(pst.num_prefixes, decide_columns)


def denoise_accept_labels(pst, dfa, *, max_samples=200, block_size=32):
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
        seen, accepts, n = set(), 0, 0
        while len(seen) < cap:
            # Draw and query a block at a time.  The stopping rule is still read
            # after every individual sample, so the label is exactly the one the
            # one-at-a-time test would give; only the oracle calls are packed, at
            # the cost of at most one block of overshoot per state.  ``n`` counts
            # samples *scored*, which now lags ``len(seen)`` by up to a block.
            target = min(block_size, cap - len(seen))
            block = []
            while len(block) < target:
                string = sample_string_reaching_state(dfa, counts, pst.rng)
                if tuple(string) in seen:
                    continue  # need distinct strings for independent oracle draws
                seen.add(tuple(string))
                block.append(string)
            for bit in pst.oracle.membership_queries(block):
                accepts += int(bit)
                n += 1
                decision = binomial_side_of_boundary(accepts, n, pst.decision_boundary)
                if decision is not None:
                    return decision
        return None

    label = {
        states_intermediate(dfa.initial_state, prefix, dfa)[-1]: None
        for prefix in pst.table.prefixes
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


def locate_incorrect_point(classifier, dfa, x, y, *, s0, s_end):
    # ``s0`` and ``s_end`` are ``classifier.classify(x)`` and
    # ``classifier.classify(x + y)``, computed by the caller so it can share them
    # across many calls: estimate_agreement_rate holds ``x`` fixed (one ``s0`` for
    # the whole loop) and batches the per-``y`` ``s_end`` through ``classify_many``.
    if s0 is None:
        return None, "could not classify initial state"
    dfa_states_each = states_intermediate(s0, y, dfa)
    if s_end == dfa_states_each[-1]:
        return None, "no inconsistency"
    correct_idx = 0
    incorrect_idx = len(y)
    # binary search for first incorrect index
    while correct_idx < incorrect_idx - 1:
        mid_idx = (correct_idx + incorrect_idx) // 2
        dt_state = classifier.classify(x + y[: mid_idx + 1])
        if dt_state is None:
            return None, "could not classify state during binary search"
        if dt_state == dfa_states_each[mid_idx + 1]:
            correct_idx = mid_idx
        else:
            incorrect_idx = mid_idx
    return x + y[: correct_idx + 1], y[correct_idx + 1]


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
    # Both read the tree decisively (accept==reject==boundary); the reduced one uses
    # a shorter slice of the family for a cheaper, noise-tolerant classification.
    reduced = _TreeClassifier(
        tree, oracle, accept=boundary, reject=boundary, suffix_limit=scaled_suffix_size
    )
    decisive = _TreeClassifier(tree, oracle, accept=boundary, reject=boundary)
    pbar = tqdm.tqdm(total=count)
    additional_prefixes = []
    num_samples = 0
    max_samples = counterexample_sample_budget(count)
    while True:
        num_samples += 1
        x = us.sample(pst.rng, pst.alphabet_size)
        y = us.sample(pst.rng, pst.alphabet_size)
        s0 = reduced.classify(x)
        prefix, sym = locate_incorrect_point(
            reduced,
            dfa,
            x,
            y,
            s0=s0,
            # Skip the endpoint classification when x itself is unclassifiable --
            # locate_incorrect_point returns on s0 without reading s_end.
            s_end=(reduced.classify(x + y) if s0 is not None else None),
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
        state_1 = decisive.classify(prefix)
        state_2 = dfa.transitions[state_1][sym]
        if state_2 == decisive.classify(prefix + [sym]):
            continue
        additional_prefixes.append(prefix)
        pbar.update()
        if len(additional_prefixes) >= count:
            pbar.close()
            return additional_prefixes


def _batch_before_possible_stop(agreements, valid, boundary, min_valid, remaining):
    """The largest number of further valid samples that provably cannot let the
    early-stop test fire -- so drawing this many and batching them changes nothing
    the sequential loop would have decided, and never draws a sample past the stop.

    The test needs ``min_valid`` samples and then significance against ``boundary``.
    The soonest it *could* fire after ``k`` more samples is the best case: all ``k``
    agree (pushes the 'above' tail) or none do (the 'below' tail).  ``possible`` is
    monotonic in ``k`` (extra all-agree/all-disagree evidence only helps), so binary
    search finds the smallest firing ``k``; below it, batching is free."""
    lo = max(min_valid - valid, 1)

    def possible(k):
        return (
            binomial_side_of_boundary(agreements + k, valid + k, boundary) is True
            or binomial_side_of_boundary(agreements, valid + k, boundary) is False
        )

    if lo >= remaining or not possible(remaining):
        return remaining
    hi = remaining
    while lo < hi:
        mid = (lo + hi) // 2
        if possible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def estimate_agreement_rate(
    pst, us, oracle, classifier, dfa, *, num_samples, acc_threshold
):
    """
    Estimate the DFA's true agreement rate with the DT on fresh random strings,
    starting from the empty prefix (so the DFA simulates from its actual
    initial_state).  Classification failures are excluded from the denominator.

    Sampling stops early as soon as a one-sided binomial test is confident which
    side of *acc_threshold* the true rate lies on.  The estimate is consumed only
    to decide ``true_acc >= acc_threshold`` (the termination test), so settling
    that decision is all the precision required.  When the true rate is far from
    the threshold a few dozen samples settle it, but near the threshold it can run
    to the full *num_samples* budget (e.g. on the poor_case guard it hits the cap
    on most calls), which is why that budget caps the cost.

    Samples whose ``y`` classifications are independent are drawn in a chunk and
    classified through ``classify_many`` -- one oracle call per tree level for the
    whole chunk rather than per sample.  Only the disagreeing minority pays the
    sequential binary search.  Each chunk is exactly the span in which the test
    provably cannot fire (``_batch_before_possible_stop``), so batching draws no
    sample past the stopping point -- same queries as the sequential loop, just
    grouped -- and needs no chunk-size constant.
    """
    # Minimum trials before the sequential test can fire: at acc_threshold near 1
    # the "above" tail cannot clear alpha with only a handful of samples anyway,
    # and this guards against an unlucky early run of (dis)agreements.
    min_valid = 30
    agreements = 0
    valid = 0
    # Every sample classifies from the empty prefix, so classifier.classify([]) is
    # constant across the loop; compute it once instead of re-querying the oracle on
    # each sample.  On multi-iteration benchmarks this empty-prefix reclassification
    # was ~24% of all oracle queries (it recurs on up to num_samples draws per call).
    s0 = classifier.classify([])
    if s0 is None:
        return 0.0  # every sample would fail to classify
    drawn = 0
    while drawn < num_samples:
        size = _batch_before_possible_stop(
            agreements, valid, acc_threshold, min_valid, num_samples - drawn
        )
        ys = [us.sample(pst.rng, pst.alphabet_size) for _ in range(size)]
        drawn += size
        ends = classifier.classify_many(ys)
        for y, s_end in zip(ys, ends):
            prefix, reason = locate_incorrect_point(
                classifier, dfa, [], y, s0=s0, s_end=s_end
            )
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
                and binomial_side_of_boundary(agreements, valid, acc_threshold)
                is not None
            ):
                return agreements / valid
    return agreements / valid if valid else 0.0


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
    decisive = _TreeClassifier(tree, pst.oracle, accept=boundary, reject=boundary)
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
        leaf = decisive.classify(p)
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
    leaves = classify_pool(pst, tree, accept=pst.accept_thresh, reject=pst.reject_thresh)
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
        boundary = pst.decision_boundary
        decisive = _TreeClassifier(dt, pst.oracle, accept=boundary, reject=boundary)
        true_acc = estimate_agreement_rate(
            pst,
            pst.sampler,
            pst.oracle,
            decisive,
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
