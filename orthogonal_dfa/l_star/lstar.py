"""
Shared classification and accuracy machinery for the direct-L* learner.

``estimate_agreement_rate`` is the termination test -- how well the exported DFA
agrees with the tree read decisively -- and ``denoise_accept_labels`` corrects
noise-flipped accept labels at the end of a run.  Both are driven by
``counterexample_synthesis``; the learner itself is in ``transition_resolver``.
"""

from automata.fa.dfa import DFA

from .dfa_utils import (
    count_paths_to_state,
    sample_string_reaching_state,
    states_intermediate,
)
from .midfix_tree import oracle_decider
from .statistics import binomial_side_of_boundary


def _oracle_classify(tree, oracle, *, accept, reject, suffix_limit=None):
    """
    Reads a midfix tree against the oracle at the given thresholds, returning a
    (classify, classify_many) pair over one string and over a batch. suffix_limit
    reads a shorter slice of the base family for a cheaper, noise-tolerant read.
    """
    base = tree.base_family if suffix_limit is None else tree.base_family[:suffix_limit]
    decide, decide_level = oracle_decider(oracle, base, accept, reject)
    return (
        lambda seq: tree.classify(seq, decide),
        lambda seqs: tree.classify_many(seqs, decide_level),
    )


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


def locate_incorrect_point(classify, dfa, x, y, *, s0, s_end):
    # s0 and s_end are classify(x) and classify(x + y), passed in so the caller can
    # share them across many calls: estimate_agreement_rate holds x fixed (one s0 for
    # the whole loop) and batches the per-y s_end through classify_many.
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
        dt_state = classify(x + y[: mid_idx + 1])
        if dt_state is None:
            return None, "could not classify state during binary search"
        if dt_state == dfa_states_each[mid_idx + 1]:
            correct_idx = mid_idx
        else:
            incorrect_idx = mid_idx
    return x + y[: correct_idx + 1], y[correct_idx + 1]


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


def estimate_agreement_rate(pst, us, oracle, tree, dfa, *, num_samples, acc_threshold):
    """
    Estimate the DFA's true agreement rate with the tree on fresh random strings,
    starting from the empty prefix (so the DFA simulates from its actual
    initial_state).  Classification failures are excluded from the denominator.

    Sampling stops early as soon as a one-sided binomial test is confident which
    side of *acc_threshold* the true rate lies on.  The estimate is consumed only
    to decide ``true_acc >= acc_threshold`` (the termination test), so settling
    that decision is all the precision required.  When the true rate is far from
    the threshold a few dozen samples settle it, but near the threshold it can run
    to the full *num_samples* budget, which is why that budget caps the cost.

    Samples whose ``y`` classifications are independent are drawn in a chunk and
    classified through ``classify_many`` -- one oracle call per tree level for the
    whole chunk rather than per sample.  Only the disagreeing minority pays the
    sequential binary search.  Each chunk is exactly the span in which the test
    provably cannot fire (``_batch_before_possible_stop``), so batching draws no
    sample past the stopping point -- same queries as the sequential loop, just
    grouped -- and needs no chunk-size constant.
    """
    boundary = pst.decision_boundary
    # The one caller that needs the batched read too, so it keeps both closures.
    classify, classify_many = _oracle_classify(
        tree, oracle, accept=boundary, reject=boundary
    )

    # Minimum trials before the sequential test can fire: at acc_threshold near 1
    # the "above" tail cannot clear alpha with only a handful of samples anyway,
    # and this guards against an unlucky early run of (dis)agreements.
    min_valid = 30
    agreements = 0
    valid = 0
    # Every sample classifies from the empty prefix, so classify([]) is constant
    # across the loop; compute it once instead of re-querying the oracle on each
    # sample.  On multi-iteration benchmarks this empty-prefix reclassification was
    # ~24% of all oracle queries (it recurs on up to num_samples draws per call).
    s0 = classify([])
    if s0 is None:
        return 0.0  # every sample would fail to classify
    drawn = 0
    while drawn < num_samples:
        size = _batch_before_possible_stop(
            agreements, valid, acc_threshold, min_valid, num_samples - drawn
        )
        ys = [us.sample(pst.rng, pst.alphabet_size) for _ in range(size)]
        drawn += size
        ends = classify_many(ys)
        for y, s_end in zip(ys, ends):
            prefix, reason = locate_incorrect_point(
                classify, dfa, [], y, s0=s0, s_end=s_end
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
