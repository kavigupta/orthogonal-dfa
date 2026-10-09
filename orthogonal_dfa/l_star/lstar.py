"""
``denoise_accept_labels`` corrects noise-flipped accept labels at the end of a
run.  The synthesis loop that drives it lives in ``counterexample_synthesis``.
"""

from automata.fa.dfa import DFA

from .dfa_utils import (
    count_paths_to_state,
    sample_string_reaching_state,
    states_intermediate,
    uniform_weights,
)
from .statistics import (
    DENOISE_FAILURE_PROB,
    binomial_side_of_boundary,
    denoise_sample_size,
)


def denoise_accept_labels(pst, dfa, *, block_size=32):
    """Recompute each reachable state's accept/reject label from fresh oracle samples.

    Discovery can noise-flip a low-support reject state to accept, leaking ~2% false
    positives (see ``TestLStarBimodalReproducer``). For each state we sample distinct
    length-``pst.sampler.length`` strings that reach it (the standard path-counting DFA
    sampler) and query the oracle, flipping the label only when a binomial test of the
    accept rate lands significantly on one side of ``pst.decision_boundary``. Correct
    labels never reach significance on the wrong side, so only noise-flips get corrected;
    a state that can't decide within the samples that test needs at this oracle's signal
    keeps its discovery label. Labels change, transitions don't.
    """
    length = pst.sampler.length
    # States that used the whole budget without deciding, as opposed to those with
    # too few strings reaching them to have had a chance.
    exhausted = []
    max_samples = denoise_sample_size(
        pst.config.min_signal_strength, pst.decision_boundary
    )
    if max_samples is None:
        print(
            f"Denoise skipped: no sample size decides at signal "
            f"{pst.config.min_signal_strength} around a boundary of "
            f"{pst.decision_boundary:.4f}"
        )
        return dfa
    weights = pst.sampler.symbol_weights(pst.alphabet_size)

    def relabel(state):
        # True=accept, False=reject, None=undecided (keep the discovery label).
        # How many distinct strings there are to draw, as against how likely the
        # learner is to draw each: the cap is the first, the walk the second.
        reachable = count_paths_to_state(dfa, state, length, uniform_weights(dfa))[
            length
        ][dfa.initial_state]
        cap = min(max_samples, reachable)
        drawn_from = count_paths_to_state(dfa, state, length, weights)
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
                string = sample_string_reaching_state(dfa, drawn_from, pst.rng, weights)
                if string in seen:
                    continue  # need distinct strings for independent oracle draws
                seen.add(string)
                block.append(string)
            for bit in pst.table.memo.membership_queries(block):
                accepts += int(bit)
                n += 1
                decision = binomial_side_of_boundary(
                    accepts,
                    n,
                    pst.decision_boundary,
                    failure_prob=DENOISE_FAILURE_PROB,
                )
                if decision is not None:
                    return decision
        if reachable >= max_samples:
            exhausted.append(state)
        return None

    label = {
        states_intermediate(dfa.initial_state, prefix, dfa)[-1]: None
        for prefix in pst.table.prefixes
    }
    label = {state: relabel(state) for state in label}
    if exhausted:
        print(
            f"Denoise spent {max_samples} samples on {sorted(exhausted)} without "
            f"reaching significance either side of {pst.decision_boundary:.4f}"
        )

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
