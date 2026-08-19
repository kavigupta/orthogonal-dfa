"""Turn a trained :class:`NeuralDFA` into an ``automata-lib`` DFA."""

import numpy as np
import torch
from automata.fa.dfa import DFA

from orthogonal_dfa.l_star.dfa_utils import (
    count_paths_to_state,
    sample_string_reaching_state,
)
from orthogonal_dfa.l_star.statistics import binomial_side_of_boundary


def accept_threshold(accept_probs, weights):
    """Boundary between the accepting and rejecting clusters of accept probabilities.

    The learned rates sit at the two *noise* rates, not at 0 and 1, and those are not
    symmetric about 0.5 — ``AsymmetricBernoulli(p_0=0.10, p_1=0.40)`` puts the boundary
    at 0.25. So the split is found by a frequency-weighted 1-D 2-means (brute force over
    cut points; there are at most a few dozen states) rather than assumed.
    """
    order = np.argsort(accept_probs)
    values, mass = accept_probs[order], weights[order]
    keep = mass > 0
    values, mass = values[keep], mass[keep]
    if len(values) < 2 or values[-1] - values[0] < 1e-6:
        return 0.5
    best, cut = None, None
    for i in range(1, len(values)):
        lo, hi = slice(0, i), slice(i, None)
        if mass[lo].sum() <= 0 or mass[hi].sum() <= 0:
            continue
        mu_lo = (values[lo] * mass[lo]).sum() / mass[lo].sum()
        mu_hi = (values[hi] * mass[hi]).sum() / mass[hi].sum()
        cost = (mass[lo] * (values[lo] - mu_lo) ** 2).sum() + (
            mass[hi] * (values[hi] - mu_hi) ** 2
        ).sum()
        if best is None or cost < best:
            best, cut = cost, (mu_lo + mu_hi) / 2
    return 0.5 if cut is None else float(cut)


def extract_dfa(model, stats, *, initial, minify=True):
    """``(dfa, boundary)`` for the current model. Unreachable states are dropped by
    ``minify``, which also collapses any deterministic refinement of Nerode back onto
    the minimal DFA — which is why over-provisioning ``num_states`` is safe."""
    with torch.no_grad():
        # The empty continuation is acceptance of the prefix itself; the longer ones exist
        # only to supervise the state partition.
        accept_probs = model.accept_probs().cpu().numpy()
        frequency = stats.support().mean(0).cpu().numpy()
    boundary = accept_threshold(accept_probs, frequency)
    dfa = DFA(
        states=set(range(model.num_states)),
        input_symbols=set(range(model.alphabet_size)),
        transitions=stats.transitions(),
        initial_state=initial,
        final_states={s for s in range(model.num_states) if accept_probs[s] > boundary},
        allow_partial=False,
    )
    return (dfa.minify(retain_names=False) if minify else dfa), boundary


def denoise_accept_labels(dfa, oracle, rng, length, *, boundary, max_samples=200):
    """Re-vote each state's accept label from strings sampled to *reach* that state.

    A state's conditional accept rate is the one quantity a frequency-weighted
    ``J_external`` can get wrong, because low-support states are exactly where the noise
    wins. Same correction as :func:`orthogonal_dfa.l_star.lstar.denoise_accept_labels`,
    and the same path-counting sampler; labels change, transitions do not.
    """

    def relabel(state):
        counts = count_paths_to_state(dfa, state, length)
        cap = min(max_samples, counts[length][dfa.initial_state])
        seen, accepts = set(), 0
        while len(seen) < cap:
            string = sample_string_reaching_state(dfa, counts, rng)
            if tuple(string) in seen:
                continue  # distinct strings only: the noise is fixed per string
            seen.add(tuple(string))
            accepts += int(oracle.membership_query(string))
            decision = binomial_side_of_boundary(accepts, len(seen), boundary)
            if decision is not None:
                return decision
        return None

    labels = {state: relabel(state) for state in dfa.states}
    # Undecided states keep the label training gave them.
    final = {
        s
        for s in dfa.states
        if (s in dfa.final_states if labels[s] is None else labels[s])
    }
    if final == set(dfa.final_states):
        return dfa
    return DFA(
        states=set(dfa.states),
        input_symbols=set(dfa.input_symbols),
        transitions={s: dict(v) for s, v in dfa.transitions.items()},
        initial_state=dfa.initial_state,
        final_states=final,
        allow_partial=False,
    ).minify(retain_names=False)
