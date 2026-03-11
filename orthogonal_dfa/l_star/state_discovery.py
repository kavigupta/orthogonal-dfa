from typing import List

import numpy as np
import scipy
import tqdm.auto as tqdm

from .cluster import sample_suffix_family
from .structures import DecisionTree, TriPredicate


def cascade(mask_1, mask_2):
    mask_1 = mask_1.copy()
    mask_1[mask_1] = mask_2
    return mask_1


def split_states(pst, vs, path, subset_mask):
    decision = pst.compute_decision(vs, subset_mask)
    vs_actual = [pst.suffix_bank[v] for v in vs]
    return [
        (
            path + [(TriPredicate(vs_actual, pst.config.evidence_thresh), True)],
            cascade(subset_mask, decision >= pst.config.evidence_thresh),
        ),
        (
            path + [(TriPredicate(vs_actual, pst.config.evidence_thresh), False)],
            cascade(subset_mask, decision < 1 - pst.config.evidence_thresh),
        ),
    ]


def prepend_to_all(pst, vs: List[int], prefix: int):
    vs_new = []
    for v in tqdm.tqdm(vs, desc="Prepending to all suffixes", delay=1):
        _, _, v_new = pst.record_suffix([prefix] + pst.suffix_bank[v])
        vs_new.append(v_new)
    return vs_new


def overlaps(pst, states, vs):
    decision = pst.compute_decision(vs, np.ones(pst.num_prefixes, dtype=bool))
    masks = np.array(
        [
            decision > pst.config.evidence_thresh,
            decision < 1 - pst.config.evidence_thresh,
        ]
    )
    existing_states = np.array([m for _, m in states])
    assert (
        existing_states.shape[1] == pst.num_prefixes
    ), f"[existing states] Expected {pst.num_prefixes}, got {existing_states.shape[1]}"
    assert (
        masks.shape[1] == pst.num_prefixes
    ), f"[masks] Expected {pst.num_prefixes}, got {masks.shape[1]}"
    valid = np.any(masks, 0) & np.any(existing_states, 0)
    masks, existing_states = masks[:, valid], existing_states[:, valid]
    freqs = (masks[:, None] & existing_states[None]).sum(-1).T
    denominators = freqs.sum(-1)
    print(freqs)
    split_idxs = []
    for i, (denom, (n1, n2)) in enumerate(zip(denominators, freqs)):
        if denom == 0:
            continue
        pvals = [
            1 - scipy.stats.binom.cdf(n1, denom, pst.config.decision_rule_fpr),
            1 - scipy.stats.binom.cdf(n2, denom, pst.config.decision_rule_fpr),
        ]
        pval = max(pvals)
        if pval < pst.config.split_pval:
            split_idxs.append(i)
    return split_idxs


def discover_states(pst) -> List[DecisionTree]:
    _, _, v_idx = pst.record_suffix([])
    vs = sample_suffix_family(pst, v_idx)
    vs_queue = [([], vs)]
    states = [([], np.ones(len(pst.prefixes), bool))]

    def split_with(state_indices, vs):
        states_to_split = [states.pop(i) for i in reversed(sorted(state_indices))]
        for decision, m2 in states_to_split:
            states.extend(split_states(pst, vs, decision, m2))

    while vs_queue:
        path, vs_current = vs_queue.pop()
        print(f"Num states: {len(states)}; processing {path}")
        ol = overlaps(pst, states, vs_current)
        if not ol:
            print("Done")
            continue
        split_with(ol, vs_current)
        vs_queue.extend(
            ([c] + path, prepend_to_all(pst, vs_current, c))
            for c in range(pst.alphabet_size)
        )

    fdt = [x for x, _ in states]
    return fdt
