from collections import defaultdict
from typing import List

import numpy as np
import scipy
import tqdm.auto as tqdm

from .cluster import sample_suffix_family
from .structures import (
    DecisionTree,
    DecisionTreeInternalNode,
    DecisionTreeLeafNode,
    TriPredicate,
)


def cascade(mask_1, mask_2):
    mask_1 = mask_1.copy()
    mask_1[mask_1] = mask_2
    return mask_1


def prepend_to_all(pst, vs: List[int], prefix: int):
    vs_new = []
    for v in tqdm.tqdm(vs, desc="Prepending to all suffixes", delay=1):
        _, _, v_new = pst.record_suffix([prefix] + pst.suffix_bank[v])
        vs_new.append(v_new)
    return vs_new


class StateTracker:
    def __init__(self, num_prefixes: int):
        self.states = [([], np.ones(num_prefixes, bool))]

    def __len__(self):
        return len(self.states)

    @property
    def paths(self):
        return [path for path, _ in self.states]

    @staticmethod
    def _extend_path(path, vs_actual, accept_thresh, reject_thresh, accepted):
        return path + [
            (TriPredicate(vs_actual, accept_thresh, reject_thresh), accepted)
        ]

    def split(self, pst, state_indices, vs):
        states_to_split = [self.states.pop(i) for i in reversed(sorted(state_indices))]
        for path, mask in states_to_split:
            decision = pst.compute_decision(vs, mask)
            vs_actual = [pst.suffix_bank[v] for v in vs]
            accept_thresh = pst.accept_thresh
            reject_thresh = pst.reject_thresh
            self.states.extend(
                [
                    (
                        self._extend_path(
                            path, vs_actual, accept_thresh, reject_thresh, True
                        ),
                        cascade(mask, decision >= accept_thresh),
                    ),
                    (
                        self._extend_path(
                            path, vs_actual, accept_thresh, reject_thresh, False
                        ),
                        cascade(mask, decision < reject_thresh),
                    ),
                ]
            )

    @property
    def state_masks(self):
        return np.array([m for _, m in self.states])

    def to_decision_tree(self) -> DecisionTree:
        paths = self.paths
        if not paths:
            raise ValueError("State tracker has no states")

        partial_tree = {
            tuple(path): DecisionTreeLeafNode(i) for i, path in enumerate(paths)
        }
        while len(partial_tree) > 1:
            path_1, path_2 = _locate_mergeable_paths(partial_tree)
            (*prefix, (predicate, is_accepting)) = path_1
            if is_accepting:
                path_1, path_2 = path_2, path_1
            node = DecisionTreeInternalNode(
                predicate=predicate,
                by_rejection=(
                    partial_tree[path_1],
                    partial_tree[path_2],
                ),
            )
            del partial_tree[path_1]
            del partial_tree[path_2]
            partial_tree[tuple(prefix)] = node
        return partial_tree[()]


def _locate_mergeable_paths(partial_tree):
    by_everything_but_last = defaultdict(list)
    for path in partial_tree:
        by_everything_but_last[path[:-1]].append(path)
    assert any(len(v) >= 2 for v in by_everything_but_last.values())
    prefix = next(p for p, v in by_everything_but_last.items() if len(v) >= 2)
    assert len(by_everything_but_last[prefix]) == 2
    first, second = by_everything_but_last[prefix]
    assert first[-1][0] == second[-1][0]
    assert {first[-1][1], second[-1][1]} == {True, False}
    return first, second


def overlapping_states(pst, tracker, vs):
    decision = pst.compute_decision(vs, np.ones(pst.num_prefixes, dtype=bool))
    masks = np.array(
        [
            decision >= pst.accept_thresh,
            decision < pst.reject_thresh,
        ]
    )
    existing_states = tracker.state_masks
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


def discover_states(pst, first_round: bool) -> DecisionTree:
    _, _, v_idx = pst.record_suffix([])
    vs, decision_boundary = sample_suffix_family(pst, v_idx, first_round)
    pst.decision_boundary = decision_boundary
    vs_queue = [([], vs)]
    tracker = StateTracker(len(pst.prefixes))

    while vs_queue:
        path, vs_current = vs_queue.pop()
        print(f"Num states: {len(tracker)}; processing {path}")
        ol = overlapping_states(pst, tracker, vs_current)
        if not ol:
            print("Done")
            continue
        tracker.split(pst, ol, vs_current)
        vs_queue.extend(
            ([c] + path, prepend_to_all(pst, vs_current, c))
            for c in range(pst.alphabet_size)
        )

    return tracker.to_decision_tree()
