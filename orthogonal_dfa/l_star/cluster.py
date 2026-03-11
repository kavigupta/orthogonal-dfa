from typing import List

import numpy as np


def identify_cluster_around(pst, seed: int, count: int) -> List[int]:
    masks = np.array(pst.corresponding_masks)
    cluster = [seed]
    loss = float("inf")
    while True:
        cluster_center = masks[cluster].mean(0) > 0.5
        losses = (masks != cluster_center).sum(1)
        cluster = losses.argsort()[:count]
        if seed not in cluster:
            cluster = np.concatenate([[seed], cluster[: count - 1]])
        new_loss = losses[cluster].sum()
        if new_loss >= loss:
            break
        loss = new_loss
    return cluster.tolist()


def sample_suffix_family(pst, v: int) -> List[int]:
    prev_fnr = 1.0
    strategy = "suffix"
    while True:
        vs = identify_cluster_around(pst, v, pst.config.suffix_family_size)

        fnr = 1 if len(vs) < pst.config.suffix_family_size else pst.compute_fnr(vs)
        if fnr <= pst.config.fnr_limit:
            print("FNR limit reached")
            return vs

        if fnr >= prev_fnr or strategy == "prefix":
            strategy = "prefix" if strategy == "suffix" else "suffix"

        prev_fnr = fnr

        print(f"FNR {fnr:.4f} too high, sampling more suffixes")

        if strategy == "suffix":
            pst.sample_more_suffixes(amount=pst.config.suffix_family_size)
        else:
            pst.sample_more_prefixes()
