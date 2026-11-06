#!/usr/bin/env python
"""Look at what the surrogate is actually doing on modulo9.

    python scripts/diagnose_surrogate.py [output_dir]

This found the memorisation bug that four rounds of reasoning missed: predicted rates were
piling up at 0.0/1.0 instead of the noise rates, which only beats predicting the rate if the
model is right about individual cells. Panels: true residue vs learned cluster, the response
rows, transition concentration, and where the rates sit relative to the conformal band.

modulo9 is the interpretable one: 9 states in a cycle, state = sum % 9, accepting {3, 6}.
So the true partition is known exactly and every learned object can be compared against it.
"""

import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliParityOracle
from orthogonal_dfa.l_star.neural.extract import accept_threshold
from orthogonal_dfa.l_star.neural.surrogate import (
    CellPool,
    SurrogateConfig,
    TableSurrogate,
    _encode_all,
    _fit,
    conformal_rate_bound,
    pad,
)
from orthogonal_dfa.l_star.structures import SymmetricBernoulli

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(OUT, exist_ok=True)
cfg = SurrogateConfig(num_prefixes=1500, rounds=5, seed=0)
oracle = BernoulliParityOracle(
    SymmetricBernoulli(0.8), 0, modulo=9, allowed_moduluses=(3, 6)
)
rng = np.random.default_rng(cfg.seed)
torch.manual_seed(cfg.seed)

prefixes = [[]] + [
    rng.integers(0, 2, size=rng.integers(0, 41)).tolist()
    for _ in range(cfg.num_prefixes - 1)
]
suffixes = [[]] + [
    rng.integers(0, 2, size=rng.integers(1, 11)).tolist()
    for _ in range(cfg.num_suffixes - 1)
]
pool = CellPool(oracle, prefixes, suffixes)
cells = [(i, j) for i in range(len(prefixes)) for j in range(len(suffixes))]
pool.observe([cells[k] for k in rng.permutation(len(cells))[: int(0.3 * len(cells))]])
observed = sorted(pool.answers)
pool.holdout.update(
    tuple(observed[k])
    for k in rng.permutation(len(observed))[: int(0.15 * len(observed))]
)

model = TableSurrogate(2, cfg)
opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
_fit(model, opt, pool, cfg=cfg, rng=rng, device="cpu", steps=1250)

residue = np.array([sum(p) % 9 for p in prefixes])
with torch.no_grad():
    v_pad, v_len = pad(suffixes, cfg.max_suffix_length)
    response = model.response(v_pad, v_len).T.numpy()  # (S, V)
current, successors = _encode_all(model, pool, cfg, "cpu", 2)
clusters = current.argmax(-1).numpy()
counts = np.bincount(clusters, minlength=cfg.num_states)
empty_col = [j for j, v in enumerate(suffixes) if len(v) == 0][0]
boundary = accept_threshold(response[:, empty_col], counts.astype(float))
q_hat = conformal_rate_bound(response, clusters, pool, pool.holdout, cfg)
print(
    f"q_hat={q_hat:.4f} boundary={boundary:.4f} clusters used={np.count_nonzero(counts)}"
)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. true residue x learned cluster
used = np.flatnonzero(counts)
confusion = np.zeros((9, len(used)), dtype=int)
for r, c in zip(residue, clusters):
    confusion[r, np.searchsorted(used, c)] += 1
# plot_clustering_results assumes a square matrix; this one is 9 x (clusters used).
plt.sca(axes[0, 0])
plt.imshow(confusion / confusion.sum(1, keepdims=True), aspect="auto", cmap="viridis")
plt.colorbar(label="fraction of the residue's prefixes")
plt.xlabel("learned cluster (by index among used)")
plt.ylabel("true residue (sum % 9)")
plt.title("true residue vs learned cluster")

# 2. response matrix: does each cluster have a distinct row?
plt.sca(axes[0, 1])
order = used[np.argsort(-counts[used])]
plt.imshow(response[order], aspect="auto", vmin=0, vmax=1, cmap="RdBu_r")
plt.colorbar(label="predicted accept rate")
plt.xlabel("suffix")
plt.ylabel("cluster (by size)")
plt.title("response rows R[s][v] -- distinct rows = distinct states")

# 3. how concentrated is each (cluster, symbol) successor distribution?
transition = torch.einsum("ps,cpt->cst", current, successors).numpy()
shares = []
for s in used:
    for c in (0, 1):
        row = transition[c, s]
        if row.sum() > 0:
            shares.append(np.sort(row / row.sum())[::-1][0])
plt.sca(axes[1, 0])
plt.hist(shares, bins=20, range=(0, 1), edgecolor="black")
plt.axvline(0.8, color="red", ls="--", label="concentrated (argmax safe)")
plt.xlabel("top-1 share of successor distribution")
plt.ylabel("(cluster, symbol) pairs")
plt.title(f"transition concentration -- median {np.median(shares):.2f}")
plt.legend()

# 4. where do the predicted rates sit relative to the conformal band?
plt.sca(axes[1, 1])
flat = response[used].ravel()
plt.hist(flat, bins=50, range=(0, 1), edgecolor="black")
for edge, label in (
    (boundary - q_hat, "boundary - q̂"),
    (boundary + q_hat, "boundary + q̂"),
):
    plt.axvline(edge, color="red", ls="--", label=label)
plt.axvline(boundary, color="black", label="boundary")
undecided = np.mean(np.abs(flat - boundary) < q_hat)
plt.xlabel("predicted accept rate")
plt.title(f"conformal band: {undecided:.1%} of cells undecided (q̂={q_hat:.3f})")
plt.legend()

plt.tight_layout()
plt.savefig(f"{OUT}/modulo9_diagnosis.png", dpi=110)
print(f"saved {OUT}/modulo9_diagnosis.png")

# Text summary of what the confusion says.
print("\nresidue -> cluster concentration:")
for r in range(9):
    row = confusion[r]
    if row.sum():
        print(
            f"  residue {r}: {row.max() / row.sum():.2f} in its modal cluster, spread over {np.count_nonzero(row)} clusters"
        )
