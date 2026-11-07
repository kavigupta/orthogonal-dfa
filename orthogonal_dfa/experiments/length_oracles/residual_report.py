"""Each residual run's kept DFA, alone on the residual labels and in product with its
seed's first stage on the flat labels, scored as held_out_phi scores them: accepting states
fit on the first half of the saved strings, phi read on the second.  Writes the minimized
products to residual_products.json.

    python -m orthogonal_dfa.experiments.length_oracles.residual_report"""

import json
from collections import Counter

import numpy as np

from orthogonal_dfa.l_star.dfa_utils import phi_maximizing_states

from . import label
from .residual import RUNS, end_states, first_stage, offsets, saved_centred_scores

PRODUCTS = RUNS.parent / "residual_products.json"


def accepting(cells, y):
    half = len(y) // 2
    counts = Counter(cells[:half].tolist())
    accepts = Counter()
    for c, v in zip(cells[:half].tolist(), y[:half]):
        accepts[c] += int(v)
    return phi_maximizing_states(counts, accepts)


def phi(cells, y):
    half = len(y) // 2
    verdicts = np.isin(cells[half:], list(accepting(cells, y)))
    return float(np.corrcoef(verdicts, y[half:])[0, 1])


def product(a, b):
    """Table of the reachable pairs of states of ``a`` and ``b``."""
    ta, tb = a["transitions"], b["transitions"]
    order = [(a["start"], b["start"])]
    index = {order[0]: 0}
    transitions = []
    while len(transitions) < len(order):
        p, q = order[len(transitions)]
        row = []
        for c in range(len(ta[p])):
            pair = (ta[p][c], tb[q][c])
            if pair not in index:
                index[pair] = len(order)
                order.append(pair)
            row.append(index[pair])
        transitions.append(row)
    return dict(transitions=transitions, start=0)


def minimize(table, accepts):
    """``table`` with accepting states ``accepts``, minimized; states are numbered in
    breadth-first order from the start."""
    t = np.array(table["transitions"])
    block = np.array([int(s in accepts) for s in range(len(t))])
    while True:
        keys = [(block[s], *block[t[s]]) for s in range(len(t))]
        ids = {k: i for i, k in enumerate(dict.fromkeys(keys))}
        refined = np.array([ids[k] for k in keys])
        if len(ids) == len(set(block.tolist())):
            break
        block = refined
    rep = {b: int(np.nonzero(block == b)[0][0]) for b in set(block.tolist())}
    order = [block[table["start"]]]
    i = 0
    while i < len(order):
        for n in block[t[rep[order[i]]]]:
            if n not in order:
                order.append(n)
        i += 1
    number = {b: i for i, b in enumerate(order)}
    return dict(
        transitions=[[number[n] for n in block[t[rep[b]]]] for b in order],
        start=0,
        accepting=[int(rep[b] in accepts) for b in order],
    )


def main():
    if PRODUCTS.exists():
        print(f"{PRODUCTS} exists; nothing to run.", flush=True)
        return
    strings, labels = label.saved_labels()
    flat = labels["flat"].astype(bool)
    scores = saved_centred_scores(0, label.COUNT)
    with open(RUNS) as f:
        restarts = [r for r in json.load(f)["restarts"] if r["oracle"] == "residual"]
    out = []
    for seed in sorted({r["seed"] for r in restarts}):
        stage = first_stage(seed)
        q1 = end_states(stage, strings)
        residual = scores > offsets(stage)[q1]
        kept = max(
            (r for r in restarts if r["seed"] == seed), key=lambda r: r["pick_score"]
        )
        pairs = product(stage, kept["dfa"])
        cells = end_states(pairs, strings)
        dfa = minimize(pairs, accepting(cells, flat))
        q = end_states(dfa, strings)
        print(
            f"seed {seed}: first stage phi {phi(q1, flat):.3f}; kept restart "
            f"{kept['restart']}, {len(kept['dfa']['transitions'])} states; "
            f"residual phi {phi(end_states(kept['dfa'], strings), residual):.3f}; "
            f"product {len(pairs['transitions'])} reachable, "
            f"{len(dfa['transitions'])} minimal states, phi {phi(cells, flat):.3f}"
        )
        out.append(
            dict(
                seed=seed,
                residual_restart=kept["restart"],
                first_stage=stage,
                residual=kept["dfa"],
                product=dfa,
                strings_per_state=np.bincount(
                    q, minlength=len(dfa["accepting"])
                ).tolist(),
                flat_rate_per_state=[
                    float(flat[q == s].mean()) if (q == s).any() else None
                    for s in range(len(dfa["accepting"]))
                ],
            )
        )
    with open(PRODUCTS, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
