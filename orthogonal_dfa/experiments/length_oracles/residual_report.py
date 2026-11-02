"""Each residual run's kept DFA, alone on the residual labels and in product with its
seed's first stage on the flat labels, scored as held_out_phi scores them: accepting states
fit on the first half of the saved strings, phi read on the second.

    python -m orthogonal_dfa.experiments.length_oracles.residual_report"""

import json
from collections import Counter

import numpy as np

from orthogonal_dfa.l_star.dfa_utils import phi_maximizing_states

from . import label
from .residual import RUNS, end_states, first_stage, offsets, saved_centred_scores


def phi(cells, y):
    half = len(y) // 2
    counts = Counter(cells[:half].tolist())
    accepts = Counter()
    for c, v in zip(cells[:half].tolist(), y[:half]):
        accepts[c] += int(v)
    accepting = phi_maximizing_states(counts, accepts)
    verdicts = np.isin(cells[half:], list(accepting))
    return float(np.corrcoef(verdicts, y[half:])[0, 1])


def main():
    strings, labels = label.saved_labels()
    flat = labels["flat"].astype(bool)
    scores = saved_centred_scores(0, label.COUNT)
    with open(RUNS) as f:
        restarts = [r for r in json.load(f)["restarts"] if r["oracle"] == "residual"]
    for seed in sorted({r["seed"] for r in restarts}):
        stage = first_stage(seed)
        q1 = end_states(stage, strings)
        residual = scores > offsets(stage)[q1]
        kept = max(
            (r for r in restarts if r["seed"] == seed), key=lambda r: r["pick_score"]
        )
        q2 = end_states(kept["dfa"], strings)
        product = q1 * len(kept["dfa"]["transitions"]) + q2
        print(
            f"seed {seed}: first stage phi {phi(q1, flat):.3f}; kept restart "
            f"{kept['restart']}, {len(kept['dfa']['transitions'])} states, "
            f"{len(np.unique(product))} product states reached; "
            f"residual phi {phi(q2, residual):.3f}, product phi {phi(product, flat):.3f}"
        )


if __name__ == "__main__":
    main()
