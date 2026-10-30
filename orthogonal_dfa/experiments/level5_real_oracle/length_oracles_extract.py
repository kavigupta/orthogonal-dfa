# pylint: disable=invalid-name,missing-function-docstring,too-many-locals,consider-using-with
"""Everything length_oracles_figure.py draws from, as one JSON of counts and DFA tables, so
rendering needs neither the runs nor this repo.

  python orthogonal_dfa/experiments/level5_real_oracle/length_oracles_extract.py \
      LABELS.npz OUT.json RUN_DIR...

RUN_DIR names start with its oracle (saw_s0, flat_s3, ...).  Strings are split into halves
(fit = the first half, read = the rest).  Counts are keyed by half, number of stops (the
compiled length is 36 + 2 x stops) and frame pattern (which of frames 0, 1, 2 a stop starts
in, "C" closed); "accepts" counts oracle accepts.  For each restart's DFA, end_counts adds the
state its strings end in, which fixes every per-string quantity a figure needs."""

import hashlib
import json
import os
import pickle
import re
import sys

import numpy as np

PATTERNS = ["...", "..C", ".C.", "C..", ".CC", "C.C", "CC.", "CCC"]
HALVES = ["fit", "read"]


def patterns(S):
    pos = np.zeros(len(S), int)
    closed = np.zeros((len(S), 3), bool)
    for t in range(S.shape[1]):
        stop = S[:, t] < 3
        closed[np.arange(len(S))[stop], pos[stop] % 3] = True
        pos += np.where(stop, 3, 1)
    names = np.array(["".join("C" if c else "." for c in row) for row in closed])
    return np.array([PATTERNS.index(p) for p in names])


def table(dfa):
    states = sorted(dfa.states, key=str)
    index = {q: i for i, q in enumerate(states)}
    return dict(
        transitions=[[index[dfa.transitions[q][c]] for c in range(5)] for q in states],
        start=index[dfa.initial_state],
        accepting=[int(q in dfa.final_states) for q in states],
    )


def ends(T, start, S):
    q = np.full(len(S), start)
    for t in range(S.shape[1]):
        q = T[q, S[:, t]]
    return q


def tally(keys, shape, weights=None):
    flat = np.ravel_multi_index(keys, shape)
    return np.bincount(flat, weights=weights, minlength=int(np.prod(shape))).reshape(
        shape
    )


def main():
    labels_path, out, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
    z = np.load(labels_path)
    S = z["s"].astype(np.int64)
    half = (np.arange(len(S)) >= len(S) // 2).astype(int)
    stops = (S < 3).sum(1)
    pat = patterns(S)
    shape = (2, S.shape[1] + 1, len(PATTERNS))
    keys = (half, stops, pat)
    n = tally(keys, shape)
    acc = {k: tally(keys, shape, z[k].astype(float)) for k in ("saw", "flat")}
    oracle_counts = [
        dict(
            half=HALVES[h],
            stops=int(k),
            pattern=PATTERNS[p],
            n=int(n[h, k, p]),
            saw_accepts=int(acc["saw"][h, k, p]),
            flat_accepts=int(acc["flat"][h, k, p]),
        )
        for h, k, p in zip(*np.nonzero(n))
    ]

    restarts, run_rows = [], []
    for run in runs:
        name = os.path.basename(run.rstrip("/"))
        kind, seed = re.match(r"(saw|flat)_s(\d+)", name).groups()
        y = z[kind].astype(float)
        log = open(f"{run}/log").read().replace("\r", "\n")
        picks = [
            float(x) for x in re.findall(r"restart \d: .*held-out phi ([0-9.]+)", log)
        ]
        summary = json.load(open(f"{run}/summary.json"))
        for i, pick in enumerate(picks):
            d = pickle.load(open(f"{run}/search_{i}.pkl", "rb"))
            dfa = table(d["dfa"])
            T = np.array(dfa["transitions"])
            q = ends(T, dfa["start"], S)
            cshape = (2, len(T), S.shape[1] + 1, len(PATTERNS))
            ckeys = (half, q, stops, pat)
            cn, ca = tally(ckeys, cshape), tally(ckeys, cshape, y)
            restarts.append(
                dict(
                    oracle=kind,
                    seed=int(seed),
                    restart=i,
                    kept=pick == max(picks),
                    pick_score=pick,
                    search_score=d["best_score"],
                    ending=d["ending"],
                    probes=d["probes"],
                    dfa=dfa,
                    end_counts=[
                        dict(
                            half=HALVES[h],
                            state=int(s),
                            stops=int(k),
                            pattern=PATTERNS[p],
                            n=int(cn[h, s, k, p]),
                            accepts=int(ca[h, s, k, p]),
                        )
                        for h, s, k, p in zip(*np.nonzero(cn))
                    ],
                )
            )
        families = [json.loads(l) for l in open(f"{run}/diag.jsonl") if '"family"' in l]
        run_rows.append(
            dict(
                oracle=kind,
                seed=int(seed),
                commit=summary["commit"],
                hours=summary["hours"],
                family_size=families[-1]["size"],
                family_fnr=families[-1]["fnr"],
                test_halves_phi=summary["best_phi"],
            )
        )
        print(name, len(picks), "restarts", flush=True)

    json.dump(
        dict(
            strings=dict(
                n=len(S),
                symbols=S.shape[1],
                alphabet=["TAG", "TAA", "TGA", "N", "N"],
                labels_sha256=hashlib.sha256(
                    open(labels_path, "rb").read()
                ).hexdigest(),
                note="36 symbols drawn with seed 78, wildcards lifted with seed 0; "
                "fit = first half, read = second",
            ),
            oracles=dict(
                saw="one threshold, the median at compiled length 40",
                flat="each compiled length 35-85 at its own median",
            ),
            oracle_counts=oracle_counts,
            runs=run_rows,
            restarts=restarts,
        ),
        open(out, "w"),
        separators=(",", ":"),
    )


if __name__ == "__main__":
    main()
