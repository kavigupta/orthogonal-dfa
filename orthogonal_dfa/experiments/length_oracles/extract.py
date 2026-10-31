"""The runs and both oracles' labels as one JSON of counts and DFA tables, written to
``data/neural/length_oracles/length_oracles_data.json``.

    python -m orthogonal_dfa.experiments.length_oracles.extract RUN_DIR...

Strings are split into halves (fit = the first half, read = the rest).  Counts are keyed
by half, number of stops (the compiled length is 36 + 2 x stops) and frame pattern (which
of frames 0, 1, 2 a stop starts in, "C" closed); "accepts" counts oracle accepts.  For each
restart's DFA, end_counts adds the state its strings end in."""

import hashlib
import json
import sys

import numpy as np

from ..capal_comparison.core import REPO_ROOT
from .label import COUNT, SEED, labels, strings
from .oracles import LEN_HI, LEN_LO, SAW_LENGTH, SYMBOLS

PATTERNS = ["...", "..C", ".C.", "C..", ".CC", "C.C", "CC.", "CCC"]
HALVES = ["fit", "read"]
OUT = REPO_ROOT / "data" / "neural" / "length_oracles" / "length_oracles_data.json"


def patterns(S):
    pos = np.zeros(len(S), int)
    closed = np.zeros((len(S), 3), bool)
    for t in range(S.shape[1]):
        stop = S[:, t] < 3
        closed[np.arange(len(S))[stop], pos[stop] % 3] = True
        pos += np.where(stop, 3, 1)
    names = np.array(["".join("C" if c else "." for c in row) for row in closed])
    return np.array([PATTERNS.index(p) for p in names])


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


def oracle_counts(z, keys):
    """``keys`` is each string's (half, stops, pattern)."""
    shape = (2, SYMBOLS + 1, len(PATTERNS))
    n = tally(keys, shape)
    acc = {k: tally(keys, shape, z[k].astype(float)) for k in ("saw", "flat")}
    return [
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


def end_counts(dfa, S, y, keys):
    T = np.array(dfa["transitions"])
    half, stops, pat = keys
    shape = (2, len(T), SYMBOLS + 1, len(PATTERNS))
    keys = (half, ends(T, dfa["start"], S), stops, pat)
    n, accepts = tally(keys, shape), tally(keys, shape, y)
    return [
        dict(
            half=HALVES[h],
            state=int(s),
            stops=int(k),
            pattern=PATTERNS[p],
            n=int(n[h, s, k, p]),
            accepts=int(accepts[h, s, k, p]),
        )
        for h, s, k, p in zip(*np.nonzero(n))
    ]


def main():
    runs = sys.argv[1:]
    z = {kind: labels(kind, COUNT) for kind in ("saw", "flat")}
    S = np.array(strings(COUNT), dtype=np.int64)
    assert S.shape[1] == SYMBOLS, S.shape
    keys = ((np.arange(len(S)) >= len(S) // 2).astype(int), (S < 3).sum(1), patterns(S))

    run_rows, restarts = [], []
    for run in runs:
        with open(f"{run}/run.json") as f:
            row = json.load(f)
        with open(f"{run}/restarts.json") as f:
            searches = json.load(f)
        run_rows.append(row)
        picks = [r["pick_score"] for r in searches]
        y = z[row["oracle"]].astype(float)
        for r in searches:
            restarts.append(
                dict(
                    oracle=row["oracle"],
                    seed=row["seed"],
                    restart=r["restart"],
                    # As counterexample_synthesis keeps it: the first best.
                    kept=r["restart"] == int(np.argmax(picks)),
                    pick_score=r["pick_score"],
                    search_score=r["search_score"],
                    ending=r["ending"],
                    probes=r["probes"],
                    dfa=r["dfa"],
                    end_counts=end_counts(r["dfa"], S, y, keys),
                )
            )
        print(run, len(searches), "restarts", flush=True)

    labels_sha256 = hashlib.sha256(z["saw"].tobytes() + z["flat"].tobytes()).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(
            dict(
                strings=dict(
                    n=len(S),
                    symbols=SYMBOLS,
                    alphabet=["TAG", "TAA", "TGA", "N", "N"],
                    labels_sha256=labels_sha256,
                    note=f"drawn with seed {SEED}, wildcards lifted with seed 0; "
                    "fit = first half, read = second",
                ),
                oracles=dict(
                    saw=f"one threshold, the median at compiled length {SAW_LENGTH}",
                    flat=f"each compiled length {LEN_LO}-{LEN_HI} at its own median",
                ),
                oracle_counts=oracle_counts(z, keys),
                runs=run_rows,
                restarts=restarts,
            ),
            f,
            separators=(",", ":"),
        )


if __name__ == "__main__":
    main()
