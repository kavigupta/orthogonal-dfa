#!/usr/bin/env python3
"""Reproducer for `data/capal_findings.md` section 5 ("Modulo η=0.30: the wall").

Runs the official CAPAL learner on the modulo-9 (allowed={3,6}) DFA at η=0.30
across the resource-heavy configs from the findings doc, and reports, for each:
states, accuracy, converged, wall time, and the number of *distinct* membership
queries (`len(mq.cache)` -- CAPAL's PersistentNoisyMQ caches every string, so
this is its true oracle cost).

The default configs match the reproducible §5 table (seed=0, K_pos=K_neg=10,
50 iterations for the m=120..1000 rows and 15 for the m=4000 rows). The harness
is validated by the m=60 default cell reproducing §2 exactly (20 states, 12 240
distinct queries).

No dependency on this repo's orthogonal_dfa package; the only in-repo import is
the sibling `capal_upstream` module in this folder. The other requirement is a
clone of github.com/lkwargs/CAPAL at the commit pinned in capal_upstream.py,
checked out clean -- a wrong commit or a dirty tree is a hard error, since §5's
numbers are only reproducible against that commit. Defaults to `../capal`
relative to the repo root; override with --capal-dir.

Example:
    python scripts/capal_modulo_wall_queries.py \
        --csv data/capal_modulo_wall_queries.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -- upstream import ----------------------------------------------------------


sys.path.insert(0, str(Path(__file__).resolve().parent))
from capal_upstream import (  # noqa: E402
    DEFAULT_CAPAL_DIR,
    PINNED_COMMIT,
    import_capal,
)


# -- target + accuracy --------------------------------------------------------


def build_modulo_dfa(M: Any, modulo: int, allowed: List[int]) -> Any:
    """DFA over {'0','1'} accepting iff (count of '1' chars) mod `modulo` is in
    `allowed`. Matches BernoulliParityOracle / the repo's test_modulo."""
    delta = {}
    for q in range(modulo):
        delta[(q, "0")] = q
        delta[(q, "1")] = (q + 1) % modulo
    return M.DFA(
        alphabet=["0", "1"],
        num_states=modulo,
        start=0,
        accept=set(allowed),
        delta=delta,
    )


def accuracy(
    target: Any, learned: Any, *, count: int, max_len: int, seed: int
) -> float:
    """Fraction of random words (length 1..max_len) on which `learned` agrees
    with the noiseless `target`."""
    rng = random.Random(seed)
    sigma = list(target.alphabet)
    ok = 0
    for _ in range(count):
        L = rng.randint(1, max_len)
        w = "".join(rng.choice(sigma) for _ in range(L))
        if bool(learned.run(w)) == bool(target.run(w)):
            ok += 1
    return ok / count


# -- one config ---------------------------------------------------------------


def run_one(
    M: Any,
    *,
    m: int,
    iters: int,
    pool_len: int,
    eta: float,
    seed: int,
    k_pos: int,
    k_neg: int,
    eval_count: int,
    eval_max_len: int,
) -> Dict[str, Any]:
    target = build_modulo_dfa(M, 9, [3, 6])
    cfg = M.LearnerConfig(
        eta=eta,
        seed=seed,
        verbose=False,
        K_pos=k_pos,
        K_neg=k_neg,
        max_same_samples=m,
        max_iters=iters,
        suffix_pool_len_max=pool_len,
    )
    learner = M.CAPALLearner(target=target, cfg=cfg)
    t0 = time.time()
    converged = True
    try:
        dfa = learner.fit()
    except RuntimeError:
        converged = False
        last = getattr(learner, "_last_hyp", None)
        if last is None or last.dfa is None:
            raise
        dfa = last.dfa
    secs = time.time() - t0
    acc = accuracy(
        target, dfa, count=eval_count, max_len=eval_max_len, seed=seed
    )
    # PersistentNoisyMQ.cache holds one entry per distinct queried string.
    distinct_mq = len(getattr(learner.mq, "cache", {}))
    return {
        "states": dfa.num_states,
        "acc": acc,
        "converged": converged,
        "secs": secs,
        "distinct_mq": distinct_mq,
    }


# The §5 configs. `pool_len=8` is the CAPAL default (suffix_pool_len_max).
CONFIGS: List[Tuple[str, Dict[str, int]]] = [
    ("m=120,  50 iter", dict(m=120, iters=50, pool_len=8)),
    ("m=240,  50 iter", dict(m=240, iters=50, pool_len=8)),
    ("m=480,  50 iter", dict(m=480, iters=50, pool_len=8)),
    ("m=1000, 50 iter", dict(m=1000, iters=50, pool_len=8)),
    ("m=4000, 15 iter", dict(m=4000, iters=15, pool_len=8)),
    ("m=4000 + pool_len_max=14", dict(m=4000, iters=15, pool_len=14)),
]


# -- entry point --------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--capal-dir",
        default=None,
        help=f"Clone of github.com/lkwargs/CAPAL, pinned at {PINNED_COMMIT[:7]} "
        f"with a clean tree. Default: {DEFAULT_CAPAL_DIR}",
    )
    ap.add_argument("--eta", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-pos", type=int, default=10)
    ap.add_argument("--k-neg", type=int, default=10)
    ap.add_argument("--eval-samples", type=int, default=5000)
    ap.add_argument("--eval-max-len", type=int, default=40)
    ap.add_argument("--csv", default=None, help="Optional CSV output path.")
    args = ap.parse_args()

    M = import_capal(args.capal_dir)

    header = (
        f"| config (seed={args.seed})          "
        "| states | acc   | conv | time | distinct MQ |"
    )
    print(header, flush=True)
    print("|" + "-" * (len(header) - 2) + "|", flush=True)

    rows: List[Tuple[str, Dict[str, Any]]] = []
    for label, kw in CONFIGS:
        r = run_one(
            M,
            eta=args.eta,
            seed=args.seed,
            k_pos=args.k_pos,
            k_neg=args.k_neg,
            eval_count=args.eval_samples,
            eval_max_len=args.eval_max_len,
            **kw,
        )
        rows.append((label, r))
        print(
            f"| {label:26s} | {r['states']:6d} | {r['acc']:.3f} "
            f"| {'Yes' if r['converged'] else 'No':3s} "
            f"| {int(round(r['secs'])):4d}s | {r['distinct_mq']:11d} |",
            flush=True,
        )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["config", "states", "accuracy", "converged", "seconds", "distinct_mq"]
            )
            for label, r in rows:
                w.writerow(
                    [
                        label,
                        r["states"],
                        f"{r['acc']:.6f}",
                        "Y" if r["converged"] else "N",
                        f"{r['secs']:.2f}",
                        r["distinct_mq"],
                    ]
                )
        print(f"\nWrote {args.csv}", flush=True)


if __name__ == "__main__":
    main()
