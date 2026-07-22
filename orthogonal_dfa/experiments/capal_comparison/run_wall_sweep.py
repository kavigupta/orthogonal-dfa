#!/usr/bin/env python3
"""Experiment 3: CAPAL hyperparameter sweep across every benchmark cell.

Generalises the old single-cell "modulo wall" probe. For each cell (target x
noise) it sweeps CAPAL's evidence knob `max_same_samples` over several seeds,
so we can tell a *cell-specific structural wall* (fails at every config and
seed) from a *default-config artifact* (a bigger m or a luckier seed cracks
it). The modulo-9 eta=0.30 cell is one row here now, not the whole experiment;
the open question is whether the wall is "one hard DFA at high noise" or "high
noise is bad for everything".

Deliberately preliminary: it omits the expensive dimensions -- `max_same_samples
>= 1000` (minutes per run) and section 10's un-exposed deep-enumeration knobs
(millions of queries) -- so we get a first read across all cells cheaply and
decide where to spend the slow budget. Seeds are run as a matter of course, not
swept as a knob.

Example:
    python -m orthogonal_dfa.experiments.capal_comparison.run_wall_sweep \
        --etas 0.20 0.30 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .core import (
    REPO_ROOT,
    Cell,
    eval_words,
    run_capal_cell,
    run_elstar_cell,
    write_experiment,
)
from .targets import our_benchmarks

DEFAULT_OUT = REPO_ROOT / "data" / "capal" / "wall_sweep.json"

DEFAULT_ETAS = [0.20, 0.30]
DEFAULT_SEEDS = [0, 1, 2]

#: CAPAL knob settings swept per (cell, eta, seed). Kept to the cheap end on
#: purpose: max_same_samples up to 480 (m=1000/4000 are the minutes-per-run
#: rows). max_iters is fixed at 50 -- section 3 showed the hypothesis reaches a
#: fixed point well before then, so more iterations never help.
CONFIGS: List[Tuple[str, Dict[str, Any]]] = [
    ("m=60 (default)", dict(max_same_samples=60, max_iters=50)),
    ("m=240", dict(max_same_samples=240, max_iters=50)),
    ("m=480", dict(max_same_samples=480, max_iters=50)),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--etas", nargs="+", type=float, default=DEFAULT_ETAS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument(
        "--targets", nargs="+", default=None, help="Restrict to these benchmarks."
    )
    ap.add_argument(
        "--with-elstar",
        action="store_true",
        help="Also run one E-L* reference per (cell, eta). Slow.",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--capal-dir", default=None)
    args = ap.parse_args()

    if args.capal_dir:
        os.environ["ORTHO_CAPAL_DIR"] = args.capal_dir

    benchmarks = our_benchmarks()
    if args.targets:
        by = {b.name: b for b in benchmarks}
        missing = set(args.targets) - by.keys()
        if missing:
            raise SystemExit(f"unknown target(s): {sorted(missing)}")
        benchmarks = [by[n] for n in args.targets]

    out_path = Path(args.out) if args.out else DEFAULT_OUT
    cells: List[Cell] = []
    config = {
        "etas": list(args.etas),
        "seeds": list(args.seeds),
        "benchmarks": [b.name for b in benchmarks],
        "configs": [label for label, _ in CONFIGS],
        "with_elstar": args.with_elstar,
    }

    def flush() -> None:
        write_experiment(
            out_path,
            experiment="wall_sweep",
            generated_by="orthogonal_dfa.experiments.capal_comparison.run_wall_sweep",
            description=(
                "CAPAL hyperparameter sweep (max_same_samples x seeds) across "
                "every benchmark cell, to separate cell-specific structural "
                "walls from default-config artifacts."
            ),
            config=config,
            cells=cells,
        )

    total = len(benchmarks) * len(args.etas) * len(CONFIGS) * len(args.seeds)
    done = 0
    for b in benchmarks:
        words = eval_words(b.symbols)
        truth = b.truth()
        for eta in args.etas:
            for (label, kwargs), seed in itertools.product(CONFIGS, args.seeds):
                done += 1
                print(
                    f"[{done}/{total}] {b.name} eta={eta:.2f} {label} seed={seed}",
                    flush=True,
                )
                cell = run_capal_cell(
                    b.target,
                    benchmark=b.name,
                    family=b.family,
                    eta=eta,
                    seed=seed,
                    words=words,
                    truth=truth,
                    alphabet=b.alphabet,
                    **kwargs,
                )
                cell.learner_config["label"] = label
                cells.append(cell)
                print(
                    f"   -> acc={cell.accuracy} conv={cell.converged} "
                    f"states={cell.learned_states}/{b.target_states} "
                    f"distinct={cell.queries_distinct} ({cell.seconds:.1f}s)",
                    flush=True,
                )
                flush()

            if args.with_elstar:
                print(f"[E-L*] {b.name} eta={eta:.2f} reference", flush=True)
                cell = run_elstar_cell(
                    b.oracle_creator,
                    benchmark=b.name,
                    family=b.family,
                    eta=eta,
                    seed=0,
                    symbols=b.symbols,
                    words=words,
                    truth=truth,
                    target_states=b.target_states,
                )
                cell.learner_config["label"] = "E-L* reference"
                cells.append(cell)
                flush()

    flush()
    print(f"\nWrote {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
