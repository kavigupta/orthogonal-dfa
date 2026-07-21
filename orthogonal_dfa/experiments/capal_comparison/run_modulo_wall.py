#!/usr/bin/env python3
"""Experiment 3: the modulo-9 eta=0.30 wall.

One fixed hard cell -- the mod-9 (allowed={3,6}) DFA at eta=0.30 -- against a
sweep of deliberately resource-heavy CAPAL configurations, plus E-L* on the
same cell for reference. The question is whether CAPAL's failure there is a
budget limit or a structural one, so the load-bearing measurement is
`queries_distinct`: if pouring in more samples does not raise it, the extra
samples carried no information.

Each config is emitted as its own cell, distinguished by `learner_config`.

Example:
    python -m orthogonal_dfa.experiments.capal_comparison.run_modulo_wall
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .core import (
    LEARNER_CAPAL,
    LEARNER_ELSTAR,
    REPO_ROOT,
    Cell,
    eval_words,
    run_capal_cell,
    run_elstar_cell,
    write_experiment,
)
from .targets import FAMILY_OURS, our_benchmarks

DEFAULT_OUT = REPO_ROOT / "data" / "capal" / "modulo_wall.json"

BENCHMARK = "parity_mod9_allowed_3_6"

#: The resource-heavy configs. `suffix_pool_len_max=8` is the CAPAL default;
#: the last row is the only one that raises it, which is the row that matters.
CONFIGS: List[Tuple[str, Dict[str, Any]]] = [
    ("m=60 (default, validation)", dict(max_same_samples=60, max_iters=200)),
    ("m=120,  50 iter", dict(max_same_samples=120, max_iters=50)),
    ("m=240,  50 iter", dict(max_same_samples=240, max_iters=50)),
    ("m=480,  50 iter", dict(max_same_samples=480, max_iters=50)),
    ("m=1000, 50 iter", dict(max_same_samples=1000, max_iters=50)),
    ("m=4000, 15 iter", dict(max_same_samples=4000, max_iters=15)),
    (
        "m=4000 + pool_len_max=14",
        dict(max_same_samples=4000, max_iters=15, suffix_pool_len_max=14),
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eta", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-pos", type=int, default=10)
    ap.add_argument("--k-neg", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--capal-dir", default=None)
    ap.add_argument(
        "--skip-elstar",
        action="store_true",
        help="Skip the E-L* reference cell (it is the slow one).",
    )
    args = ap.parse_args()

    if args.capal_dir:
        os.environ["ORTHO_CAPAL_DIR"] = args.capal_dir

    bench = {b.name: b for b in our_benchmarks()}[BENCHMARK]
    words = eval_words(bench.symbols)
    truth = bench.truth()
    out_path = Path(args.out) if args.out else DEFAULT_OUT

    cells: List[Cell] = []
    config = {
        "eta": args.eta,
        "seed": args.seed,
        "benchmark": BENCHMARK,
        "K_pos": args.k_pos,
        "K_neg": args.k_neg,
        "configs": [label for label, _ in CONFIGS],
    }

    def flush() -> None:
        write_experiment(
            out_path,
            experiment="modulo_wall",
            generated_by=(
                "orthogonal_dfa.experiments.capal_comparison.run_modulo_wall"
            ),
            description=(
                "CAPAL resource-heavy config sweep on the modulo-9 eta=0.30 "
                "cell, with an E-L* reference run on the same cell."
            ),
            config=config,
            cells=cells,
        )

    for label, kwargs in CONFIGS:
        print(f"[capal] {label}", flush=True)
        cell = run_capal_cell(
            bench.target,
            benchmark=BENCHMARK,
            family=FAMILY_OURS,
            eta=args.eta,
            seed=args.seed,
            words=words,
            truth=truth,
            alphabet=bench.alphabet,
            k_pos=args.k_pos,
            k_neg=args.k_neg,
            **kwargs,
        )
        # The label is what the report's row header shows; the full config is
        # already in learner_config.
        cell.learner_config["label"] = label
        cells.append(cell)
        print(
            f"   -> states={cell.learned_states} acc={cell.accuracy} "
            f"conv={cell.converged} distinct={cell.queries_distinct} "
            f"({cell.seconds:.1f}s)",
            flush=True,
        )
        flush()

    if not args.skip_elstar:
        print(f"[{LEARNER_ELSTAR}] reference run", flush=True)
        cell = run_elstar_cell(
            bench.oracle_creator,
            benchmark=BENCHMARK,
            family=FAMILY_OURS,
            eta=args.eta,
            seed=args.seed,
            symbols=bench.symbols,
            words=words,
            truth=truth,
            target_states=bench.target_states,
        )
        cell.learner_config["label"] = "E-L* reference"
        cells.append(cell)
        print(
            f"   -> states={cell.learned_states} acc={cell.accuracy} "
            f"distinct={cell.queries_distinct} ({cell.seconds:.1f}s)",
            flush=True,
        )
        flush()

    print(f"\nWrote {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
