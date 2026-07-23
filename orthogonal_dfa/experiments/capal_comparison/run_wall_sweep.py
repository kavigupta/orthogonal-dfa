#!/usr/bin/env python3
"""Experiment 3: full CAPAL hyperparameter sweep across every benchmark cell.

A full factorial over CAPAL's three real knobs -- `max_same_samples` (evidence
per pairwise test), `suffix_pool_len_max` (suffix pool length), and `alpha`
(comparison significance) -- run on every benchmark cell (target x noise) for
three seeds. For each cell this tells a *cell-specific structural wall* (fails
at every config and seed) from a *default-config artifact* (some config or seed
cracks it), and whether the modulo-9 wall is "one hard DFA at high noise" or
"high noise is bad for everything".

`max_same_samples` is capped at 240 on purpose: it is the entire runtime cost
(m=60 ~2s, m=240 ~19s, m=480 ~146s with minutes-long outliers), and section 5 +
the preliminary m=480 sweep already show larger m adds no convergence under
persistent noise. `max_iters` is fixed at 50 -- section 3 showed the hypothesis
reaches a fixed point well before then. Section 10's un-exposed deep-enumeration
knobs (millions of queries) stay out of scope.

Example:
    python -m orthogonal_dfa.experiments.capal_comparison.run_wall_sweep \
        --etas 0.05 0.10 0.20 0.30 --seeds 0 1 2
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

DEFAULT_ETAS = [0.05, 0.10, 0.20, 0.30]
DEFAULT_SEEDS = [0, 1, 2]

#: Full factorial over the three knobs that actually move CAPAL. m is capped at
#: 240 (see module docstring); max_iters fixed at 50.
M_VALUES = [60, 240]
POOL_VALUES = [8, 24]
ALPHA_VALUES = [1e-3, 0.05]
CONFIGS: List[Tuple[str, Dict[str, Any]]] = [
    (
        f"m={m},pool={p},alpha={a}",
        dict(max_same_samples=m, suffix_pool_len_max=p, alpha=a, max_iters=50),
    )
    for m, p, a in itertools.product(M_VALUES, POOL_VALUES, ALPHA_VALUES)
]

#: Section 10's "matched query budget" probe: uncap the suffix-enumeration knobs
#: so SAMESTATE draws thousands of long suffixes per pair, pushing CAPAL's
#: distinct-query count up to millions (E-L*'s range). If accuracy still stalls,
#: the wall is the pairwise-test shape, not the number of labels. Slow: minutes
#: per run. Run against eta=0.30 only, where every cell walls at capped budget.
MATCHED_BUDGET_CONFIGS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "enum=8,extra=16,pool=16,m=2000",
        dict(
            max_same_samples=2000,
            suffix_pool_len_max=16,
            enum_depth=8,
            extra_len_max=16,
            max_iters=15,
        ),
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--etas", nargs="+", type=float, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument(
        "--targets", nargs="+", default=None, help="Restrict to these benchmarks."
    )
    ap.add_argument(
        "--matched-budget",
        action="store_true",
        help="Section-10 mode: uncap suffix enumeration to match E-L*'s query "
        "budget (slow); defaults to eta=0.30 and its own output file.",
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

    sweep_configs = MATCHED_BUDGET_CONFIGS if args.matched_budget else CONFIGS
    etas = args.etas or ([0.30] if args.matched_budget else DEFAULT_ETAS)
    experiment = "matched_budget" if args.matched_budget else "wall_sweep"
    default_out = REPO_ROOT / "data" / "capal" / f"{experiment}.json"

    benchmarks = our_benchmarks()
    if args.targets:
        by = {b.name: b for b in benchmarks}
        missing = set(args.targets) - by.keys()
        if missing:
            raise SystemExit(f"unknown target(s): {sorted(missing)}")
        benchmarks = [by[n] for n in args.targets]

    out_path = Path(args.out) if args.out else default_out
    cells: List[Cell] = []
    config = {
        "etas": list(etas),
        "seeds": list(args.seeds),
        "benchmarks": [b.name for b in benchmarks],
        "configs": [label for label, _ in sweep_configs],
        "matched_budget": args.matched_budget,
        "with_elstar": args.with_elstar,
    }

    description = (
        "CAPAL at a matched query budget (uncapped suffix enumeration) on the "
        "eta=0.30 wall cells: does hitting E-L*'s distinct-query count break "
        "the wall?"
        if args.matched_budget
        else (
            "CAPAL hyperparameter sweep (max_same_samples x pool x alpha x seeds) "
            "across every benchmark cell, to separate cell-specific structural "
            "walls from default-config artifacts."
        )
    )

    def flush() -> None:
        write_experiment(
            out_path,
            experiment=experiment,
            generated_by="orthogonal_dfa.experiments.capal_comparison.run_wall_sweep",
            description=description,
            config=config,
            cells=cells,
        )

    total = len(benchmarks) * len(etas) * len(sweep_configs) * len(args.seeds)
    done = 0
    for b in benchmarks:
        words = eval_words(b.symbols)
        truth = b.truth()
        for eta in etas:
            for (label, kwargs), seed in itertools.product(sweep_configs, args.seeds):
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
