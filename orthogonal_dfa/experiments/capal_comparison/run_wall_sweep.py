#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""
Experiment 3, the hyperparameter sweep, and experiment 4, the matched budget.

The sweep runs every combination of max_same_samples, suffix_pool_len_max and
alpha on every benchmark cell, for three seeds. Its grid starts at upstream's
own benchmark settings and goes up along each axis.

m stops at 240 because it is essentially the whole runtime cost (m=80 ~3s,
m=240 ~19s, m=480 ~146s) and a preliminary m=480 sweep added no convergence.

Three seeds per cell, against one for the head-to-head experiments.

--matched-budget runs CAPAL with exactly the membership queries E-L* spent on
the same cell, at eta=0.30. It is slow because it runs CAPAL with a large query budget,
and so is not a sweep. A saved cell is reused only while E-L*'s spend is unchanged.

    python -m orthogonal_dfa.experiments.capal_comparison.run_wall_sweep --processes 4
    python -m orthogonal_dfa.experiments.capal_comparison.run_wall_sweep --processes 4 --matched-budget
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import multiprocessing
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .benchmark import Benchmark
from .core import REPO_ROOT, Cell, eval_words, write_experiment
from .our_targets import OUR_BENCHMARK_NAMES, our_benchmarks
from .sweep import capal_cell, reusable_cells

DEFAULT_ETAS = [0.05, 0.10, 0.20, 0.30]
DEFAULT_SEEDS = [0, 1, 2]

#: Anchored at upstream's benchmark settings (m=80, pool=10, alpha=1e-3).
M_VALUES = [80, 240]
POOL_VALUES = [10, 24]
ALPHA_VALUES = [1e-3, 0.05]
CONFIGS: List[Tuple[str, Dict[str, Any]]] = [
    (
        f"m={m},pool={p},alpha={a}",
        dict(max_same_samples=m, suffix_pool_len_max=p, alpha=a),
    )
    for m, p, a in itertools.product(M_VALUES, POOL_VALUES, ALPHA_VALUES)
]

MATCHED_BUDGET_CONFIGS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "matched,enum=8,extra=16,pool=24,m=2000",
        dict(
            max_same_samples=2000,
            suffix_pool_len_max=24,
            enum_depth=8,
            extra_len_max=16,
            max_iters=500,
        ),
    ),
]


def cell_key(cell: dict) -> tuple:
    return (
        cell["benchmark"],
        cell["learner_config"]["label"],
        cell["eta"],
        cell["seed"],
    )


def is_current(cached: Optional[dict], budget: Optional[int]) -> bool:
    return cached is not None and cached["learner_config"].get("query_budget") == budget


def elstar_budgets() -> Dict[Tuple[str, float], int]:
    """
    E-L*'s membership queries per (benchmark, eta), from experiment 2.

    Cells E-L* was not run on have no spend to match, and so carry no budget.
    """
    path = REPO_ROOT / "data" / "capal" / "our_benchmarks.json"
    if not path.exists():
        raise SystemExit(
            f"{path} is missing; run run_our_bench first -- the matched budget "
            "is defined by what E-L* spent."
        )
    cells = json.loads(path.read_text())["cells"]
    return {
        (c["benchmark"], c["eta"]): c["queries_total"]
        for c in cells
        if c["learner"] == "E-L*" and c.get("queries_total") is not None
    }


def select_configs(
    configs: List[Tuple[str, Dict[str, Any]]], labels: Optional[Sequence[str]]
) -> List[Tuple[str, Dict[str, Any]]]:
    """Apply `--configs`, in the order the caller named them."""
    if not labels:
        return list(configs)
    by_label = dict(configs)
    missing = sorted(set(labels) - by_label.keys())
    if missing:
        raise SystemExit(f"unknown config(s): {missing}\nknown: {sorted(by_label)}")
    return [(label, by_label[label]) for label in labels]


def main() -> None:  # pylint: disable=too-many-statements
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--etas", nargs="+", type=float, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument(
        "--targets", nargs="+", default=None, help="Restrict to these benchmarks."
    )
    ap.add_argument(
        "--matched-budget",
        action="store_true",
        help="Experiment 4: give CAPAL E-L*'s query budget instead of sweeping "
        "(slow); defaults to eta=0.30 and its own output file.",
    )
    ap.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Restrict to these config labels (see CONFIGS / MATCHED_BUDGET_CONFIGS).",
    )
    ap.add_argument("--processes", type=int, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sweep_configs = select_configs(
        MATCHED_BUDGET_CONFIGS if args.matched_budget else CONFIGS, args.configs
    )
    etas = args.etas or ([0.30] if args.matched_budget else DEFAULT_ETAS)
    experiment = "matched_budget" if args.matched_budget else "wall_sweep"
    default_out = REPO_ROOT / "data" / "capal" / f"{experiment}.json"
    out_path = Path(args.out) if args.out else default_out

    names = args.targets or OUR_BENCHMARK_NAMES
    missing = set(names) - set(OUR_BENCHMARK_NAMES)
    if missing:
        raise SystemExit(f"unknown target(s): {sorted(missing)}")

    labels = [label for label, _ in sweep_configs]
    budgets = elstar_budgets() if args.matched_budget else {}
    grid = [
        (name, label, eta, seed)
        for name, label, eta, seed in itertools.product(names, labels, etas, args.seeds)
        if not args.matched_budget or (name, eta) in budgets
    ]
    previous = reusable_cells(out_path, cell_key)
    if all(is_current(previous.get(k), budgets.get((k[0], k[2]))) for k in grid):
        print(f"{out_path} has every cell; nothing to run.", flush=True)
        return

    cells: Dict[tuple, Cell] = {k: Cell(**c) for k, c in previous.items()}
    config = {
        "etas": list(etas),
        "seeds": list(args.seeds),
        "benchmarks": list(names),
        "configs": labels,
        "matched_budget": args.matched_budget,
    }

    description = (
        "CAPAL given exactly the membership queries E-L* spent on the same "
        "cell, at eta=0.30."
        if args.matched_budget
        else (
            "CAPAL hyperparameter sweep (max_same_samples x pool x alpha x "
            "seeds) across every benchmark cell."
        )
    )

    def write(keys: Sequence[tuple], complete: bool) -> None:
        write_experiment(
            out_path,
            experiment=experiment,
            generated_by="orthogonal_dfa.experiments.capal_comparison.run_wall_sweep",
            description=description,
            config=config,
            cells=[cells[k] for k in keys],
            complete=complete,
        )

    kwargs_of = dict(sweep_configs)
    tasks = [
        (name, label, kwargs_of[label], eta, seed, budgets.get((name, eta)))
        for name, label, eta, seed in grid
        if not is_current(
            previous.get((name, label, eta, seed)), budgets.get((name, eta))
        )
    ]
    print(f"Reusing {len(grid) - len(tasks)} of {len(grid)} cells.", flush=True)
    # Until the final write, which is the grid alone, carry saved cells outside the
    # grid: this file is their only copy.
    with multiprocessing.Pool(args.processes) as pool:
        for i, cell in enumerate(pool.imap_unordered(_run_cell, tasks), start=1):
            cells[cell_key(asdict(cell))] = cell
            print(
                f"[{i}/{len(tasks)}] {cell.benchmark} eta={cell.eta:.2f} "
                f"{cell.learner_config['label']} seed={cell.seed}\n"
                f"   -> acc={cell.accuracy} conv={cell.converged} "
                f"states={cell.learned_states} "
                f"mq={cell.queries_total} eq={cell.equivalence_queries} "
                f"({cell.seconds:.1f}s)",
                flush=True,
            )
            write(list(cells), complete=False)

    write(grid, complete=True)
    print(f"\nWrote {out_path} ({len(grid)} cells)")


@functools.cache
def _our_benchmarks() -> Dict[str, Benchmark]:
    return {b.name: b for b in our_benchmarks()}


def _run_cell(task: tuple) -> Cell:
    name, label, kwargs, eta, seed, budget = task
    b = _our_benchmarks()[name]
    cell = capal_cell(
        b,
        eta=eta,
        seed=seed,
        words=eval_words(b.symbols),
        truth=b.truth(),
        query_budget=budget,
        **kwargs,
    )
    cell.learner_config["label"] = label
    if budget is not None:
        cell.learner_config["query_budget"] = budget
    return cell


if __name__ == "__main__":
    main()
