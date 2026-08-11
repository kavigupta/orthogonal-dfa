#!/usr/bin/env python3
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
and so is not a sweep.

    python -m orthogonal_dfa.experiments.capal_comparison.run_wall_sweep
    python -m orthogonal_dfa.experiments.capal_comparison.run_wall_sweep --matched-budget
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .core import REPO_ROOT, Cell, eval_words, write_experiment
from .our_targets import our_benchmarks
from .sweep import capal_cell

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
        help="Experiment 4: give CAPAL E-L*'s query budget instead of sweeping "
        "(slow); defaults to eta=0.30 and its own output file.",
    )
    ap.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Restrict to these config labels (see CONFIGS / MATCHED_BUDGET_CONFIGS).",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sweep_configs = select_configs(
        MATCHED_BUDGET_CONFIGS if args.matched_budget else CONFIGS, args.configs
    )
    etas = args.etas or ([0.30] if args.matched_budget else DEFAULT_ETAS)
    experiment = "matched_budget" if args.matched_budget else "wall_sweep"
    default_out = REPO_ROOT / "data" / "capal" / f"{experiment}.json"

    budgets = elstar_budgets() if args.matched_budget else {}
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

    def flush(complete: bool = False) -> None:
        write_experiment(
            out_path,
            experiment=experiment,
            generated_by="orthogonal_dfa.experiments.capal_comparison.run_wall_sweep",
            description=description,
            config=config,
            cells=cells,
            complete=complete,
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
                budget = budgets.get((b.name, eta)) if args.matched_budget else None
                if args.matched_budget and budget is None:
                    print("   skipped: E-L* has no spend recorded here", flush=True)
                    continue
                cell = capal_cell(
                    b,
                    eta=eta,
                    seed=seed,
                    words=words,
                    truth=truth,
                    query_budget=budget,
                    **kwargs,
                )
                cell.learner_config["label"] = label
                cells.append(cell)
                print(
                    f"   -> acc={cell.accuracy} conv={cell.converged} "
                    f"states={cell.learned_states}/{b.target_states} "
                    f"mq={cell.queries_total} eq={cell.equivalence_queries} ({cell.seconds:.1f}s)",
                    flush=True,
                )
                flush()

    flush(complete=True)
    print(f"\nWrote {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
