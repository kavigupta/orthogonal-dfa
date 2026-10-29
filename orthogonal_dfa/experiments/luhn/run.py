"""
E-L* on `luhn_dfa`, the checksum credit card numbers satisfy.

    python -m orthogonal_dfa.experiments.luhn.run --processes 4
    python -m orthogonal_dfa.experiments.luhn.analyze

Results are cached as in `random_minimal.run`, one JSON file per (eta, seed)
under ``data/luhn``.
"""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path
from typing import Tuple

from orthogonal_dfa.experiments.capal_comparison.core import REPO_ROOT
from orthogonal_dfa.experiments.random_minimal.run import (
    has_current_result,
    run_and_write,
)
from orthogonal_dfa.l_star.examples.luhn import luhn_dfa

ETAS = [0.3]
SEEDS = list(range(5))
DATA_DIR = REPO_ROOT / "data" / "luhn"
FAMILY = "luhn"


def result_path(data_dir: Path, eta: float, seed: int) -> Path:
    return data_dir / "results" / f"luhn_eta{eta:.2f}_seed{seed}.json"


def _run_result(args: Tuple[Path, float, int]) -> str:
    data_dir, eta, seed = args
    return run_and_write(
        luhn_dfa(),
        result_path(data_dir, eta, seed),
        benchmark="luhn",
        family=FAMILY,
        eta=eta,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--processes", type=int, required=True)
    parser.add_argument("--etas", type=float, nargs="+", default=ETAS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    results = [
        (args.data_dir, eta, seed)
        for eta in args.etas
        for seed in args.seeds
        if not has_current_result(result_path(args.data_dir, eta, seed))
    ]
    print(f"{len(results)} results to run", flush=True)
    with multiprocessing.Pool(args.processes) as pool:
        for i, line in enumerate(pool.imap_unordered(_run_result, results)):
            print(f"[result {i + 1}/{len(results)}] {line}", flush=True)


if __name__ == "__main__":
    main()
