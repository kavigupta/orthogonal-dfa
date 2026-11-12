"""
E-L* on `luhn_dfa`, the checksum credit card numbers satisfy.

    python -m orthogonal_dfa.experiments.luhn.run --processes 4
    python -m orthogonal_dfa.experiments.luhn.analyze

Results are cached as in `random_minimal.run`, one JSON file per (sample
length, eta, seed) under ``data/luhn``.
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

# every sampled string is a 16-digit number, as a credit card number is
SAMPLE_LENGTH = 16
ETAS = [0.3]
SEEDS = list(range(5))
DATA_DIR = REPO_ROOT / "data" / "luhn"
FAMILY = "luhn"


def result_path(data_dir: Path, eta: float, seed: int) -> Path:
    name = f"luhn_len{SAMPLE_LENGTH}_eta{eta:.2f}_seed{seed}.json"
    return data_dir / "results" / name


def _run_result(args: Tuple[Path, float, int]) -> str:
    data_dir, eta, seed = args
    return run_and_write(
        luhn_dfa(),
        result_path(data_dir, eta, seed),
        benchmark="luhn",
        family=FAMILY,
        eta=eta,
        seed=seed,
        sample_length=SAMPLE_LENGTH,
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
