# pylint: disable=duplicate-code
"""
CAPAL, at `make_learner`'s defaults, on the random minimal targets `run` draws.

    python -m orthogonal_dfa.experiments.random_minimal.capal_random --processes 12

Each (target, eta, seed) result is its own JSON file under
``data/random_minimal/capal_results``, resumable the same way as `run`'s.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from dataclasses import asdict
from pathlib import Path
from typing import Any, Tuple

from automata.fa.dfa import DFA

from orthogonal_dfa.capal_official import import_capal
from orthogonal_dfa.experiments.capal_comparison.core import eval_words, run_capal_cell

from .run import (
    COUNT,
    DATA_DIR,
    ETAS,
    FAMILY,
    SEEDS,
    SIZES,
    load_target,
    target_name,
    write_json,
)

RESULT_SCHEMA_VERSION = 1


def capal_result_path(
    data_dir: Path, size: int, index: int, eta: float, seed: int
) -> Path:
    name = f"{target_name(size, index)}_eta{eta:.2f}_seed{seed}.json"
    return data_dir / "capal_results" / name


def to_capal_dfa(aut: DFA, alphabet: list) -> Any:
    """Symbol index i becomes ``alphabet[i]``."""
    return import_capal().DFA(
        alphabet=alphabet,
        num_states=len(aut.states),
        start=aut.initial_state,
        accept=set(aut.final_states),
        delta={
            (q, alphabet[c]): aut.transitions[q][c]
            for q in aut.states
            for c in range(len(alphabet))
        },
    )


def _has_current_result(path: Path) -> bool:
    return (
        path.exists()
        and json.loads(path.read_text()).get("schema_version") == RESULT_SCHEMA_VERSION
    )


def _run_result(args: Tuple[Path, int, int, float, int]) -> str:
    data_dir, size, index, eta, seed = args
    aut = load_target(data_dir, size, index)
    alphabet = [str(c) for c in range(len(aut.input_symbols))]
    cell = run_capal_cell(
        to_capal_dfa(aut, alphabet),
        benchmark=target_name(size, index),
        family=FAMILY,
        eta=eta,
        seed=seed,
        words=eval_words(len(alphabet)),
        truth=aut.accepts_input,
        alphabet=alphabet,
    )
    write_json(
        capal_result_path(data_dir, size, index, eta, seed),
        {"schema_version": RESULT_SCHEMA_VERSION, **asdict(cell)},
    )
    return (
        f"{cell.benchmark} eta={eta:.2f} seed={seed}: accuracy={cell.accuracy} "
        f"states={cell.learned_states} queries={cell.queries_total} "
        f"converged={cell.converged} {cell.seconds:.0f}s"
        + (f" {cell.error_type}" if cell.error_type else "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--processes", type=int, required=True)
    parser.add_argument("--count", type=int, default=COUNT)
    parser.add_argument("--etas", type=float, nargs="+", default=ETAS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    results = [
        (args.data_dir, size, index, eta, seed)
        for eta in args.etas
        for size in SIZES
        for index in range(args.count)
        for seed in args.seeds
        if not _has_current_result(
            capal_result_path(args.data_dir, size, index, eta, seed)
        )
    ]
    print(f"{len(results)} results to run", flush=True)
    with multiprocessing.Pool(args.processes) as pool:
        for i, line in enumerate(pool.imap_unordered(_run_result, results)):
            print(f"[result {i + 1}/{len(results)}] {line}", flush=True)


if __name__ == "__main__":
    main()
