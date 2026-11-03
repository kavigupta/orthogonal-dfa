"""
CAPAL, at `make_learner`'s defaults, on the random minimal targets `run` draws.

    python -m orthogonal_dfa.experiments.random_minimal.capal_random --processes 12

Each (target, eta, seed) result is its own JSON file under
``data/random_minimal/capal_results``, resumable the same way as `run`'s. Once
all are in, how many of the `QUOTED_SIZE`-state targets CAPAL succeeds on at
`QUOTED_ETA` is written to ``figures/random-minimal/capal_successes.tex``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Tuple

from automata.fa.dfa import DFA

from orthogonal_dfa.capal_official import import_capal
from orthogonal_dfa.experiments.capal_comparison.core import eval_words, run_capal_cell

from .analyze import OUT_DIR, succeeded
from .run import (
    COUNT,
    DATA_DIR,
    ETAS,
    FAMILY,
    SEEDS,
    SIZES,
    experiment_parser,
    load_target,
    run_results,
    target_name,
    write_json,
)

RESULT_SCHEMA_VERSION = 1
QUOTED_SIZE = 12
QUOTED_ETA = 0.10


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
    parser = experiment_parser(
        __doc__.split("\n\n", maxsplit=1)[0], etas=ETAS, seeds=SEEDS, data_dir=DATA_DIR
    )
    parser.add_argument("--count", type=int, default=COUNT)
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
    run_results(args.processes, _run_result, results)
    write_quoted_successes(args.data_dir)


def write_quoted_successes(data_dir: Path) -> None:
    results = [
        json.loads(
            capal_result_path(
                data_dir, QUOTED_SIZE, index, QUOTED_ETA, seed
            ).read_text()
        )
        for index in range(COUNT)
        for seed in SEEDS
    ]
    path = OUT_DIR / "capal_successes.tex"
    # the % stops \input's trailing newline from becoming a space
    path.write_text(f"{sum(succeeded(r) for r in results)}%\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
