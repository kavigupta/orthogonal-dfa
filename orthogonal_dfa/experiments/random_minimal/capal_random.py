"""
CAPAL, at `make_learner`'s defaults, on the random minimal targets `run` draws.

    python -m orthogonal_dfa.experiments.random_minimal.capal_random --processes 12

Each (target, eta, seed) result is its own JSON file under
``data/random_minimal/capal_results``, resumable the same way as `run`'s. Once
all are in, how many targets CAPAL succeeds on per size and eta is written to
``figures/random-minimal/capal_successes.tex`` as macros, e.g. 12 states at
eta = 0.10 as ``\\capalSuccessesTwelveEtaTen``.
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
_ONES = (
    "Zero One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve "
    "Thirteen Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen"
).split()
_TENS = "_ _ Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety".split()


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
    write_success_macros(args.data_dir)


def _number_name(n: int) -> str:
    """For n < 100, since LaTeX macro names cannot contain digits."""
    if n < 20:
        return _ONES[n]
    return _TENS[n // 10] + (_ONES[n % 10] if n % 10 else "")


def write_success_macros(data_dir: Path) -> None:
    lines = []
    for size in SIZES:
        for eta in ETAS:
            successes = sum(
                succeeded(
                    json.loads(
                        capal_result_path(data_dir, size, index, eta, seed).read_text()
                    )
                )
                for index in range(COUNT)
                for seed in SEEDS
            )
            name = (
                f"capalSuccesses{_number_name(size)}Eta{_number_name(round(eta * 100))}"
            )
            lines.append(f"\\newcommand{{\\{name}}}{{{successes}}}\n")
    path = OUT_DIR / "capal_successes.tex"
    path.write_text("".join(lines))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
