"""Drive both learners across a benchmark family and emit the experiment JSON.

Experiments 1 and 2 differ only in which benchmarks they run, so they share
this driver. Results are flushed after every cell: these sweeps run for hours,
and a crash in cell 300 must not cost the first 299.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List, Sequence

from orthogonal_dfa.l_star.preconditions import PreconditionReport

from .benchmark import Benchmark
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

#: Every sweep runs the whole grid: these noise levels, this seed, both
#: learners, every benchmark handed to it.
ETAS = [0.05, 0.10, 0.20, 0.30]
SEEDS = [0]
LEARNERS = [LEARNER_CAPAL, LEARNER_ELSTAR]


def run_cell(
    b: Benchmark,
    *,
    learner: str,
    eta: float,
    seed: int,
    words: Sequence[List[int]],
    truth: Callable[[List[int]], bool],
    regime: PreconditionReport,
) -> Cell:
    """One (benchmark, learner, eta, seed) cell, run or explicitly excluded.

    `regime` is this benchmark's `Benchmark.regime_report()`. CAPAL runs on
    everything; E-L* runs only where that report says it is in regime.
    """
    if learner == LEARNER_CAPAL:
        return run_capal_cell(
            b.target,
            benchmark=b.name,
            family=b.family,
            eta=eta,
            seed=seed,
            words=words,
            truth=truth,
            alphabet=b.alphabet,
        )
    if regime.satisfied:
        return run_elstar_cell(
            b.oracle_creator,
            benchmark=b.name,
            family=b.family,
            eta=eta,
            seed=seed,
            symbols=b.symbols,
            words=words,
            truth=truth,
            target_states=b.target_states,
        )
    # Outside E-L*'s designed regime: this repo's own benchmark generator would
    # have discarded this target. Recorded as an explicit, reasoned exclusion
    # rather than run -- a number here would measure the benchmark, not the
    # learner.
    return Cell(
        benchmark=b.name,
        family=b.family,
        learner=LEARNER_ELSTAR,
        eta=eta,
        seed=seed,
        target_states=b.target_states,
        alphabet_size=b.symbols,
        learner_config=asdict(regime),
        seconds=0.0,
        error_type="ExcludedOutOfRegime",
        error="; ".join(regime.reasons),
    ).finalize()


def describe(cell: Cell) -> str:
    """The one-line progress summary printed after each cell."""
    acc = cell.accuracy if cell.accuracy is None else round(cell.accuracy, 4)
    return (
        f"      -> states={cell.learned_states} acc={acc} "
        f"conv={cell.converged} mq={cell.queries_distinct} "
        f"eq={cell.equivalence_queries} ({cell.seconds:.1f}s)"
        + (f" ERR={cell.error}" if cell.error else "")
    )


def run_sweep(
    benchmarks: Sequence[Benchmark],
    *,
    experiment: str,
    description: str,
    generated_by: str,
) -> Path:
    """Run every learner on every benchmark at every noise level, and write
    `data/capal/<experiment>.json`."""
    out_path = REPO_ROOT / "data" / "capal" / f"{experiment}.json"
    config = {
        "etas": list(ETAS),
        "seeds": list(SEEDS),
        "learners": list(LEARNERS),
        "benchmarks": [b.name for b in benchmarks],
    }

    total = len(benchmarks) * len(ETAS) * len(SEEDS) * len(LEARNERS)
    cells: List[Cell] = []
    done = 0

    def flush() -> None:
        write_experiment(
            out_path,
            experiment=experiment,
            generated_by=generated_by,
            description=description,
            config=config,
            cells=cells,
        )

    # Before any cell runs, decide per target whether E-L* is in its designed
    # regime, via preconditions.satisfies_preconditions (acceptance balance +
    # class-preservation + the covered-accuracy ceiling, all at SAMPLE_LENGTH).
    # CAPAL runs on everything -- its PerfectEQ finds counterexamples
    # structurally, so none of these conditions constrain it.
    regime = {b.name: b.regime_report() for b in benchmarks}
    config["elstar_regime"] = {n: asdict(r) for n, r in regime.items()}
    config["elstar_regime_source"] = (
        "orthogonal_dfa/l_star/preconditions.py "
        "(satisfies_preconditions, default thresholds)"
    )
    excluded = [n for n, r in regime.items() if not r.satisfied]
    if excluded:
        print(
            f"E-L* EXCLUDED on {len(excluded)}/{len(benchmarks)} targets "
            f"(outside its designed regime): {', '.join(excluded)}",
            flush=True,
        )

    for b in benchmarks:
        # One word list per benchmark, shared by every learner/eta/seed cell on
        # it -- this is what makes the accuracies comparable.
        words = eval_words(b.symbols)
        truth = b.truth()
        for eta, seed, learner in itertools.product(ETAS, SEEDS, LEARNERS):
            done += 1
            print(
                f"[{done}/{total}] {b.name} eta={eta:.2f} seed={seed} {learner}",
                flush=True,
            )
            cell = run_cell(
                b,
                learner=learner,
                eta=eta,
                seed=seed,
                words=words,
                truth=truth,
                regime=regime[b.name],
            )
            cells.append(cell)
            print(describe(cell), flush=True)
            flush()

    flush()
    print(f"\nWrote {out_path} ({len(cells)} cells)")
    return out_path
