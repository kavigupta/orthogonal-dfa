"""Drive both learners across a benchmark family and emit the experiment JSON.

Experiments 1 and 2 differ only in which benchmarks they run, so they share
this driver. Results are flushed after every cell: these sweeps run for hours,
and a crash in cell 300 must not cost the first 299.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .core import (
    LEARNER_CAPAL,
    LEARNER_ELSTAR,
    Cell,
    eval_words,
    run_capal_cell,
    run_elstar_cell,
    write_experiment,
)
from .targets import (
    MIN_ACCEPT_OR_REJECT,
    MIN_CLASS_PRESERVING_FRAC,
    Benchmark,
)

DEFAULT_ETAS = [0.05, 0.10, 0.20, 0.30]
DEFAULT_SEEDS = [0]
DEFAULT_LEARNERS = [LEARNER_CAPAL, LEARNER_ELSTAR]


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--etas",
        nargs="+",
        type=float,
        default=DEFAULT_ETAS,
        help="Persistent-noise levels to sweep.",
    )
    ap.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Learner seeds."
    )
    ap.add_argument(
        "--learners",
        nargs="+",
        default=DEFAULT_LEARNERS,
        choices=[LEARNER_CAPAL, LEARNER_ELSTAR],
        help="Which learners to run.",
    )
    ap.add_argument(
        "--targets", nargs="+", default=None, help="Restrict to these benchmark names."
    )
    ap.add_argument("--out", default=None, help="Output JSON path.")
    ap.add_argument(
        "--capal-dir", default=None, help="Override the pinned CAPAL checkout."
    )


def run_sweep(
    benchmarks: Sequence[Benchmark],
    *,
    experiment: str,
    description: str,
    generated_by: str,
    out_path: Path,
    etas: Sequence[float],
    seeds: Sequence[int],
    learners: Sequence[str],
    capal_kwargs: Optional[Dict[str, Any]] = None,
) -> Path:
    capal_kwargs = capal_kwargs or {}
    config = {
        "etas": list(etas),
        "seeds": list(seeds),
        "learners": list(learners),
        "benchmarks": [b.name for b in benchmarks],
        "capal_learner_defaults": capal_kwargs,
    }

    total = len(benchmarks) * len(etas) * len(seeds) * len(learners)
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

    # E-L*'s sampling length is tuned per target before any cell runs, and
    # recorded: on CAPAL's dataset the default 40 is degenerate for 13 of 28
    # targets (near-0 or near-1 acceptance), which starves the acceptance-rate
    # test of signal entirely. CAPAL is unaffected -- its PerfectEQ finds
    # counterexamples structurally rather than by sampling.
    # Before any cell runs, decide per target whether E-L* is being asked
    # something it was designed for, using the same two filters this repo's own
    # benchmark generator applies. CAPAL runs on everything -- its PerfectEQ
    # finds counterexamples structurally, so neither filter constrains it.
    tuning = {b.name: b.regime_report() for b in benchmarks}
    config["elstar_regime"] = tuning
    config["elstar_regime_filters"] = {
        "min_accept_or_reject": MIN_ACCEPT_OR_REJECT,
        "min_class_preserving_frac": MIN_CLASS_PRESERVING_FRAC,
        "source": (
            "orthogonal_dfa/l_star/examples/benchmark_generator.py"
            " (sample_balanced_benchmark), thresholds as passed in"
            " tests/test_lstar.py"
        ),
    }
    excluded = [n for n, t in tuning.items() if not t["in_regime"]]
    retuned = [n for n, t in tuning.items() if t["tuned_from_default"]]
    if retuned:
        print(
            f"E-L* sampling length tuned away from 40 on {len(retuned)} target(s): "
            + ", ".join(f"{n}->{tuning[n]['sample_length']}" for n in retuned),
            flush=True,
        )
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
        for eta in etas:
            for seed in seeds:
                for learner in learners:
                    done += 1
                    print(
                        f"[{done}/{total}] {b.name} eta={eta:.2f} seed={seed} "
                        f"{learner}",
                        flush=True,
                    )
                    if learner == LEARNER_CAPAL:
                        cell = run_capal_cell(
                            b.target,
                            benchmark=b.name,
                            family=b.family,
                            eta=eta,
                            seed=seed,
                            words=words,
                            truth=truth,
                            alphabet=b.alphabet,
                            **capal_kwargs,
                        )
                    elif not tuning[b.name]["in_regime"]:
                        # Outside E-L*'s designed regime: this repo's own
                        # benchmark generator would have discarded this target.
                        # Recorded as an explicit, reasoned exclusion rather
                        # than run -- a number here would measure the benchmark,
                        # not the learner.
                        cell = Cell(
                            benchmark=b.name,
                            family=b.family,
                            learner=LEARNER_ELSTAR,
                            eta=eta,
                            seed=seed,
                            target_states=b.target_states,
                            alphabet_size=b.symbols,
                            learner_config=dict(tuning[b.name]),
                            seconds=0.0,
                            error_type="ExcludedOutOfRegime",
                            error="; ".join(tuning[b.name]["excluded_because"]),
                        ).finalize()
                    else:
                        cell = run_elstar_cell(
                            b.oracle_creator,
                            benchmark=b.name,
                            family=b.family,
                            eta=eta,
                            seed=seed,
                            symbols=b.symbols,
                            words=words,
                            truth=truth,
                            target_states=b.target_states,
                            sample_length=tuning[b.name]["sample_length"],
                            accept_rate_at_sample_length=tuning[b.name][
                                "accept_rate_at_sample_length"
                            ],
                        )
                    cells.append(cell)
                    print(
                        f"      -> states={cell.learned_states} "
                        f"acc={cell.accuracy if cell.accuracy is None else round(cell.accuracy, 4)} "
                        f"conv={cell.converged} q={cell.queries_distinct} "
                        f"({cell.seconds:.1f}s)"
                        + (f" ERR={cell.error}" if cell.error else ""),
                        flush=True,
                    )
                    flush()

    flush()
    print(f"\nWrote {out_path} ({len(cells)} cells)")
    return out_path
