"""
E-L* on random minimal DFAs drawn by `sample_minimal_admitted_dfa`.

    python -m orthogonal_dfa.experiments.random_minimal.run --processes 4
    python -m orthogonal_dfa.experiments.random_minimal.analyze

Every target and every (target, eta, seed) result is its own JSON file under
``data/random_minimal``, written atomically. A target or a result of the current
schema that exists is never recomputed, so a run can be stopped anywhere and
resumed. Delete the results after changing the learner, and the targets after
changing the sampler.

A result carries the learner's `Cell`, the target's `difficulty.target_metrics`
and what `_RoundRecorder` keeps of each round.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from automata.fa.dfa import DFA

from orthogonal_dfa.experiments.capal_comparison.core import (
    REPO_ROOT,
    eval_words,
    run_elstar_cell,
)
from orthogonal_dfa.experiments.capal_comparison.sweep import ETAS as CAPAL_ETAS
from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.examples.random_dfa import (
    dfa_from_table,
    sample_minimal_admitted_dfa,
)
from orthogonal_dfa.l_star.learn import DEFAULT_SAMPLE_LENGTH
from orthogonal_dfa.l_star.structures import NoisyOracle
from orthogonal_dfa.l_star.tracker import SynthesisTracker

from .difficulty import target_metrics

SIZES = [6, 12]
COUNT = 100
ETAS = CAPAL_ETAS
SEEDS = [0]
ALPHABET_SIZE = 2
TARGET_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 2
DATA_DIR = REPO_ROOT / "data" / "random_minimal"
FAMILY = "random_minimal"


def target_name(size: int, index: int) -> str:
    return f"n{size:02d}_{index:03d}"


def target_path(data_dir: Path, size: int, index: int) -> Path:
    return data_dir / "targets" / f"{target_name(size, index)}.json"


def result_path(data_dir: Path, size: int, index: int, eta: float, seed: int) -> Path:
    name = f"{target_name(size, index)}_eta{eta:.2f}_seed{seed}.json"
    return data_dir / "results" / name


def write_json(path: Path, payload: Any) -> None:
    """Via a temporary file named for this process, so neither an interrupted
    write nor two processes writing the same file leave a broken one behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def load_target(data_dir: Path, size: int, index: int) -> DFA:
    """The ``index``-th target of ``size`` states, drawn with the rng seeded by
    ``(size, index)`` the first time it is asked for and read back after."""
    path = target_path(data_dir, size, index)
    if not path.exists():
        aut = sample_minimal_admitted_dfa(
            np.random.default_rng((size, index)),
            num_states=size,
            alphabet_size=ALPHABET_SIZE,
            length=DEFAULT_SAMPLE_LENGTH,
        )
        write_json(
            path,
            {
                "schema_version": TARGET_SCHEMA_VERSION,
                "size": size,
                "index": index,
                "table": [
                    [aut.transitions[q][c] for c in range(ALPHABET_SIZE)]
                    for q in range(size)
                ],
                "final_mask": sum(1 << q for q in aut.final_states),
            },
        )
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == TARGET_SCHEMA_VERSION, path
    return dfa_from_table(np.array(payload["table"]), payload["final_mask"])


def has_current_result(path: Path) -> bool:
    return (
        path.exists()
        and json.loads(path.read_text()).get("schema_version") == RESULT_SCHEMA_VERSION
    )


class _CountingDFAOracle(DFAOracle):
    def __init__(self, dfa: DFA):
        super().__init__(dfa)
        self.count = 0

    def membership_query(self, string: bytes) -> bool:
        self.count += 1
        return super().membership_query(string)


class _RoundRecorder(SynthesisTracker):
    """Per round, its `RoundSummary`, consistency, whether it was certified
    (None if not asked) and the queries spent by the end of its search."""

    def __init__(self, oracle: _CountingDFAOracle):
        self.oracle = oracle
        self.rounds = []
        self.chosen_round = None

    def on_round_summarized(self, summary, round_index):
        assert len(self.rounds) == round_index
        self.rounds.append(
            {
                **asdict(summary),
                "queries_at_search_end": self.oracle.count,
                "certified": None,
            }
        )

    def on_consistency_estimated(self, consistency, round_index):
        self.rounds[round_index]["consistency"] = consistency

    def on_certificate_decided(self, certified, round_index):
        self.rounds[round_index]["certified"] = certified

    def on_corrected_dfa_found(self, dfa, round_index):
        self.chosen_round = round_index


def _ensure_target(args: Tuple[Path, int, int]) -> str:
    data_dir, size, index = args
    load_target(data_dir, size, index)
    return target_name(size, index)


def _run_result(args: Tuple[Path, int, int, float, int]) -> str:
    data_dir, size, index, eta, seed = args
    target = load_target(data_dir, size, index)
    counting = _CountingDFAOracle(target)
    recorder = _RoundRecorder(counting)
    cell = run_elstar_cell(
        lambda noise_model, s: NoisyOracle(counting, noise_model, s),
        benchmark=target_name(size, index),
        family=FAMILY,
        eta=eta,
        seed=seed,
        symbols=ALPHABET_SIZE,
        words=eval_words(ALPHABET_SIZE),
        truth=target.accepts_input,
        target_states=size,
        tracker=recorder,
    )
    write_json(
        result_path(data_dir, size, index, eta, seed),
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            **asdict(cell),
            "target": target_metrics(target, length=DEFAULT_SAMPLE_LENGTH),
            "rounds": recorder.rounds,
            "chosen_round": recorder.chosen_round,
        },
    )
    return (
        f"{cell.benchmark} eta={eta:.2f} seed={seed}: accuracy={cell.accuracy} "
        f"states={cell.learned_states} queries={cell.queries_total} "
        f"rounds={len(recorder.rounds)} "
        f"{cell.seconds:.0f}s" + (f" {cell.error_type}" if cell.error_type else "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--processes", type=int, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    parser.add_argument("--count", type=int, default=COUNT)
    parser.add_argument("--etas", type=float, nargs="+", default=ETAS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    targets = [
        (args.data_dir, size, index)
        for size in args.sizes
        for index in range(args.count)
        if not target_path(args.data_dir, size, index).exists()
    ]
    results = [
        (args.data_dir, size, index, eta, seed)
        for eta in args.etas
        for size in args.sizes
        for index in range(args.count)
        for seed in args.seeds
        if not has_current_result(result_path(args.data_dir, size, index, eta, seed))
    ]
    print(f"{len(targets)} targets to draw, {len(results)} results to run", flush=True)
    with multiprocessing.Pool(args.processes) as pool:
        for i, name in enumerate(pool.imap_unordered(_ensure_target, targets)):
            print(f"[target {i + 1}/{len(targets)}] {name}", flush=True)
        for i, line in enumerate(pool.imap_unordered(_run_result, results)):
            print(f"[result {i + 1}/{len(results)}] {line}", flush=True)


if __name__ == "__main__":
    main()
