"""Run CAPAL on the oracles used by tests/test_lstar.py and
tests/test_baseline_lstar.py at the user-requested noise levels and report
an accuracy table.

usage::

    python -m orthogonal_dfa.experiments.capal_noise_sweep

The oracles mirror the ones in `tests/test_lstar.py` so the comparison
numbers are directly aligned with the existing L*/orthonormal-L* tests."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from orthogonal_dfa.baseline_lstar.baseline_lstar import run_baseline_lstar
from orthogonal_dfa.capal import run_capal
from orthogonal_dfa.l_star.examples.bernoulli_parity import (
    AllFramesClosedOracle,
    BernoulliParityOracle,
    BernoulliRegex,
)
from orthogonal_dfa.l_star.lstar import do_counterexample_driven_synthesis
from orthogonal_dfa.l_star.structures import SymmetricBernoulli
from tests.test_lstar import compute_pst, evaluate_accuracy

NOISE_LEVELS = [0.05, 0.10, 0.20, 0.30]


@dataclass
class OracleCase:
    name: str
    creator: Callable
    symbols: int = 2
    expected_states: Optional[int] = None
    # Some oracles have larger alphabets; their CAPAL pool can be left at
    # length 4 to keep it tractable. Binary stays at length 6.
    pool_max_len: int = 6
    pool_long_len: int = 10
    pool_num_long: int = 40


def cases() -> List[OracleCase]:
    return [
        OracleCase(
            name="parity_mod9_allowed_3_6",
            creator=lambda nm, s: BernoulliParityOracle(
                nm, s, modulo=9, allowed_moduluses=(3, 6)
            ),
            expected_states=9,
        ),
        OracleCase(
            name="regex_subseq_1010101",
            creator=lambda nm, s: BernoulliRegex(nm, s, regex=r".*1010101.*"),
            expected_states=8,
        ),
        OracleCase(
            name="regex_two_1111",
            creator=lambda nm, s: BernoulliRegex(nm, s, regex=r".*1111.*1111.*"),
        ),
        OracleCase(
            name="regex_alt_1111_or_0000_11",
            creator=lambda nm, s: BernoulliRegex(nm, s, regex=r".*(1111|0000)11.*"),
        ),
        OracleCase(
            name="regex_alt_111_or_000_3sym",
            creator=lambda nm, s: BernoulliRegex(
                nm, s, regex=r".*(111|000).*", alphabet_size=3
            ),
            symbols=3,
            pool_max_len=4,
            pool_long_len=8,
            pool_num_long=40,
        ),
        OracleCase(
            name="no_orf_4sym",
            creator=AllFramesClosedOracle,
            symbols=4,
            pool_max_len=4,
            pool_long_len=8,
            pool_num_long=40,
        ),
    ]


@dataclass
class Result:
    case: str
    learner: str
    noise: float
    p_correct: float
    states: Optional[int]
    accuracy: Optional[float]
    seconds: float
    error: Optional[str]


def _eval(case: OracleCase, dfa) -> float:
    return evaluate_accuracy(dfa, case.creator, symbols=case.symbols, count=5000)


def _run_capal_one(case: OracleCase, noise: float, *, seed: int) -> Result:
    p_correct = 1.0 - noise
    noisy_oracle = case.creator(SymmetricBernoulli(p_correct=p_correct), seed)
    truth_oracle = case.creator(SymmetricBernoulli(p_correct=1.0), seed)
    t0 = time.time()
    try:
        dfa = run_capal(
            noisy_oracle,
            truth_oracle,
            eta=noise,
            pool_max_len=case.pool_max_len,
            pool_long_len=case.pool_long_len,
            pool_num_long=case.pool_num_long,
        )
        return Result(
            case.name,
            "CAPAL",
            noise,
            p_correct,
            len(dfa.states),
            _eval(case, dfa),
            time.time() - t0,
            None,
        )
    except Exception as exc:  # noqa: BLE001
        return Result(
            case.name,
            "CAPAL",
            noise,
            p_correct,
            None,
            None,
            time.time() - t0,
            f"{type(exc).__name__}: {exc}",
        )


def _run_baseline_one(case: OracleCase, noise: float, *, seed: int) -> Result:
    p_correct = 1.0 - noise
    noisy_oracle = case.creator(SymmetricBernoulli(p_correct=p_correct), seed)
    t0 = time.time()
    try:
        dfa = run_baseline_lstar(noisy_oracle, max_states=50)
        return Result(
            case.name,
            "aalpy-L*",
            noise,
            p_correct,
            len(dfa.states),
            _eval(case, dfa),
            time.time() - t0,
            None,
        )
    except Exception as exc:  # noqa: BLE001
        return Result(
            case.name,
            "aalpy-L*",
            noise,
            p_correct,
            None,
            None,
            time.time() - t0,
            f"{type(exc).__name__}: {exc}",
        )


def _run_ortho_one(case: OracleCase, noise: float, *, seed: int) -> Result:
    """Run this repo's orthonormal L*. signal strength = 0.5 - noise."""
    p_correct = 1.0 - noise
    t0 = time.time()
    try:
        min_signal_strength = max(0.05, 0.5 - noise)
        pst = compute_pst(case.creator, min_signal_strength, seed)
        dfa, _ = do_counterexample_driven_synthesis(
            pst,
            additional_counterexamples=200,
            acc_threshold=0.98,
        )
        return Result(
            case.name,
            "ortho-L*",
            noise,
            p_correct,
            len(dfa.states),
            _eval(case, dfa),
            time.time() - t0,
            None,
        )
    except Exception as exc:  # noqa: BLE001
        return Result(
            case.name,
            "ortho-L*",
            noise,
            p_correct,
            None,
            None,
            time.time() - t0,
            f"{type(exc).__name__}: {exc}",
        )


def fmt_cell(r: Optional[Result]) -> str:
    if r is None:
        return "-"
    if r.error:
        return "ERR"
    return f"{r.accuracy:.2f}({r.states}st)"


def print_table(
    results: List[Result], cases_: List[OracleCase], learners: List[str]
) -> None:
    by_key = {(r.case, r.learner, r.noise): r for r in results}
    headers = ["oracle / learner"] + [f"eta={n:.2f}" for n in NOISE_LEVELS]
    rows: List[List[str]] = []
    for c in cases_:
        for learner in learners:
            row = [f"{c.name} / {learner}"]
            for n in NOISE_LEVELS:
                row.append(fmt_cell(by_key.get((c.name, learner, n))))
            rows.append(row)
        rows.append([""] + ["" for _ in NOISE_LEVELS])
    widths = [
        max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))
    ]
    print()
    print("| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        print("| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |")


def main() -> None:
    cases_ = cases()
    learners = ["CAPAL", "aalpy-L*", "ortho-L*"]
    runners = {
        "CAPAL": _run_capal_one,
        "aalpy-L*": _run_baseline_one,
        "ortho-L*": _run_ortho_one,
    }
    results: List[Result] = []
    for c in cases_:
        for learner in learners:
            for n in NOISE_LEVELS:
                print(f"=== {c.name} / {learner} eta={n:.2f} ===", flush=True)
                r = runners[learner](c, n, seed=0)
                tag = (
                    f"ERR={r.error}"
                    if r.error
                    else f"states={r.states} acc={r.accuracy:.4f}"
                )
                print(f"  -> {tag}  ({r.seconds:.1f}s)", flush=True)
                results.append(r)

    out_dir = Path(__file__).resolve().parents[2] / "data"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "capal_noise_sweep.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "case",
                "learner",
                "noise",
                "p_correct",
                "states",
                "accuracy",
                "seconds",
                "error",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.case,
                    r.learner,
                    r.noise,
                    r.p_correct,
                    r.states if r.states is not None else "",
                    f"{r.accuracy:.6f}" if r.accuracy is not None else "",
                    f"{r.seconds:.2f}",
                    r.error or "",
                ]
            )
    print(f"\nWrote {out_csv}")
    print_table(results, cases_, learners)


if __name__ == "__main__":
    main()
