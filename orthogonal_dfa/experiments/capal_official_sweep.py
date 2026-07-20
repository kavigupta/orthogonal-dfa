"""Re-run the noise-level analysis using the official CAPAL implementation
(github.com/lkwargs/CAPAL).

The upstream code is imported via orthogonal_dfa.capal_official. We use
PerfectEQ from upstream (needs an explicit target DFA), so for each of our
oracle creators we hand the official CAPAL a target DFA built from the
underlying regex / modulo logic. The noiseless evaluation samples 5k random
words and queries our oracle for ground truth.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

from orthogonal_dfa.capal_official import (
    build_modulo_dfa,
    build_regex_dfa,
    evaluate_official_dfa,
    resolve_capal_dir,
    run_official_capal,
    verify_pinned,
)
from orthogonal_dfa.l_star.examples.bernoulli_parity import (
    BernoulliParityOracle,
    BernoulliRegex,
)

NOISE_LEVELS = [0.05, 0.10, 0.20, 0.30]


@dataclass
class OfficialCase:
    name: str
    target_builder: Callable[[], Any]
    oracle_creator: Callable
    alphabet_chars: List[str]
    symbols: int


def cases() -> List[OfficialCase]:
    return [
        OfficialCase(
            name="parity_mod9_allowed_3_6",
            target_builder=lambda: build_modulo_dfa(9, (3, 6)),
            oracle_creator=lambda nm, s: BernoulliParityOracle(
                nm, s, modulo=9, allowed_moduluses=(3, 6)
            ),
            alphabet_chars=["0", "1"],
            symbols=2,
        ),
        OfficialCase(
            name="regex_subseq_1010101",
            target_builder=lambda: build_regex_dfa(r".*1010101.*", 2),
            oracle_creator=lambda nm, s: BernoulliRegex(nm, s, regex=r".*1010101.*"),
            alphabet_chars=["0", "1"],
            symbols=2,
        ),
        OfficialCase(
            name="regex_two_1111",
            target_builder=lambda: build_regex_dfa(r".*1111.*1111.*", 2),
            oracle_creator=lambda nm, s: BernoulliRegex(nm, s, regex=r".*1111.*1111.*"),
            alphabet_chars=["0", "1"],
            symbols=2,
        ),
        OfficialCase(
            name="regex_alt_1111_or_0000_11",
            target_builder=lambda: build_regex_dfa(r".*(1111|0000)11.*", 2),
            oracle_creator=lambda nm, s: BernoulliRegex(
                nm, s, regex=r".*(1111|0000)11.*"
            ),
            alphabet_chars=["0", "1"],
            symbols=2,
        ),
        OfficialCase(
            name="regex_alt_111_or_000_3sym",
            target_builder=lambda: build_regex_dfa(r".*(111|000).*", 3),
            oracle_creator=lambda nm, s: BernoulliRegex(
                nm, s, regex=r".*(111|000).*", alphabet_size=3
            ),
            alphabet_chars=["0", "1", "2"],
            symbols=3,
        ),
    ]


@dataclass
class Result:
    case: str
    noise: float
    target_states: Optional[int]
    learned_states: Optional[int]
    accuracy: Optional[float]
    converged: Optional[bool]
    seconds: float
    error: Optional[str]


def run_one(c: OfficialCase, noise: float, *, seed: int = 0) -> Result:
    t0 = time.time()
    try:
        target = c.target_builder()
        dfa = run_official_capal(target, eta=noise, seed=seed, verbose=False)
        converged = not getattr(dfa, "converged", True) is False
        acc = evaluate_official_dfa(
            dfa, c.oracle_creator, c.alphabet_chars, c.symbols, count=5000
        )
        return Result(
            case=c.name,
            noise=noise,
            target_states=target.num_states,
            learned_states=dfa.num_states,
            accuracy=acc,
            converged=converged,
            seconds=time.time() - t0,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        return Result(
            case=c.name,
            noise=noise,
            target_states=None,
            learned_states=None,
            accuracy=None,
            converged=None,
            seconds=time.time() - t0,
            error=f"{type(exc).__name__}: {exc}",
        )


def fmt_cell(r: Optional[Result]) -> str:
    if r is None:
        return "-"
    if r.error:
        return "ERR"
    tag = "" if r.converged else "*"
    return f"{r.accuracy:.2f} ({r.learned_states}st{tag})"


def print_table(results: List[Result], cases_: List[OfficialCase]) -> None:
    by_key = {(r.case, r.noise): r for r in results}
    headers = ["oracle (target states)"] + [f"eta={n:.2f}" for n in NOISE_LEVELS]
    rows: List[List[str]] = []
    for c in cases_:
        target_states = c.target_builder().num_states
        row = [f"{c.name} ({target_states}st)"]
        for n in NOISE_LEVELS:
            row.append(fmt_cell(by_key.get((c.name, n))))
        rows.append(row)
    widths = [
        max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))
    ]
    print()
    print("| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        print("| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |")


def main() -> None:
    # Fail before spending an hour of compute on an unpinned checkout.
    verify_pinned(resolve_capal_dir())
    cases_ = cases()
    results: List[Result] = []
    for c in cases_:
        for n in NOISE_LEVELS:
            print(f"=== {c.name} eta={n:.2f} ===", flush=True)
            r = run_one(c, n)
            if r.error:
                print(f"  -> ERR={r.error} ({r.seconds:.1f}s)", flush=True)
            else:
                print(
                    f"  -> states={r.learned_states} acc={r.accuracy:.4f} "
                    f"({r.seconds:.1f}s)",
                    flush=True,
                )
            results.append(r)

    out_dir = Path(__file__).resolve().parents[2] / "data"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "capal_official_sweep.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "case",
                "noise",
                "target_states",
                "learned_states",
                "accuracy",
                "converged",
                "seconds",
                "error",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.case,
                    r.noise,
                    r.target_states if r.target_states is not None else "",
                    r.learned_states if r.learned_states is not None else "",
                    f"{r.accuracy:.6f}" if r.accuracy is not None else "",
                    "" if r.converged is None else ("Y" if r.converged else "N"),
                    f"{r.seconds:.2f}",
                    r.error or "",
                ]
            )
    print(f"\nWrote {out_csv}")
    print_table(results, cases_)


if __name__ == "__main__":
    main()
