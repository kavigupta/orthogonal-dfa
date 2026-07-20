#!/usr/bin/env python3
"""Run the official CAPAL learner (github.com/lkwargs/CAPAL) on five target
DFAs at a configurable list of persistent-noise levels.

No dependency on this repo's orthogonal_dfa package; the only in-repo import is
the sibling `capal_upstream` module in this folder. The other requirements are
    - `automata-lib`  (for regex -> DFA compilation), and
    - a clone of github.com/lkwargs/CAPAL at the commit pinned in
      capal_upstream.py, checked out clean. Defaults to `../capal` relative to
      the repo root; pass --capal-dir to point elsewhere.

The pin is enforced: a wrong commit or a dirty tree is a hard error, since the
numbers in data/capal_findings.md are only reproducible against that commit.

Example:
    python scripts/run_capal_official.py --noises 0.05 0.1 0.2 0.3 \
        --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# -- upstream import ----------------------------------------------------------


sys.path.insert(0, str(Path(__file__).resolve().parent))
from capal_upstream import (  # noqa: E402
    DEFAULT_CAPAL_DIR,
    PINNED_COMMIT,
    import_capal,
)


# -- target-DFA builders ------------------------------------------------------


def build_modulo_dfa(M: Any, modulo: int, allowed: List[int]) -> Any:
    """DFA over {'0','1'} that accepts iff (count of '1' chars) mod `modulo`
    is in `allowed`. Matches BernoulliParityOracle."""
    delta = {}
    for q in range(modulo):
        delta[(q, "0")] = q
        delta[(q, "1")] = (q + 1) % modulo
    return M.DFA(
        alphabet=["0", "1"],
        num_states=modulo,
        start=0,
        accept=set(allowed),
        delta=delta,
    )


def build_regex_dfa(M: Any, regex: str, alphabet_size: int = 2) -> Any:
    """Compile a regex over the digit alphabet {'0', ..., str(alphabet_size-1)}
    to a minimal DFA, then re-encode in CAPAL's DFA format."""
    from automata.fa.dfa import DFA as AutDFA
    from automata.fa.nfa import NFA

    syms = {str(i) for i in range(alphabet_size)}
    nfa = NFA.from_regex(regex, input_symbols=syms)
    aut = AutDFA.from_nfa(nfa, minify=True)

    state_list = sorted(aut.states, key=str)
    sidx = {s: i for i, s in enumerate(state_list)}
    delta = {}
    for s in state_list:
        for a, dst in aut.transitions[s].items():
            delta[(sidx[s], a)] = sidx[dst]
    return M.DFA(
        alphabet=sorted(aut.input_symbols),
        num_states=len(state_list),
        start=sidx[aut.initial_state],
        accept={sidx[s] for s in aut.final_states},
        delta=delta,
    )


def default_cases(M: Any) -> List[Tuple[str, Any]]:
    return [
        ("parity_mod9_allowed_3_6", build_modulo_dfa(M, 9, [3, 6])),
        ("regex_subseq_1010101", build_regex_dfa(M, r".*1010101.*", 2)),
        ("regex_two_1111", build_regex_dfa(M, r".*1111.*1111.*", 2)),
        ("regex_alt_1111_or_0000_11", build_regex_dfa(M, r".*(1111|0000)11.*", 2)),
        ("regex_alt_111_or_000_3sym", build_regex_dfa(M, r".*(111|000).*", 3)),
    ]


# -- learner driver -----------------------------------------------------------


def run_one(
    M: Any,
    target: Any,
    eta: float,
    *,
    max_iters: int,
    seed: int,
) -> Tuple[Any, bool, float]:
    """Run CAPALLearner.fit on `target` at the given noise level. If `fit`
    raises (hit max_iters), return the learner's last hypothesis so accuracy
    can still be measured."""
    cfg = M.LearnerConfig(eta=eta, seed=seed, max_iters=max_iters, verbose=False)
    learner = M.CAPALLearner(target=target, cfg=cfg)
    t0 = time.time()
    try:
        dfa = learner.fit()
        converged = True
    except RuntimeError:
        converged = False
        last = getattr(learner, "_last_hyp", None)
        if last is None or last.dfa is None:
            raise
        dfa = last.dfa
    return dfa, converged, time.time() - t0


def eval_accuracy(
    target: Any,
    learned: Any,
    *,
    count: int,
    max_len: int,
    seed: int,
) -> float:
    """Sample random words of length [1, max_len] and report the fraction on
    which `learned` agrees with `target`."""
    rng = random.Random(seed)
    sigma = list(target.alphabet)
    ok = 0
    for _ in range(count):
        L = rng.randint(1, max_len)
        w = "".join(rng.choice(sigma) for _ in range(L))
        if bool(learned.run(w)) == bool(target.run(w)):
            ok += 1
    return ok / count


# -- presentation -------------------------------------------------------------


def print_table(
    targets: List[Tuple[str, Any]],
    noises: List[float],
    results: Dict[Tuple[str, float], Dict[str, Any]],
) -> None:
    headers = ["oracle (target st)"] + [f"eta={n:.2f}" for n in noises]
    rows: List[List[str]] = []
    for name, target in targets:
        row = [f"{name} ({target.num_states}st)"]
        for n in noises:
            r = results.get((name, n))
            if r is None or r["error"] is not None:
                row.append("ERR")
                continue
            tag = "" if r["converged"] else "*"
            row.append(f"{r['acc']:.3f} ({r['states']}st{tag})")
        rows.append(row)

    widths = [max(len(r[i]) for r in [headers] + rows) for i in range(len(headers))]

    def line(r: List[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"

    print()
    print(line(headers))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        print(line(r))
    print()
    print("`*` = CAPAL hit --max-iters without PerfectEQ accepting; the cell")
    print("shows the last hypothesis CAPAL produced.")


def write_csv(
    csv_path: str,
    targets: List[Tuple[str, Any]],
    noises: List[float],
    results: Dict[Tuple[str, float], Dict[str, Any]],
) -> None:
    target_by_name = dict(targets)
    with open(csv_path, "w", newline="") as f:
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
        for (name, eta), r in results.items():
            target = target_by_name[name]
            w.writerow(
                [
                    name,
                    eta,
                    target.num_states,
                    r["states"] if r["states"] is not None else "",
                    f"{r['acc']:.6f}" if r["acc"] is not None else "",
                    "" if r["converged"] is None else ("Y" if r["converged"] else "N"),
                    f"{r['secs']:.2f}",
                    r["error"] or "",
                ]
            )


# -- entry point --------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--capal-dir",
        default=None,
        help=f"Clone of github.com/lkwargs/CAPAL, pinned at {PINNED_COMMIT[:7]} "
        f"with a clean tree. Default: {DEFAULT_CAPAL_DIR}",
    )
    ap.add_argument(
        "--noises",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.20, 0.30],
        help="Persistent-noise rates to sweep.",
    )
    ap.add_argument(
        "--max-iters",
        type=int,
        default=200,
        help="CAPAL fit() iteration cap.",
    )
    ap.add_argument(
        "--eval-samples",
        type=int,
        default=5000,
        help="Random-word samples for accuracy.",
    )
    ap.add_argument(
        "--eval-max-len",
        type=int,
        default=40,
        help="Max length of accuracy-eval words.",
    )
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument(
        "--csv",
        default=None,
        help="Optional path to write per-cell results as CSV.",
    )
    args = ap.parse_args()

    M = import_capal(args.capal_dir)
    targets = default_cases(M)

    results: Dict[Tuple[str, float], Dict[str, Any]] = {}
    for name, target in targets:
        for eta in args.noises:
            print(f"=== {name} eta={eta:.2f} ===", flush=True)
            try:
                dfa, converged, secs = run_one(
                    M, target, eta, max_iters=args.max_iters, seed=args.seed
                )
                acc = eval_accuracy(
                    target,
                    dfa,
                    count=args.eval_samples,
                    max_len=args.eval_max_len,
                    seed=args.seed,
                )
                results[(name, eta)] = {
                    "states": dfa.num_states,
                    "acc": acc,
                    "converged": converged,
                    "secs": secs,
                    "error": None,
                }
                tag = "Y" if converged else "N"
                print(
                    f"  -> states={dfa.num_states} acc={acc:.4f} conv={tag} ({secs:.1f}s)",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                results[(name, eta)] = {
                    "states": None,
                    "acc": None,
                    "converged": None,
                    "secs": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"  -> ERR {type(exc).__name__}: {exc}", flush=True)

    print_table(targets, args.noises, results)
    if args.csv:
        write_csv(args.csv, targets, args.noises, results)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
