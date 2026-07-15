#!/usr/bin/env python3
"""Benchmark a PR by counting oracle membership queries on the L* synthesis
tasks, comparing the current branch against a base branch.

Modelled on egg-stitch's ``scripts/bench_pr.py``: it checks out ``BASE`` and the
``PR`` branch into two ephemeral git worktrees and drives all measurements from a
single process, running *this same script* as an in-worktree measurer
(``--emit-json --root <wt>``) so the harness is identical on both sides and only
the imported ``orthogonal_dfa`` library differs. No ``git checkout`` happens in
the main repo.

Because synthesis is deterministic (``membership_query`` is a pure stable-hash of
the string — no RNG), one run per task is exact; there is no rep-sampling like
bench_pr needs for its stochastic search. Each task reports total oracle queries,
distinct queries, learned-state count, and noiseless accuracy. The comparison
table flags any task whose PR query count went **up** or whose **accuracy
regressed** (the guard tasks ``modulo_hard`` / ``modulo_asym`` exist to catch a
change that speeds up easy cells but breaks the high-noise / asymmetric ones),
and the report is posted to the PR body via ``gh`` when one exists.

Usage:
    python scripts/count_queries.py                     # compare current branch vs main
    python scripts/count_queries.py main my-branch      # explicit base / pr
    python scripts/count_queries.py --local             # just measure the working tree
    python scripts/count_queries.py --local modulo subseq   # named tasks, working tree
    python scripts/count_queries.py --no-pr             # compare but don't touch the PR
"""

import argparse
import contextlib
import io
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# name -> spec. `oracle` is a serialisable description the measurer turns into an
# oracle_creator; `signal` is min_signal_strength; `noise` is None (symmetric,
# p_correct=0.5+signal) or an asymmetric {p_0, p_1}. `slow` marks the high-cost
# guard tasks so `--fast` can skip them.
BENCHMARKS = {
    "modulo":     {"oracle": {"kind": "modulo", "modulo": 9, "allowed": [3, 6]},
                   "signal": 0.3, "symbols": 2, "noise": None},
    "subseq":     {"oracle": {"kind": "regex", "regex": r".*1010101.*"},
                   "signal": 0.3, "symbols": 2, "noise": None},
    "two_subseq": {"oracle": {"kind": "regex", "regex": r".*1111.*1111.*"},
                   "signal": 0.3, "symbols": 2, "noise": None},
    "poor_case":  {"oracle": {"kind": "poor_case"},
                   "signal": 0.3, "symbols": 2, "noise": None},
    # --- guard tasks: the cells only ortho-L* solves; a query win must not break these ---
    "modulo_hard": {"oracle": {"kind": "modulo", "modulo": 9, "allowed": [3, 6]},
                    "signal": 0.2, "symbols": 2, "noise": None, "slow": True},  # η=0.30 wall
    "modulo_asym": {"oracle": {"kind": "modulo", "modulo": 9, "allowed": [3, 6]},
                    "signal": 0.15, "symbols": 2,
                    "noise": {"p_0": 0.10, "p_1": 0.40}, "slow": True},  # non-straddling
}

# The 10-state adversarial DFA used by the `poor_case` task (kept in-script so the
# measurer can rebuild it against whichever branch's DFAOracle it imports).
POOR_CASE_DFA = {
    "transitions": {
        0: {1: 8, 0: 0}, 1: {1: 1, 0: 1}, 2: {1: 1, 0: 6}, 3: {1: 9, 0: 2},
        4: {1: 3, 0: 8}, 5: {1: 8, 0: 4}, 6: {1: 3, 0: 9}, 7: {1: 8, 0: 6},
        8: {1: 8, 0: 5}, 9: {1: 3, 0: 7},
    },
    "initial": 0,
    "final": [1],
}

# Regression bands: a query ratio inside +/- QUERY_BAND is "no change"; an accuracy
# drop beyond ACC_EPS is a regression regardless of the query win.
QUERY_BAND = 0.02
ACC_EPS = 0.005


# ---------------------------------------------------------------------------
# Measurer: runs INSIDE a worktree (imports that branch's orthogonal_dfa).
# Imports are lazy so `--root` is on sys.path before the library is loaded.
# ---------------------------------------------------------------------------


def _measure(root: str, names: list[str]) -> dict:
    sys.path.insert(0, root)
    from automata.fa.dfa import DFA
    from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
    from orthogonal_dfa.l_star.examples.bernoulli_parity import (
        BernoulliParityOracle, BernoulliRegex,
    )
    from orthogonal_dfa.l_star.structures import Oracle, AsymmetricBernoulli
    from tests.test_lstar import compute_dfa_for_oracle, evaluate_accuracy

    class CountingOracle(Oracle):
        def __init__(self, inner):
            self._inner = inner
            self.count = 0
            self.distinct = set()

        @property
        def alphabet_size(self):
            return self._inner.alphabet_size

        def membership_query(self, string):
            self.count += 1
            self.distinct.add(tuple(string))
            return self._inner.membership_query(string)

    def build_creator(spec):
        kind = spec["kind"]
        if kind == "modulo":
            return lambda nm, s: BernoulliParityOracle(
                nm, s, modulo=spec["modulo"], allowed_moduluses=tuple(spec["allowed"]))
        if kind == "regex":
            return lambda nm, s: BernoulliRegex(nm, s, regex=spec["regex"])
        if kind == "poor_case":
            dfa = DFA(
                states=set(range(10)), input_symbols={0, 1},
                transitions=POOR_CASE_DFA["transitions"],
                initial_state=POOR_CASE_DFA["initial"],
                final_states=set(POOR_CASE_DFA["final"]), allow_partial=False)
            return lambda nm, s: DFAOracle(nm, s, dfa)
        raise ValueError(f"unknown oracle kind {kind!r}")

    results = {}
    for name in names:
        bench = BENCHMARKS[name]
        creator = build_creator(bench["oracle"])
        counters = []

        def counting_creator(noise_model, s):
            o = CountingOracle(creator(noise_model, s))
            counters.append(o)
            return o

        noise_model = None
        if bench["noise"] is not None:
            noise_model = AsymmetricBernoulli(**bench["noise"])

        t0 = time.time()
        # Silence the synthesis chatter; we only want the JSON on stdout.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _, dfa, _ = compute_dfa_for_oracle(
                counting_creator, min_signal_strength=bench["signal"], seed=0,
                noise_model=noise_model)
            acc = evaluate_accuracy(dfa, creator, symbols=bench["symbols"])
        results[name] = {
            "queries": sum(c.count for c in counters),
            "distinct": len(set().union(*[c.distinct for c in counters])),
            "states": len(dfa.states),
            "accuracy": acc,
            "seconds": time.time() - t0,
        }
    return results


# ---------------------------------------------------------------------------
# Driver: git worktrees, run the measurer on each branch, compare.
# ---------------------------------------------------------------------------


def sh(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, cwd=ROOT, **kw)


def rev_parse(ref: str):
    """Commit SHA ``ref`` resolves to, or None if it doesn't exist."""
    res = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", ref],
        cwd=ROOT, capture_output=True, text=True)
    return res.stdout.strip() or None


def preflight(base: str, *, allow_stale: bool):
    """Abort unless the working tree is clean and (unless --allow-stale-base)
    local ``base`` is current with ``origin/base``.

    A dirty tree usually means the user is mid-edit (the worktree comparison uses
    committed state, so uncommitted work would be silently ignored). A stale
    ``base`` would compare against the wrong baseline — as the accidental
    stale-main run during development showed.
    """
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty:
        raise SystemExit(
            "count_queries: working tree is not clean — commit or stash before a "
            "base-vs-PR run (or use --local to measure the working tree).\n" + dirty)
    if allow_stale:
        return
    if rev_parse(base) is None:
        raise SystemExit(f"count_queries: baseline ref `{base}` does not exist locally")
    fetch = subprocess.run(
        ["git", "fetch", "origin", base], cwd=ROOT, capture_output=True, text=True)
    if fetch.returncode != 0:
        raise SystemExit(
            f"count_queries: `git fetch origin {base}` failed (pass --allow-stale-base "
            f"to skip this check when offline):\n{fetch.stderr.strip()}")
    local, remote = rev_parse(base), rev_parse("FETCH_HEAD")
    if local != remote:
        raise SystemExit(
            f"count_queries: local `{base}` ({local[:12]}) is behind "
            f"`origin/{base}` ({(remote or '?')[:12]}); run `git pull` on `{base}` "
            f"first, or pass --allow-stale-base")


def setup_worktree(ref: str, wt_dir: Path):
    # --detach so we don't conflict with the branch the main worktree has out.
    sh(["git", "worktree", "add", "--detach", str(wt_dir), ref])


def teardown_worktree(wt_dir: Path):
    subprocess.run(["git", "worktree", "remove", "--force", str(wt_dir)],
                   cwd=ROOT, check=False)


def measure_worktree(root: Path, names: list[str]) -> dict:
    """Run this script as an in-worktree measurer; parse the JSON it prints."""
    out = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--emit-json",
         "--root", str(root), "--tasks", *names],
        cwd=root, text=True, capture_output=True)
    if out.returncode != 0:
        raise SystemExit(f"count_queries: measurer failed in {root}:\n{out.stderr}")
    return json.loads(out.stdout)


def _emoji(ratio: float, acc_ok: bool, states_ok: bool) -> str:
    if not acc_ok or not states_ok:
        return "🔴"  # correctness regression trumps any query win
    if ratio < 1 - QUERY_BAND:
        return "🟢"  # fewer queries
    if ratio > 1 + QUERY_BAND:
        return "🔴"  # more queries
    return "⚪"


def comparison_report(base_ref: str, pr_ref: str, base: dict, pr: dict,
                      names: list[str]) -> str:
    lines = [
        f"## Query count — `{pr_ref}` vs `{base_ref}`",
        "",
        f"|   | task | queries `{base_ref}` | queries `{pr_ref}` | ratio | "
        f"acc `{base_ref}` | acc `{pr_ref}` | states |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    ratios = []
    any_regression = False
    for name in names:
        b, p = base[name], pr[name]
        ratio = p["queries"] / b["queries"] if b["queries"] else float("inf")
        ratios.append(ratio)
        acc_ok = p["accuracy"] >= b["accuracy"] - ACC_EPS
        states_ok = p["states"] == b["states"]
        emoji = _emoji(ratio, acc_ok, states_ok)
        if emoji == "🔴":
            any_regression = True
        st = f"{b['states']}" if states_ok else f"{b['states']}→{p['states']} ‼️"
        lines.append(
            f"| {emoji} | {name} | {b['queries']:,} | {p['queries']:,} | {ratio:.2f}x | "
            f"{b['accuracy']:.3f} | {p['accuracy']:.3f} | {st} |")
    geo = math.prod(ratios) ** (1 / len(ratios)) if ratios else float("nan")
    lines.append(
        f"| {'🟢' if geo < 1 - QUERY_BAND else '🔴' if geo > 1 + QUERY_BAND else '⚪'} "
        f"| **geomean** |  |  | **{geo:.2f}x** |  |  |  |")
    lines.append("")
    lines.append("**❌ REGRESSION** (queries up or accuracy/states broke on ≥1 task)"
                 if any_regression else
                 "**✅ no regressions** (accuracy held, no task's queries rose beyond band)")
    return "\n".join(lines)


def update_pr_report(pr_ref: str, report: str):
    """Best-effort: replace/append the managed query-count block in the PR body."""
    try:
        body = subprocess.check_output(
            ["gh", "pr", "view", pr_ref, "--json", "body", "-q", ".body"],
            cwd=ROOT, text=True, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        msg = getattr(e, "stderr", "") or str(e)
        print(f"\ncount_queries: no PR found for {pr_ref!r}, skipping PR update.\n  {msg}")
        return
    import re
    body = body.rstrip("\n")
    pattern = re.compile(r"(?m)^## Query count\b.*?(?=^## |\Z)", re.DOTALL)
    if pattern.search(body):
        new_body = pattern.sub(lambda _: report.rstrip() + "\n\n", body).rstrip() + "\n"
    else:
        new_body = (body + ("\n\n" if body else "") + report.rstrip() + "\n")
    res = subprocess.run(["gh", "pr", "edit", pr_ref, "--body-file", "-"],
                         cwd=ROOT, input=new_body, text=True, capture_output=True)
    print(f"\ncount_queries: {'updated' if res.returncode == 0 else 'FAILED to update'}"
          f" the Query count section on PR {pr_ref}."
          + ("" if res.returncode == 0 else f"\n{res.stderr}"))


def print_local_table(results: dict, names: list[str]):
    print("\n===== QUERY COUNT SUMMARY =====")
    header = f"{'task':<14}{'queries':>12}{'distinct':>11}{'states':>8}{'acc':>8}{'sec':>8}"
    print(header)
    print("-" * len(header))
    for name in names:
        r = results[name]
        print(f"{name:<14}{r['queries']:>12,}{r['distinct']:>11,}{r['states']:>8}"
              f"{r['accuracy']:>8.3f}{r['seconds']:>8.1f}")
    print(f"\nTOTAL queries: {sum(results[n]['queries'] for n in names):,}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("base", nargs="?", default="main", help="baseline ref (default: main)")
    p.add_argument("pr", nargs="?", default=None, help="PR ref (default: current branch)")
    p.add_argument("--local", action="store_true",
                   help="measure the working tree only; print a table, no comparison")
    p.add_argument("--fast", action="store_true", help="skip the slow guard tasks")
    p.add_argument("--no-pr", action="store_true", help="compare but don't edit the PR body")
    p.add_argument("--allow-stale-base", action="store_true",
                   help="skip the 'base up to date with origin' check (e.g. offline)")
    p.add_argument("--emit-json", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--root", default=str(ROOT), help=argparse.SUPPRESS)
    p.add_argument("--tasks", nargs="*", default=None,
                   help="task names to run (default: all, or all-but-slow with --fast)")
    a = p.parse_args()

    # Positional base/pr double as task names in --local/--emit-json usage
    # (e.g. `--local modulo subseq`); collect any that are benchmark names.
    positional_tasks = [x for x in (a.base, a.pr) if x in BENCHMARKS]
    names = a.tasks if a.tasks is not None else (positional_tasks or list(BENCHMARKS))
    if a.fast:
        names = [n for n in names if not BENCHMARKS[n].get("slow")]

    # In-worktree measurer: emit JSON and exit.
    if a.emit_json:
        print(json.dumps(_measure(a.root, names)))
        return

    # Local mode: measure the working tree, print a table.
    if a.local:
        print_local_table(_measure(str(ROOT), names), names)
        return

    # Comparison mode: base vs PR over worktrees.
    base = a.base if a.base not in BENCHMARKS else "main"
    pr = a.pr if (a.pr and a.pr not in BENCHMARKS) else subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    preflight(base, allow_stale=a.allow_stale_base)
    session = time.strftime("%Y-%m-%d_%H-%M-%S")
    wt_root = Path(f"/tmp/count_queries_{session}")
    wt_base, wt_pr = wt_root / "base", wt_root / "pr"
    print(f"base={base}  pr={pr}  tasks={names}  session={session}", flush=True)
    try:
        setup_worktree(base, wt_base)
        setup_worktree(pr, wt_pr)
        base_res = measure_worktree(wt_base, names)
        pr_res = measure_worktree(wt_pr, names)
    finally:
        teardown_worktree(wt_base)
        teardown_worktree(wt_pr)
    report = comparison_report(base, pr, base_res, pr_res, names)
    print("\n" + report)
    if not a.no_pr:
        update_pr_report(pr, report)


if __name__ == "__main__":
    main()
