#!/usr/bin/env python3
"""Benchmark a PR by counting oracle membership queries on the L* synthesis
tasks, comparing the current branch against a base branch.

Modelled on egg-stitch's ``scripts/bench_pr.py``: it checks out ``BASE`` and the
``PR`` branch into two ephemeral git worktrees and drives all measurements from a
single process, running *this same script* as an in-worktree measurer
(``--emit-json --root <wt>``) so the harness is identical on both sides and only
the imported ``orthogonal_dfa`` library differs. No ``git checkout`` happens in
the main repo.

A run is deterministic given its seed, but one seed is one draw of a noisy
distribution, so each task runs ``NUM_SEEDS`` seeds and reports the geomean of its
queries over them, the worst accuracy, and the learned-state counts. The comparison
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
import concurrent.futures
import contextlib
import io
import json
import math
import os
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
    # Overlaps itself, so its back edges don't all return to the start; most strings accept.
    "overlap":    {"oracle": {"kind": "regex", "regex": r".*11011.*"},
                   "signal": 0.3, "symbols": 2, "noise": None},
    # --- guard tasks: the cells only ortho-L* solves; a query win must not break these ---
    "modulo_hard": {"oracle": {"kind": "modulo", "modulo": 9, "allowed": [3, 6]},
                    "signal": 0.2, "symbols": 2, "noise": None, "slow": True},  # η=0.30 wall
    "modulo_asym": {"oracle": {"kind": "modulo", "modulo": 9, "allowed": [3, 6]},
                    "signal": 0.15, "symbols": 2,
                    "noise": {"p_0": 0.10, "p_1": 0.40}, "slow": True},  # non-straddling
}

# Regression bands: a query ratio inside +/- QUERY_BAND is "no change"; an accuracy
# drop beyond ACC_EPS is a regression regardless of the query win.
QUERY_BAND = 0.02
ACC_EPS = 0.005

# A call of N strings is costed at ceil(N / cap) forward passes
BATCH_CAPS = (32, 128, 1024)

NUM_SEEDS = 5


# ---------------------------------------------------------------------------
# Measurer: runs INSIDE a worktree (imports that branch's orthogonal_dfa).
# Imports are lazy so `--root` is on sys.path before the library is loaded.
# ---------------------------------------------------------------------------


def _measure(root: str, names: list[str], seeds: list[int]) -> dict:
    """``{task: {seed: result}}``, seeds as strings so it survives JSON."""
    sys.path.insert(0, root)
    from orthogonal_dfa.l_star.examples.bernoulli_parity import (
        BernoulliParityOracle, BernoulliRegex,
    )
    from orthogonal_dfa.l_star.structures import NoisyOracle
    from orthogonal_dfa.l_star.structures import Oracle, AsymmetricBernoulli
    from orthogonal_dfa.l_star.learn import learn_dfa
    from tests.lstar_common import DEFAULT_SAMPLER, evaluate_accuracy

    class CountingOracle(Oracle):
        def __init__(self, inner):
            self._inner = inner
            self.count = 0
            self.distinct = set()
            self.batches = []  # size of each issued call

        @property
        def alphabet_size(self):
            return self._inner.alphabet_size

        def membership_query(self, string):
            self.count += 1
            self.distinct.add(tuple(string))
            self.batches.append(1)
            return self._inner.membership_query(string)

        def membership_queries(self, strings):
            self.count += len(strings)
            self.distinct.update(tuple(s) for s in strings)
            self.batches.append(len(strings))
            return self._inner.membership_queries(strings)

    def build_creator(spec):
        kind = spec["kind"]
        if kind == "modulo":
            return lambda nm, s: NoisyOracle(BernoulliParityOracle(modulo=spec["modulo"], allowed_moduluses=tuple(spec["allowed"])), nm, s)
        if kind == "regex":
            return lambda nm, s: NoisyOracle(BernoulliRegex(regex=spec["regex"]), nm, s)
        raise ValueError(f"unknown oracle kind {kind!r}")

    def run(bench, seed):
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
            dfa = learn_dfa(
                counting_creator, min_signal_strength=bench["signal"], seed=seed,
                noise_model=noise_model)
            acc = evaluate_accuracy(
                dfa, creator, symbols=bench["symbols"], sampler=DEFAULT_SAMPLER)
        all_batches = [n for c in counters for n in c.batches]
        return {
            "queries": sum(c.count for c in counters),
            "distinct": len(set().union(*[c.distinct for c in counters])),
            "batches": len(all_batches),
            "forward_passes": {
                str(cap): sum(math.ceil(n / cap) for n in all_batches)
                for cap in BATCH_CAPS},
            "states": len(dfa.states),
            "accuracy": acc,
            "seconds": time.time() - t0,
        }

    return {name: {str(seed): run(BENCHMARKS[name], seed) for seed in seeds}
            for name in names}


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


def measure_worktrees(roots: list[Path], names: list[str]) -> list[dict]:
    """Per root, ``{task: {seed: result}}``, each (root, task, seed) run as its own
    in-worktree measurer, all in parallel."""

    def run(root, name, seed):
        out = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--emit-json",
             "--root", str(root), "--tasks", name, "--seeds", str(seed)],
            cwd=root, text=True, capture_output=True)
        if out.returncode != 0:
            raise SystemExit(f"count_queries: measurer failed in {root}:\n{out.stderr}")
        return root, name, json.loads(out.stdout)[name]

    results = {root: {name: {} for name in names} for root in roots}
    jobs = [(root, name, seed) for root in roots for name in names
            for seed in range(NUM_SEEDS)]
    with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as pool:
        for root, name, by_seed in pool.map(lambda job: run(*job), jobs):
            results[root][name].update(by_seed)
    return [results[root] for root in roots]


def _geomean(xs) -> float:
    xs = list(xs)
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def summarize(by_seed: dict) -> dict:
    """One task's runs as one row: geomeans over seeds of the counts, the worst
    accuracy, and the state count of each seed in seed order."""
    runs = [by_seed[s] for s in sorted(by_seed, key=int)]
    return {
        "queries": _geomean(r["queries"] for r in runs),
        "distinct": _geomean(r["distinct"] for r in runs),
        "batches": _geomean(r["batches"] for r in runs),
        "forward_passes": {
            str(cap): _geomean(r["forward_passes"][str(cap)] for r in runs)
            for cap in BATCH_CAPS},
        "states": [r["states"] for r in runs],
        "accuracy": min(r["accuracy"] for r in runs),
        "seconds": sum(r["seconds"] for r in runs),
    }


def _states_str(states: list[int]) -> str:
    return str(states[0]) if len(set(states)) == 1 else "/".join(map(str, states))


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
        f"*Queries and forward passes are geomeans over seeds 0–{NUM_SEEDS - 1}; "
        "accuracy is the worst seed's; states are per seed where they differ.*",
        "",
        f"|   | task | queries `{base_ref}` | queries `{pr_ref}` | ratio | "
        f"acc `{base_ref}` | acc `{pr_ref}` | states |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    ratios = []
    any_regression = False
    base = {name: summarize(base[name]) for name in names}
    pr = {name: summarize(pr[name]) for name in names}
    for name in names:
        b, p = base[name], pr[name]
        ratio = p["queries"] / b["queries"] if b["queries"] else float("inf")
        ratios.append(ratio)
        acc_ok = p["accuracy"] >= b["accuracy"] - ACC_EPS
        states_ok = p["states"] == b["states"]
        emoji = _emoji(ratio, acc_ok, states_ok)
        if emoji == "🔴":
            any_regression = True
        st = (_states_str(b["states"]) if states_ok else
              f"{_states_str(b['states'])}→{_states_str(p['states'])} ‼️")
        lines.append(
            f"| {emoji} | {name} | {b['queries']:,.0f} | {p['queries']:,.0f} | "
            f"{_ratio_str(ratio)} | "
            f"{b['accuracy']:.3f} | {p['accuracy']:.3f} | {st} |")
    geo = math.prod(ratios) ** (1 / len(ratios)) if ratios else float("nan")
    lines.append(
        f"| {'🟢' if geo < 1 - QUERY_BAND else '🔴' if geo > 1 + QUERY_BAND else '⚪'} "
        f"| **geomean** |  |  | **{_ratio_str(geo)}** |  |  |  |")
    lines.append("")
    lines.append("**❌ REGRESSION** (queries up or accuracy/states broke on ≥1 task)"
                 if any_regression else
                 "**✅ no regressions** (accuracy held, no task's queries rose beyond band)")

    lines.append("")
    lines.append(f"### Forward passes — `{pr_ref}` vs `{base_ref}` "
                 "(neural passes at each batch cap; `base → pr (ratio, pr packing)`)")
    lines.append("")
    lines.append("*Packing is `ceil(pr queries / cap) / pr fp` — the floor if every "
                 "call filled its batch. Below 100% the remaining passes are "
                 "under-filled calls, i.e. headroom from co-batching more call sites, "
                 "not irreducible work.*")
    lines.append("")
    lines.append("| task | " + " | ".join(f"fp@{c}" for c in BATCH_CAPS) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(BATCH_CAPS)) + "|")
    for name in names:
        b, p = base[name], pr[name]
        cells = []
        for cap in BATCH_CAPS:
            bf, pf = b["forward_passes"][str(cap)], p["forward_passes"][str(cap)]
            packed = math.ceil(p["queries"] / cap) / pf if pf else float("nan")
            ratio = pf / bf if bf else float("inf")
            cells.append(f"{bf:,.0f} → {pf:,.0f} ({_ratio_str(ratio)}, "
                         f"{100 * packed:.0f}% packed)")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _ratio_str(ratio: float) -> str:
    """pr/base as ``0.02x``. Two decimals, except below 0.01 (batching moves
    forward passes by 100x+) where that would print ``0.00x``."""
    return f"{ratio:.2f}x" if ratio >= 0.01 else f"{ratio:.2g}x"


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
    results = {name: summarize(results[name]) for name in names}
    print(f"\n===== QUERY COUNT SUMMARY (geomean over {NUM_SEEDS} seeds) =====")
    caps = "".join(f"{f'fp@{cap}':>11}" for cap in BATCH_CAPS)
    header = (f"{'task':<14}{'queries':>12}{'distinct':>11}{'batches':>10}{caps}"
              f"{'states':>8}{'acc':>8}{'sec':>8}")
    print(header)
    print("-" * len(header))
    for name in names:
        r = results[name]
        fps = "".join(f"{r['forward_passes'][str(cap)]:>11,.0f}" for cap in BATCH_CAPS)
        print(f"{name:<14}{r['queries']:>12,.0f}{r['distinct']:>11,.0f}{r['batches']:>10,.0f}"
              f"{fps}{_states_str(r['states']):>8}{r['accuracy']:>8.3f}{r['seconds']:>8.1f}")
    queries = sum(results[n]["queries"] for n in names)
    print(f"\nTOTAL queries: {queries:,.0f}")
    for cap in BATCH_CAPS:
        total = sum(results[n]["forward_passes"][str(cap)] for n in names)
        ideal = math.ceil(queries / cap)
        print(f"TOTAL forward passes @ batch {cap}: {total:,.0f} "
              f"(ideal {ideal:,}, {100 * ideal / total:.0f}% packed)")


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
    p.add_argument("--seeds", nargs="*", type=int, default=list(range(NUM_SEEDS)),
                   help=argparse.SUPPRESS)
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
        print(json.dumps(_measure(a.root, names, a.seeds)))
        return

    # Local mode: measure the working tree, print a table.
    if a.local:
        print_local_table(measure_worktrees([ROOT], names)[0], names)
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
        base_res, pr_res = measure_worktrees([wt_base, wt_pr], names)
    finally:
        teardown_worktree(wt_base)
        teardown_worktree(wt_pr)
    report = comparison_report(base, pr, base_res, pr_res, names)
    print("\n" + report)
    if not a.no_pr:
        update_pr_report(pr, report)


if __name__ == "__main__":
    main()
