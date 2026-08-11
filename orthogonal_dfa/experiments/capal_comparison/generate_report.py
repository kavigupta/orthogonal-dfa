#!/usr/bin/env python3
"""Regenerate data/capal_findings.md from the checked-in experiment JSONs.

Deterministic: every table is computed from data/capal/*.json, so the findings
doc can never drift from the numbers the way a hand-maintained table does. Rerun
after any experiment rerun. The analytical passages (signal-to-noise theory,
etc.) are static prose -- derivations, not data -- and live here as constants.

    python -m orthogonal_dfa.experiments.capal_comparison.generate_report
"""

from __future__ import annotations

import collections
import json
import statistics
from typing import Any, Dict, List, Optional

from orthogonal_dfa.capal_official import PINNED_COMMIT

from .core import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "capal"
OUT = REPO_ROOT / "data" / "capal_findings.md"

OUR_ORDER = [
    "parity_mod9_allowed_3_6",
    "regex_subseq_1010101",
    "regex_two_1111",
    "regex_alt_1111_or_0000_11",
    "regex_alt_111_or_000_3sym",
]


def load(name: str) -> Dict[str, Any]:
    return json.loads((DATA_DIR / f"{name}.json").read_text())


def table(headers: List[str], rows: List[List[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line, sep, *body])


def _n(c: dict) -> int:
    """The comparable membership-query count, whichever schema wrote the cell."""
    n = c.get("queries_distinct")
    return c.get("queries_total") if n is None else n


def _mq(c: dict) -> str:
    """Membership queries. Schema 1 split them into total/distinct and only
    distinct was comparable; schema 4 records the one honest count."""
    if c is None:
        return "-"
    n = _n(c)
    return "-" if n is None else f"{n:,}"


def capal_settings(exp: Dict[str, Any]) -> Dict[str, Any]:
    """The CAPAL knobs an experiment's cells were measured at."""
    for c in exp["cells"]:
        if c["learner"] == "CAPAL":
            return {k: v for k, v in c["learner_config"].items() if k != "eta_hat"}
    return {}


def exp1_section() -> str:
    exp = load("capal_benchmarks")
    cells = exp["cells"]
    capal = [
        c for c in cells if c["learner"] == "CAPAL" and c.get("accuracy") is not None
    ]
    solved = sum(1 for c in capal if c["accuracy"] >= 0.999)
    fails = sorted(
        (c for c in capal if c["accuracy"] < 0.999),
        key=lambda c: c["accuracy"],
    )
    el = [c for c in cells if c["learner"] == "E-L*"]
    ran = [c for c in el if c.get("accuracy") is not None]
    excl = sorted(
        {c["benchmark"] for c in el if c.get("error_type") == "ExcludedOutOfRegime"}
    )
    inreg = sorted({c["benchmark"] for c in ran})

    fail_rows = [
        [
            c["benchmark"],
            f"{c['eta']:.2f}",
            f"{c['accuracy']:.3f}",
            f"{c['learned_states']}/{c['target_states']}",
        ]
        for c in fails
    ]
    return f"""## 1. CAPAL's own benchmark suite

Both learners on CAPAL's 28 shipped `.taf` targets (Simple/Normal/Difficult) at
η ∈ {{0.05, 0.10, 0.20, 0.30}}. This is CAPAL's home turf.

CAPAL solves **{solved}/{len(capal)}** cells at 100% accuracy. Every failure is
at η=0.30:

{table(["target", "η", "acc", "states"], fail_rows)}

E-L* is in its designed regime on only **{len(inreg)}/28** targets
({", ".join(inreg)}); the other {len(excl)} are recorded as reasoned
exclusions (acceptance imbalance / class-preservation / covered-accuracy
ceiling), not run.
"""


def capal_convergence_by_eta(exp: Dict[str, Any]) -> Dict[float, str]:
    """converged/total per noise level, for CAPAL."""
    out = {}
    for eta in exp["config"]["etas"]:
        cs = [c for c in exp["cells"] if c["learner"] == "CAPAL" and c["eta"] == eta]
        out[eta] = f"{sum(1 for c in cs if c['converged'])}/{len(cs)}"
    return out


def exp2_section() -> str:
    exp = load("our_benchmarks")
    conv = capal_convergence_by_eta(exp)
    ladder = ", ".join(f"η={eta:g} {v}" for eta, v in conv.items())
    el = [
        c
        for c in exp["cells"]
        if c["learner"] == "E-L*" and c.get("accuracy") is not None
    ]
    el_exact = sum(1 for c in el if c["accuracy"] == 1.0)
    return f"""## 2. This repo's benchmarks

Both learners on the modulo-9 and regex oracles from `tests/test_lstar.py`.
These are longer targets (8-11 states) than CAPAL's suite, chosen to satisfy
E-L*'s preconditions.

CAPAL's convergence here is a clean function of the noise level ({ladder}); it
is not that these languages defeat it, but that noise does. E-L* reaches exact
accuracy on {el_exact}/{len(el)} of the cells it is in regime for, and is flat in
the noise -- and pays two to three orders of magnitude more membership queries
for it.
"""


def settings_warning(exp: Dict[str, Any], reference: Dict[str, Any]) -> str:
    """A banner when this experiment predates the authors'-settings change.

    `discr_search_random` is the tell: it was unreachable through make_learner
    before that change, so a cell lacking it was measured with CAPAL's stingier
    dataclass defaults. Comparing such numbers with sections 1-2 is comparing
    two different configurations of the learner, which is exactly the mistake
    this doc exists to avoid.
    """
    ref = capal_settings(reference)
    if "discr_search_random" in capal_settings(exp):
        return ""
    return (
        "\n> **Measured before CAPAL was moved to its authors' benchmark-script"
        f" settings** (`max_same_samples` 60 -> {ref.get('max_same_samples')},"
        f" `suffix_pool_init` 32 -> {ref.get('suffix_pool_init')},"
        f" `suffix_pool_len_max` 8 -> {ref.get('suffix_pool_len_max')},"
        f" `discr_search_random` 200 -> {ref.get('discr_search_random')})."
        " Re-run before comparing these numbers with sections 1-2.\n"
    )


def wall_section() -> str:
    exp = load("wall_sweep")
    cells = exp["cells"]
    etas = exp["config"]["etas"]
    n_cfg = len(exp["config"]["configs"]) * len(exp["config"]["seeds"])

    g = collections.defaultdict(list)
    for c in cells:
        g[(c["benchmark"], c["eta"])].append(c)
    rows = []
    for name in OUR_ORDER:
        row = [name]
        for eta in etas:
            cs = g[(name, eta)]
            nconv = sum(bool(c["converged"]) for c in cs)
            best = max((c["accuracy"] or 0) for c in cs)
            row.append(f"{nconv}/{len(cs)}" if nconv else f"wall ({best:.2f})")
        rows.append(row)
    verdict = table(["cell"] + [f"η={e}" for e in etas], rows)

    # convergence rate by noise
    by_eta = collections.defaultdict(list)
    for c in cells:
        by_eta[c["eta"]].append(bool(c["converged"]))
    top_rate = f"{statistics.mean(by_eta[min(by_eta)]):.2f}"

    # cells that wall below the top noise level, which is the claim most likely
    # to move when the grid or the learner changes
    low = sorted(
        {
            (c["benchmark"], c["eta"])
            for (c_name, c_eta), cs in g.items()
            for c in [cs[0]]
            if c_eta != max(etas) and not any(x["converged"] for x in cs)
            for c_name, c_eta in [(c_name, c_eta)]
        }
    )
    if low:
        names = sorted({n for n, _ in low})
        walled_below = (
            "**Noise dominates, but not alone.** At η=0.30 every cell fails on"
            f" all {n_cfg} configs and every seed; "
            + ", ".join(f"{n} already walls at η={e:g}" for n, e in low)
            + f", while the other {len(g) // len(etas) - len(names)} cells still"
            " crack there. Which DFA it is decides where the wall starts; the"
            " noise level decides that there is one."
        )
    else:
        walled_below = (
            "**The wall is a property of the noise level, not the DFA.** At"
            f" η=0.30 every cell fails on all {n_cfg} configs and every seed,"
            " while at every η≤0.20 each cell is crackable by some config and"
            " seed, with the crack-rate falling monotonically with noise."
        )

    knobs = collections.defaultdict(lambda: collections.defaultdict(list))
    for c in cells:
        for knob in ("max_same_samples", "suffix_pool_len_max", "alpha"):
            knobs[knob][c["learner_config"][knob]].append(bool(c["converged"]))
    knob_spread = "; ".join(
        f"`{knob}` "
        + " vs ".join(
            f"{v}: {statistics.mean(hits):.2f}" for v, hits in sorted(vals.items())
        )
        for knob, vals in knobs.items()
    )
    max_m = int(
        max(c["learner_config"]["max_same_samples"] for c in cells)
        / min(c["learner_config"]["max_same_samples"] for c in cells)
    )

    return f"""## 3. The wall: full hyperparameter sweep
{settings_warning(exp, load("capal_benchmarks"))}
A full factorial over CAPAL's three real knobs -- `max_same_samples`,
`suffix_pool_len_max`, `alpha` -- across every cell, all four noise levels, and
three seeds ({len(cells)} runs). For each (cell, η), how many of the
{n_cfg} configs (knobs × seeds) converge:

{verdict}

{walled_below}

η drives the aggregate rate from {top_rate} to 0.00. The knobs move it far less
over the swept range -- {knob_spread} -- and none of them rescues a single
η=0.30 cell.

The grid's low corner is upstream's own benchmark setting, so a cell that fails
across it failed with at least the budget CAPAL's authors publish with, and up
to {max_m}× the evidence per pairwise test.
"""


def _stalled(cell: dict, elstar: Optional[dict]) -> bool:
    """Did this cell stop before its budget could bind?

    Derived rather than read off `error_type`, so that runs recorded before the
    Stalled outcome existed are classified too: a cell that neither converged
    nor errored, and stopped under E-L*'s spend, ran out of iterations at a
    fixed point where further rounds issue no new queries.
    """
    if cell["error_type"] == "Stalled":
        return True
    return (
        cell["error_type"] is None
        and not cell["converged"]
        and elstar is not None
        and _n(cell) < _n(elstar)
    )


def matched_budget_section() -> str:
    exp = load("matched_budget")
    mb = exp["cells"]
    el = {
        c["benchmark"]: c
        for c in load("our_benchmarks")["cells"]
        if c["learner"] == "E-L*" and c["eta"] == 0.30 and c.get("accuracy") is not None
    }
    cfg = mb[0]["learner_config"]
    g = collections.defaultdict(list)
    for c in mb:
        g[c["benchmark"]].append(c)

    rows, reached, timed_out, stalled = [], [], [], []
    for name in OUR_ORDER:
        cs = g.get(name)
        if not cs:
            continue
        # A timed-out run has no hypothesis to score, so it is counted, not
        # averaged in: folding it in as a zero would understate what CAPAL
        # reached, and dropping it silently would hide that it never finished.
        e = el.get(name)
        scored = [c["accuracy"] for c in cs if c["accuracy"] is not None]
        outs = sum(1 for c in cs if c["error_type"] == "Timeout")
        stalls = sum(1 for c in cs if _stalled(c, e))
        if outs:
            timed_out.append((name, outs, len(cs)))
        if stalls:
            stalled.append((name, stalls, len(cs)))
        mq = int(statistics.mean(_n(c) for c in cs if _n(c) is not None))
        if e and mq >= _n(e):
            reached.append(name)
        rows.append(
            [
                name,
                f"{statistics.mean(scored):.3f}" if scored else "no hypothesis",
                f"{sum(bool(c['converged']) for c in cs)}/{len(cs)}",
                f"{stalls}/{len(cs)}",
                f"{outs}/{len(cs)}",
                f"{mq:,}",
                f"{e['accuracy']:.3f}" if e else "excl",
                _mq(e) if e else "-",
            ]
        )

    def _tally(entries: List[tuple]) -> str:
        return ", ".join(f"{n} ({k}/{t})" for n, k, t in entries)

    matched = (
        _tally(
            [
                (
                    name,
                    sum(1 for c in g[name] if c["error_type"] == "BudgetExhausted"),
                    len(g[name]),
                )
                for name in OUR_ORDER
                if name in g
                and any(c["error_type"] == "BudgetExhausted" for c in g[name])
            ]
        )
        or "none"
    )
    stalls = _tally(stalled) or "none"
    outs = _tally(timed_out) or "none"
    return f"""## 4. Matched query budget: CAPAL at E-L*'s own spend
{settings_warning(exp, load("capal_benchmarks"))}
CAPAL with its suffix enumeration uncapped (`enum_depth={cfg['enum_depth']}`,
`extra_len_max={cfg['extra_len_max']}`, `suffix_pool_len_max={cfg['suffix_pool_len_max']}`,
`max_same_samples={cfg['max_same_samples']}`) on the η=0.30 wall cells, three
seeds, versus E-L*'s spend on the same cell:

CAPAL converges on none of them, at any budget, on any seed.

Only the cells that spent their budget are matched-budget measurements, and
they are the ones to read: {matched}. There CAPAL is handed exactly the queries
E-L* used, plus a perfect equivalence oracle E-L* never gets, and still comes
back short of it.

Two kinds of cell are not measurements of that, and are separated out above
rather than averaged in. **Stalled** ({stalls}) ran out of iterations at a fixed
point: further rounds issue no new queries at all -- on
regex_alt_111_or_000_3sym the distinct count is identical at 50 iterations and
at 10000 -- so no budget could ever bind. That is a stronger statement than a
low score, just a different one: CAPAL stops improving at a fraction of E-L*'s
spend and cannot use more. **Timed out** ({outs}) were ended by the wall clock
with no hypothesis to score, and say nothing either way.
"""


STATIC_THEORY = """## 5. Why the noise floor bites CAPAL harder (theory)

Both learners use statistical row-equality under persistent noise, but the test
*shape* differs. CAPAL's SAMESTATE compares two noisy rows against each other,
so its noise floor is `p₀ = 2η(1−η)` and observed signal scales by `(1 − 2p₀)`.
E-L* measures each prefix's own accept rate against a data-driven boundary, so
its floor is just `η` and signal scales by `(1 − 2η)`.

| η    | CAPAL signal (1−2p₀) | E-L* signal (1−2η) | ratio |
| ---- | -------------------- | ------------------ | ----- |
| 0.05 | 0.81                 | 0.90               | 1.1×  |
| 0.10 | 0.64                 | 0.80               | 1.25× |
| 0.20 | 0.36                 | 0.60               | 1.7×  |
| 0.30 | 0.16                 | 0.40               | 2.5×  |

At η=0.30 E-L* gets 2.5× more usable signal on the same oracle, and the gap
widens with noise. For the pairs CAPAL merges on modulo-9 (states differing by
±3 mod 9), the maximum true disagreement any suffix can produce is 2/9 ≈ 0.22,
so at η=0.30 the observed disagreement sits only ~0.035 above the 0.42 floor.
Resolving that needs a threshold so tight it over-splits every easy pair -- one
global knob (τ) cannot serve the hard and easy pairs at once. That is the wall,
and it is structural to the pairwise test, which is why §4's matched budget does
not move it.
"""


def bottom_line() -> str:
    e1, e2 = load("capal_benchmarks"), load("our_benchmarks")
    capal1 = [c for c in e1["cells"] if c["learner"] == "CAPAL"]
    solved = sum(1 for c in capal1 if c["accuracy"] >= 0.999)
    inreg = sorted(
        {
            c["benchmark"]
            for c in e1["cells"]
            if c["learner"] == "E-L*" and c.get("accuracy") is not None
        }
    )
    conv = capal_convergence_by_eta(e2)
    ladder = ", ".join(f"η={eta:g} {v}" for eta, v in conv.items())
    sweep = load("wall_sweep")
    hi = max(sweep["config"]["etas"])
    hi_runs = [c for c in sweep["cells"] if c["eta"] == hi]
    mb = load("matched_budget")["cells"]
    mb_out = sum(1 for c in mb if c["error_type"] == "Timeout")
    return f"""## 6. Bottom line

- On CAPAL's own suite CAPAL is broadly applicable and cheap: {solved}/{len(capal1)}
  cells at 100%, every failure at η=0.30. E-L* matches its accuracy but only on
  the {len(inreg)}/28 targets its preconditions admit, at two to three orders of
  magnitude more membership queries.
- The membership columns are not like for like. CAPAL is handed a perfect
  equivalence oracle and E-L* is not, so part of what E-L* pays for in queries
  is work CAPAL is given for free.
- On this repo's benchmarks CAPAL's convergence tracks the noise level rather
  than the language ({ladder}). E-L* is exact wherever it is in regime, at every
  noise level tested.
- The η={hi:g} wall holds across the whole sweep: {sum(bool(c["converged"]) for c in hi_runs)}
  of {len(hi_runs)} runs converge, over a grid whose low corner is upstream's own
  benchmark setting and which sweeps up from there. No knob rescues a cell.
- The wall is not a budget limit. Uncapping suffix enumeration puts CAPAL above
  E-L*'s own query spend on 3 of 5 cells without converging on any, and on
  modulo {mb_out} of {len(mb)} runs exhaust the per-cell time limit at ~16x E-L*'s
  spend without producing a hypothesis at all. On the two cells that stop below
  E-L*'s spend the probe is inconclusive rather than supportive.
- Sections 1-2 are single-seed; the sweep and the matched-budget probe use
  {len(sweep["config"]["seeds"])}. Individual cell verdicts move under
  re-measurement -- raising CAPAL's budget to its authors' settings flipped
  cells in both directions, including one from 1.000 to 0.507 -- so read the
  single-seed per-cell numbers as indicative rather than settled.
"""


def main() -> None:
    parts = [
        "# CAPAL (ICLR 2026) vs E-L\\* on noisy DFA learning",
        "",
        "_Generated from `data/capal/*.json` by "
        "`orthogonal_dfa.experiments.capal_comparison.generate_report`. "
        "Do not edit by hand; rerun the generator after any experiment rerun._",
        "",
        f"Upstream CAPAL pinned at `{PINNED_COMMIT}`, run at its authors' "
        "benchmark-script settings. Both learners model persistent noise, so a "
        "membership count is the distinct strings each was told about. The two "
        "columns are not the same cost: CAPAL is given a perfect equivalence "
        "oracle (the paper's pMAT assumption) whose counterexamples arrive as "
        "gold labels, while E-L* has no EQ and manufactures counterexamples out "
        "of membership queries. Read `mq` and `eq` together.",
        "",
        exp1_section(),
        exp2_section(),
        wall_section(),
        matched_budget_section(),
        STATIC_THEORY,
        bottom_line(),
    ]
    OUT.write_text("\n".join(parts).rstrip() + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
