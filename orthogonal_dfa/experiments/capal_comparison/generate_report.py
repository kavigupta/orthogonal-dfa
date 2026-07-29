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
from typing import Any, Dict, List

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


def _cell_index(cells: List[dict]) -> Dict[tuple, dict]:
    """(benchmark, eta, learner) -> cell, for the single-seed head-to-heads."""
    return {(c["benchmark"], c["eta"], c["learner"]): c for c in cells}


def _acc(c: dict) -> str:
    if c is None:
        return "-"
    if c.get("error_type") == "ExcludedOutOfRegime":
        return "excl"
    a = c.get("accuracy")
    return "-" if a is None else f"{a:.3f}"


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


def _eq(c: dict) -> str:
    """Equivalence queries -- absent before schema 4, so `-` means unrecorded,
    not zero."""
    if c is None or c.get("equivalence_queries") is None:
        return "-"
    return f"{c['equivalence_queries']:,}"


def capal_settings(exp: Dict[str, Any]) -> Dict[str, Any]:
    """The CAPAL knobs an experiment's cells were measured at."""
    for c in exp["cells"]:
        if c["learner"] == "CAPAL":
            return {k: v for k, v in c["learner_config"].items() if k != "eta_hat"}
    return {}


def _conv(c: dict) -> str:
    if c is None or c.get("converged") is None:
        return "-"
    return "yes" if c["converged"] else "no"


def head_to_head(exp: Dict[str, Any], order: List[str]) -> str:
    cells = exp["cells"]
    idx = _cell_index(cells)
    etas = exp["config"]["etas"]
    rows = []
    for name in order:
        for eta in etas:
            ca = idx.get((name, eta, "CAPAL"))
            el = idx.get((name, eta, "E-L*"))
            if ca is None and el is None:
                continue
            rows.append(
                [
                    name,
                    f"{eta:.2f}",
                    _acc(ca),
                    _conv(ca),
                    _mq(ca),
                    _eq(ca),
                    _acc(el),
                    _conv(el),
                    _mq(el),
                ]
            )
    return table(
        [
            "target",
            "η",
            "CAPAL acc",
            "conv",
            "CAPAL mq",
            "eq",
            "E-L* acc",
            "conv",
            "E-L* mq",
        ],
        rows,
    )


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
    hh = head_to_head(exp, inreg)
    return f"""## 1. CAPAL's own benchmark suite

Both learners on CAPAL's 28 shipped `.taf` targets (Simple/Normal/Difficult) at
η ∈ {{0.05, 0.10, 0.20, 0.30}}. This is CAPAL's home turf.

CAPAL solves **{solved}/{len(capal)}** cells at 100% accuracy. Every failure is
at η=0.30:

{table(["target", "η", "acc", "states"], fail_rows)}

E-L* is in its designed regime on only **{len(inreg)}/28** targets
({", ".join(inreg)}); the other {len(excl)} are recorded as reasoned
exclusions (acceptance imbalance / class-preservation / covered-accuracy
ceiling), not run. On the shared in-regime cells both are accurate, but the
query cost differs by orders of magnitude:

{hh}
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
    hh = head_to_head(exp, OUR_ORDER)
    conv = capal_convergence_by_eta(exp)
    ladder = ", ".join(f"η={eta:g} {v}" for eta, v in conv.items())
    el = [
        c
        for c in exp["cells"]
        if c["learner"] == "E-L*" and c.get("accuracy") is not None
    ]
    el_exact = sum(1 for c in el if c["accuracy"] == 1.0)
    return f"""## 2. This repo's benchmarks (head-to-head)

Both learners on the modulo-9 and regex oracles from `tests/test_lstar.py`.
These are longer targets (8-11 states) than CAPAL's suite, chosen to satisfy
E-L*'s preconditions.

CAPAL's convergence here is a clean function of the noise level ({ladder}); it
is not that these languages defeat it, but that noise does. E-L* reaches exact
accuracy on {el_exact}/{len(el)} of the cells it is in regime for, and is flat in
the noise -- and pays two to three orders of magnitude more membership queries
for it.

{hh}
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
    rate_rows = [[f"{e}", f"{statistics.mean(by_eta[e]):.2f}"] for e in sorted(by_eta)]

    return f"""## 3. The wall: full hyperparameter sweep
{settings_warning(exp, load("capal_benchmarks"))}
A full factorial over CAPAL's three real knobs -- `max_same_samples`,
`suffix_pool_len_max`, `alpha` -- across every cell, all four noise levels, and
three seeds ({len(cells)} runs). For each (cell, η), how many of the
{n_cfg} configs (knobs × seeds) converge:

{verdict}

**The wall is a property of the noise level, not the DFA.** At η=0.30 every
cell fails on all {n_cfg} configs; at η≤0.20 every cell -- modulo included --
is crackable by some config and seed, with the crack-rate falling monotonically
with noise. Convergence rate by η, over all configs:

{table(["η", "convergence rate"], rate_rows)}

The hyperparameters are near-neutral within the swept ranges (each knob value
moves the aggregate rate by <0.05); **η alone drives convergence from 75% to
0%.** The earlier impression that modulo is uniquely hard was an artifact of
sweeping only `max_same_samples`; adding pool/alpha cracks it at η≤0.20.
"""


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
    rows = []
    for name in OUR_ORDER:
        cs = g.get(name)
        if not cs:
            continue
        acc = statistics.mean(c["accuracy"] for c in cs)
        dq = int(statistics.mean(_n(c) for c in cs))
        nconv = sum(bool(c["converged"]) for c in cs)
        e = el.get(name)
        rows.append(
            [
                name,
                f"{acc:.3f}",
                f"{nconv}/{len(cs)}",
                f"{dq:,}",
                f"{e['accuracy']:.3f}" if e else "excl",
                _mq(e) if e else "-",
            ]
        )
    return f"""## 4. Matched query budget: the wall is structural
{settings_warning(exp, load("capal_benchmarks"))}
CAPAL with its suffix enumeration uncapped (`enum_depth={cfg['enum_depth']}`,
`extra_len_max={cfg['extra_len_max']}`, `suffix_pool_len_max={cfg['suffix_pool_len_max']}`,
`max_same_samples={cfg['max_same_samples']}`) on the η=0.30 wall cells, three
seeds, versus E-L*'s spend on the same cell:

{table(["cell", "CAPAL acc", "conv", "CAPAL distinct", "E-L* acc", "E-L* distinct"], rows)}

CAPAL never converges (0/3 everywhere) even at 0.08–2.45M distinct queries. On
modulo it spends **more** than E-L* and still fails, while E-L* succeeds at
100%; the regex cells plateau below E-L*'s budget and fail. Throwing queries at
CAPAL does not break the wall: the limiter is the pairwise SAMESTATE test shape,
not the label count.
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
    stale = " (still at the pre-change settings; see the banners above)"
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
- Sections 3 and 4 -- the hyperparameter wall and the matched-budget probe --
  predate the settings change{stale}. Their conclusions, including "the wall is
  a property of the noise level, not the DFA" and "the wall is structural, not a
  budget limit", are not supported by the current data until they are re-run.
- Single seed throughout. Individual cell verdicts move under re-measurement:
  raising CAPAL's budget to its authors' settings flipped cells in both
  directions, including one from 1.000 to 0.507. Read the per-cell numbers as
  indicative, not as settled.
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
