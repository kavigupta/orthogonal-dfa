#!/usr/bin/env python3
"""Render the CAPAL vs E-L* summary tables as LaTeX fragments to `\\input` into a paper.

Two tables, both read from `data/capal/*.json`:

1. success rates -- how often each learner reaches SUCCESS_THRESHOLD, with
   E-L* scored only over the targets its preconditions admit, plus how many
   targets that is.
2. query cost -- what each learner spends on the targets *both* solve, so the
   ratio is like for like. Rows with no such target are dropped: two numbers
   from disjoint target sets do not make a comparison.

Plus a grid of which learner solves each target E-L* admits at each eta.

    python -m orthogonal_dfa.experiments.capal_comparison.figures
"""

from __future__ import annotations

import collections
import json
import re
import statistics
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .core import LEARNER_CAPAL, LEARNER_ELSTAR, REPO_ROOT

#: Accuracy at which a hypothesis counts as a success. E-L* synthesises to
#: `learn.DEFAULT_ACC_THRESHOLD`, so holding either learner to exactness would
#: be holding it to a bar neither aims at.
SUCCESS_THRESHOLD = 0.98

DATA_DIR = REPO_ROOT / "data" / "capal"
OUT_DIR = REPO_ROOT / "figures" / "capal-comparison"

EXPERIMENTS = ["capal_benchmarks", "our_benchmarks"]
SOURCE_NAMES = {"capal_dataset": "CAPAL suite", "ours": "Our benchmarks"}
SOURCE_ORDER = ["capal_dataset", "ours"]


def load_experiment(name: str) -> List[dict]:
    """One experiment's cells, or an empty list if it has not been run."""
    path = DATA_DIR / f"{name}.json"
    return json.loads(path.read_text())["cells"] if path.exists() else []


def load_cells() -> List[dict]:
    out: List[dict] = []
    for name in EXPERIMENTS:
        out += json.loads((DATA_DIR / f"{name}.json").read_text())["cells"]
    return out


def succeeded(cell: Optional[dict]) -> bool:
    return (
        cell is not None
        and cell.get("accuracy") is not None
        and cell["accuracy"] >= SUCCESS_THRESHOLD
    )


def applicable(cell: dict) -> bool:
    return cell.get("error_type") != "ExcludedOutOfRegime"


def _sci(x: float) -> str:
    """Scientific notation as LaTeX math, e.g. $9.08 \\times 10^{2}$."""
    mantissa, exponent = f"{x:.2e}".split("e")
    return f"${mantissa} \\times 10^{{{int(exponent)}}}$"


def _pct(x: float) -> str:
    return f"{100 * x:.0f}\\%"


def _grouped(cells: Sequence[dict]) -> Dict[tuple, dict]:
    return {(c["family"], c["benchmark"], c["learner"], c["eta"]): c for c in cells}


def _axes(cells: Sequence[dict]) -> tuple:
    targets = collections.defaultdict(set)
    for c in cells:
        targets[c["family"]].add(c["benchmark"])
    etas = sorted({c["eta"] for c in cells})
    return targets, etas


def success_rows(cells: Sequence[dict]) -> List[List[str]]:
    idx = _grouped(cells)
    targets, etas = _axes(cells)
    rows = []
    for family in SOURCE_ORDER:
        for eta in etas:
            names = sorted(targets[family])
            capal = [idx[(family, t, LEARNER_CAPAL, eta)] for t in names]
            elstar = [idx[(family, t, LEARNER_ELSTAR, eta)] for t in names]
            admitted = [c for c in elstar if applicable(c)]
            rows.append(
                [
                    SOURCE_NAMES[family],
                    f"{eta:g}",
                    _pct(sum(map(succeeded, capal)) / len(capal)),
                    (
                        _pct(sum(map(succeeded, admitted)) / len(admitted))
                        if admitted
                        else "--"
                    ),
                    _pct(len(admitted) / len(names)),
                ]
            )
    return rows


def cost_rows(cells: Sequence[dict]) -> List[Union[List[str], str]]:
    idx = _grouped(cells)
    targets, etas = _axes(cells)
    rows: List[Union[List[str], str]] = []
    for family in SOURCE_ORDER:
        group = []
        for eta in etas:
            pairs = [
                (
                    idx[(family, t, LEARNER_CAPAL, eta)],
                    idx[(family, t, LEARNER_ELSTAR, eta)],
                )
                for t in sorted(targets[family])
            ]
            both = [(c, e) for c, e in pairs if succeeded(c) and succeeded(e)]
            if not both:
                continue
            capal_mq = statistics.mean(c["queries_total"] for c, _ in both)
            capal_eq = statistics.mean(c["equivalence_queries"] for c, _ in both)
            elstar_mq = statistics.mean(e["queries_total"] for _, e in both)
            group.append(
                [
                    f"{eta:g}",
                    _sci(capal_mq),
                    f"{capal_eq:.1f}",
                    _sci(elstar_mq),
                    f"{elstar_mq / capal_mq:,.0f}$\\times$",
                ]
            )
        if not group:
            continue
        if rows:
            rows.append(r"\midrule")
        # \multirow needs \usepackage{multirow} in the including document.
        source = f"\\multirow{{{len(group)}}}{{*}}{{{SOURCE_NAMES[family]}}}"
        rows.append([source] + group[0])
        rows += [[""] + row for row in group[1:]]
    return rows


def sweep_rows(_cells: Sequence[dict]) -> List[List[str]]:
    """Experiment 3: the hyperparameter grid, one row per configuration.

    The grid is the rows rather than a count of "configs that worked", so the
    table says what was actually varied. Cells are successes out of the runs at
    that (config, eta), which is targets x seeds.
    """
    cells = load_experiment("wall_sweep")
    if not cells:
        return []
    etas = sorted({c["eta"] for c in cells})
    grid = collections.defaultdict(list)
    for c in cells:
        lc = c["learner_config"]
        grid[
            (lc["max_same_samples"], lc["suffix_pool_len_max"], lc["alpha"], c["eta"])
        ].append(c)
    rows = []
    for m, pool, alpha in sorted({(m, p, a) for m, p, a, _ in grid}):
        # Bold marks upstream's own benchmark setting, which is also what
        # experiments 1, 2 and 4 run at.
        baseline = (m, pool, alpha) == (80, 10, 1e-3)

        def mark(value: Any, bold: bool = baseline) -> str:
            return f"\\textbf{{{value}}}" if bold else str(value)

        rows.append(
            [mark(m), mark(pool), mark(alpha)]
            + [
                mark(
                    f"{sum(map(succeeded, grid[(m, pool, alpha, e)]))}/"
                    f"{len(grid[(m, pool, alpha, e)])}"
                )
                for e in etas
            ]
        )
    return rows


def budget_rows(_cells: Sequence[dict]) -> List[List[str]]:
    """Experiment 4: CAPAL at E-L*'s own spend, per target.

    Matched and stalled cells are averaged separately. Only the matched ones are
    like-for-like; folding a stalled cell in would report CAPAL's score "at
    E-L*'s budget" for a run that never approached it.
    """
    cells = load_experiment("matched_budget")
    if not cells:
        return []
    elstar = {
        c["benchmark"]: c
        for c in load_experiment("our_benchmarks")
        if c["learner"] == LEARNER_ELSTAR and c["eta"] == 0.30
    }
    by_target = collections.defaultdict(list)
    for c in cells:
        by_target[c["benchmark"]].append(c)

    def mean_acc(subset: List[dict]) -> str:
        scored = [c["accuracy"] for c in subset if c["accuracy"] is not None]
        return f"{statistics.mean(scored):.3f}" if scored else "--"

    rows = []
    for name in sorted(by_target, key=lambda n: elstar[n]["queries_total"]):
        cs = by_target[name]
        matched = [c for c in cs if c["error_type"] == "BudgetExhausted"]
        stalled = [c for c in cs if c["error_type"] == "Stalled"]
        rows.append(
            [
                name.replace("_", r"\_"),
                _sci(elstar[name]["queries_total"]),
                f"{len(matched)}/{len(cs)}",
                f"{len(stalled)}/{len(cs)}",
                mean_acc(matched),
                mean_acc(stalled),
                f"{elstar[name]['accuracy']:.3f}",
            ]
        )
    return rows


def tabular(
    headers: Sequence[str], align: str, rows: Sequence[Union[Sequence[str], str]]
) -> str:
    """A row given as a string, such as a rule, goes in as it is."""
    body = "\n".join(r if isinstance(r, str) else " & ".join(r) + r" \\" for r in rows)
    return "\n".join(
        [
            f"\\begin{{tabular}}{{{align}}}",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
            body,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


#: Readable names for this repo's targets; CAPAL's are named like "Simple01".
OUR_LABELS = {
    "parity_mod9_allowed_3_6": r"$\#_1(x) \bmod 9 \in \{3, 6\}$",
    "regex_subseq_1010101": r"\texttt{.*1010101.*}",
    "regex_two_1111": r"\texttt{.*1111.*1111.*}",
    "regex_alt_1111_or_0000_11_3sym": r"\texttt{.*(1111|0000)11.*}, ternary",
    "regex_alt_111_or_000_3sym": r"\texttt{.*(111|000).*}, ternary",
}
DIFFICULTY_ORDER = ["Simple", "Normal", "Difficult"]
GRID_PANELS = [
    ("capal_benchmarks", "CAPAL suite"),
    ("our_benchmarks", "our benchmarks"),
]

# Each cell is a square split on its diagonal: CAPAL upper-left, E-L* lower-right.
# Rows run downward from y = 0, so panels with different row counts align at the top.
GRID_MACROS = r"""\definecolor{bgsolved}{HTML}{1F5FA8}
\definecolor{bgfailed}{HTML}{BF8330}
\def\bgcell#1#2#3#4{
    \fill[bg#3] (#1, -#2 - 1) rectangle +(1, 1);
    \fill[bg#4] (#1, -#2 - 1) -- (#1 + 1, -#2 - 1) -- (#1 + 1, -#2) -- cycle;
    \draw[white, line width=0.8pt] (#1, -#2 - 1) rectangle +(1, 1);
}
\def\bgtarget#1#2{\node[anchor=east, font=\scriptsize, inner sep=2pt] at (0, -#1 - 0.5) {#2};}
\def\bgeta#1#2{\node[anchor=base, font=\scriptsize, inner sep=0pt] at (#1 + 0.5, 0.25) {#2};}
\def\bgtitle#1#2{
    \node[anchor=base east, font=\scriptsize, inner sep=2pt] at (0, 0.25) {$\eta$};
    \draw[black!50] (0.1, 1.15) -- (#1 - 0.1, 1.15);
    \node[anchor=south, font=\scriptsize, inner sep=2pt] at (#1 / 2, 1.15) {#2};
}"""

GRID_LEGEND = r"""\begin{tikzpicture}[x=0.5cm, y=0.5cm, baseline=0pt, font=\scriptsize]
    \draw[black!50] (0, -1.5) rectangle (1.5, 0) (0, -1.5) -- (1.5, 0);
    \node[anchor=south, inner sep=2pt] at (0.75, 0) {CAPAL};
    \node[anchor=north, inner sep=2pt] at (0.75, -1.5) {\ours{}};
    \foreach \kind/\name [count=\i] in {solved/solved, failed/not solved} {
        \fill[bg\kind] (0, -2.5 - \i) rectangle +(0.8, 0.8);
        \node[anchor=west, inner sep=2pt] at (0.8, -2.1 - \i) {\name};
    }
\end{tikzpicture}"""


def target_label(name: str) -> str:
    if name in OUR_LABELS:
        return OUR_LABELS[name]
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", name)
    assert match, name
    return f"{match.group(1)} {int(match.group(2))}"


def target_order(name: str) -> tuple:
    kinds = [i for i, kind in enumerate(DIFFICULTY_ORDER) if name.startswith(kind)]
    return (kinds[0] if kinds else 0, name)


def grid_panel(cells: Sequence[dict], title: str) -> str:
    """One suite's targets E-L* admits as rows, eta as columns."""
    excluded = {c["benchmark"] for c in cells if not applicable(c)}
    targets = sorted({c["benchmark"] for c in cells} - excluded, key=target_order)
    etas = sorted({c["eta"] for c in cells})
    idx = {(c["benchmark"], c["learner"], c["eta"]): c for c in cells}

    def outcome(target: str, learner: str, eta: float) -> str:
        return "solved" if succeeded(idx[(target, learner, eta)]) else "failed"

    lines = [f"\\bgtitle{{{len(etas)}}}{{{title}}}"]
    lines += [f"\\bgeta{{{col}}}{{{eta:g}}}" for col, eta in enumerate(etas)]
    for row, target in enumerate(targets):
        lines.append(f"\\bgtarget{{{row}}}{{{target_label(target)}}}")
        for col, eta in enumerate(etas):
            capal = outcome(target, LEARNER_CAPAL, eta)
            elstar = outcome(target, LEARNER_ELSTAR, eta)
            lines.append(f"\\bgcell{{{col}}}{{{row}}}{{{capal}}}{{{elstar}}}")
    body = "\n".join("    " + line for line in lines)
    return (
        "\\begin{tikzpicture}[x=0.5cm, y=0.5cm, baseline=0pt]\n"
        + body
        + "\n\\end{tikzpicture}"
    )


def grid_fragment() -> str:
    """Whether each learner solves each target E-L* admits, at each eta."""
    panels = [grid_panel(load_experiment(name), title) for name, title in GRID_PANELS]
    return GRID_MACROS + "\n" + "\n\\qquad\n".join(panels + [GRID_LEGEND])


TABLES: List[Dict[str, Any]] = [
    {
        "name": "success_rates",
        "headers": [
            "Benchmark source",
            r"$\eta$",
            "CAPAL success",
            r"\ours{} success",
            r"\ours{} applicable",
        ],
        "align": "llrrr",
        "rows": success_rows,
    },
    {
        "name": "query_cost",
        "headers": [
            "Benchmark source",
            r"$\eta$",
            "CAPAL MQ",
            "CAPAL EQ",
            r"\ours{} MQ",
            "MQ ratio",
        ],
        "align": "llrrrr",
        "rows": cost_rows,
    },
    {
        "name": "hyperparameter_sweep",
        "headers": ["$m$", "pool", r"$\alpha$"]
        + [rf"$\eta$={e:g}" for e in (0.05, 0.1, 0.2, 0.3)],
        "align": "rrrrrrr",
        "rows": sweep_rows,
    },
    {
        "name": "matched_budget",
        "headers": [
            "Target",
            r"Budget (\ours{} MQ)",
            "matched",
            "stalled",
            "CAPAL (matched)",
            "CAPAL (stalled)",
            r"\ours{}",
        ],
        "align": "lrrrrrr",
        "rows": budget_rows,
    },
]


def main() -> None:
    cells = load_cells()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fragments = {}
    for spec in TABLES:
        rows: Callable[[Sequence[dict]], List[Union[List[str], str]]] = spec["rows"]
        fragments[spec["name"]] = tabular(spec["headers"], spec["align"], rows(cells))
    fragments["benchmark_grid"] = grid_fragment()
    for name, fragment in fragments.items():
        tex_path = OUT_DIR / f"{name}.tex"
        tex_path.write_text(fragment + "\n")
        print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
