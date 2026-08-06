#!/usr/bin/env python3
"""Render the CAPAL vs E-L* summary tables as LaTeX, and as images of it.

Two tables, both read from `data/capal/*.json`:

1. success rates -- how often each learner reaches SUCCESS_THRESHOLD, with
   E-L* scored only over the targets its preconditions admit, plus how many
   targets that is.
2. query cost -- what each learner spends on the targets *both* solve, so the
   ratio is like for like. Rows with no such target are dropped: two numbers
   from disjoint target sets do not make a comparison.

Each table is written twice: a `\\tabular` fragment to `\\input` into a paper,
and a PNG of it to look at.

    python -m orthogonal_dfa.experiments.capal_comparison.figures
"""

from __future__ import annotations

import collections
import json
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .core import LEARNER_CAPAL, LEARNER_ELSTAR, REPO_ROOT

#: Accuracy at which a hypothesis counts as a success. E-L* synthesises to
#: `learn.DEFAULT_ACC_THRESHOLD`, so holding either learner to exactness would
#: be holding it to a bar neither aims at.
SUCCESS_THRESHOLD = 0.98

DATA_DIR = REPO_ROOT / "data" / "capal"
OUT_DIR = REPO_ROOT / "figures" / "capal-comparison"

EXPERIMENTS = ["capal_benchmarks", "our_benchmarks"]
SOURCE_NAMES = {"capal_dataset": "CAPAL suite", "ours": "This repo"}
SOURCE_ORDER = ["capal_dataset", "ours"]


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


def cost_rows(cells: Sequence[dict]) -> List[List[str]]:
    idx = _grouped(cells)
    targets, etas = _axes(cells)
    rows = []
    for family in SOURCE_ORDER:
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
            rows.append(
                [
                    SOURCE_NAMES[family],
                    f"{eta:g}",
                    str(len(both)),
                    _sci(capal_mq),
                    f"{capal_eq:.1f}",
                    _sci(elstar_mq),
                    _sci(elstar_mq / capal_mq),
                ]
            )
    return rows


def tabular(headers: Sequence[str], align: str, rows: Sequence[Sequence[str]]) -> str:
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
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


def render(fragment: str, png_path: Path) -> bool:
    """Compile `fragment` standalone and write a PNG beside it.

    Returns False when there is no LaTeX toolchain, so the fragments still get
    written on a machine that cannot render them.
    """
    if not (shutil.which("pdflatex") and shutil.which("pdftoppm")):
        return False
    document = "\n".join(
        [
            r"\documentclass[border=6pt]{standalone}",
            r"\usepackage{booktabs}",
            r"\begin{document}",
            fragment,
            r"\end{document}",
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        tex = Path(tmp) / "table.tex"
        tex.write_text(document)
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
            cwd=tmp,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-r",
                "300",
                "-singlefile",
                "table.pdf",
                png_path.stem,
            ],
            cwd=tmp,
            check=True,
            capture_output=True,
        )
        shutil.move(str(Path(tmp) / f"{png_path.stem}.png"), png_path)
    return True


TABLES: List[Dict[str, Any]] = [
    {
        "name": "success_rates",
        "headers": [
            "Benchmark source",
            r"$\eta$",
            "CAPAL success",
            "E-L* success",
            "E-L* applicable",
        ],
        "align": "llrrr",
        "rows": success_rows,
    },
    {
        "name": "query_cost",
        "headers": [
            "Benchmark source",
            r"$\eta$",
            "$n$",
            "CAPAL MQ",
            "CAPAL EQ",
            "E-L* MQ",
            "Ratio",
        ],
        "align": "llrrrrr",
        "rows": cost_rows,
    },
]


def main() -> None:
    cells = load_cells()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in TABLES:
        rows: Callable[[Sequence[dict]], List[List[str]]] = spec["rows"]
        fragment = tabular(spec["headers"], spec["align"], rows(cells))
        tex_path = OUT_DIR / f"{spec['name']}.tex"
        png_path = OUT_DIR / f"{spec['name']}.png"
        tex_path.write_text(fragment + "\n")
        rendered = render(fragment, png_path)
        print(
            f"wrote {tex_path}"
            + (
                f" and {png_path}"
                if rendered
                else " (no LaTeX toolchain; skipped the PNG)"
            )
        )


if __name__ == "__main__":
    main()
