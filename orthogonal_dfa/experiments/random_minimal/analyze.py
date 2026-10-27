"""
Render E-L* on random minimal DFAs as LaTeX fragments to `\\input` into a paper.

Two tables, both read from the results `run` caches, one row per target size
and noise level:

1. summary -- how often E-L* learns the target, exactly and at the right size,
   and what it costs in membership queries.
2. difficulty -- the Spearman correlation between a target's queries and each
   of its `difficulty.target_metrics`.

    python -m orthogonal_dfa.experiments.random_minimal.analyze
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np
import scipy.stats

from ..capal_comparison.core import REPO_ROOT
from ..capal_comparison.figures import SUCCESS_THRESHOLD, _pct, _sci, tabular
from .run import DATA_DIR, RESULT_SCHEMA_VERSION

OUT_DIR = REPO_ROOT / "figures" / "random-minimal"

DIFFICULTY_METRICS = {
    "endpoint_mass_min": "rarest state",
    "class_preserving_fraction": "class-preserving",
    "separating_length_max": "longest separator",
    "synchronizing_probability": "synchronizing",
    "depth_max": "deepest state",
    "acceptance_rate": "acceptance",
}


def load_results(data_dir: Path) -> List[dict]:
    results = []
    stale = 0
    for path in sorted((data_dir / "results").glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
            stale += 1
            continue
        results.append(payload)
    if stale:
        print(f"skipped {stale} results from another schema version")
    return results


def succeeded(result: dict) -> bool:
    return result["accuracy"] is not None and result["accuracy"] >= SUCCESS_THRESHOLD


def _by_size(results: Sequence[dict], cells) -> List[Union[List[str], str]]:
    """``cells(runs)`` per (states, eta), with the size spanning its etas."""
    groups = defaultdict(list)
    for r in results:
        groups[r["target_states"], r["eta"]].append(r)
    rows: List[Union[List[str], str]] = []
    for states in sorted({s for s, _ in groups}):
        etas = sorted(e for s, e in groups if s == states)
        if rows:
            rows.append(r"\midrule")
        # \multirow needs \usepackage{multirow} in the including document.
        rows.append(
            [f"\\multirow{{{len(etas)}}}{{*}}{{{states}}}", f"{etas[0]:g}"]
            + cells(groups[states, etas[0]])
        )
        rows += [["", f"{eta:g}"] + cells(groups[states, eta]) for eta in etas[1:]]
    return rows


def _summary_cells(runs: Sequence[dict]) -> List[str]:
    """A run with an error or a timeout has no accuracy and counts as failed."""
    queries = [r["queries_total"] for r in runs if r["accuracy"] is not None]
    return [
        str(len(runs)),
        _pct(np.mean([succeeded(r) for r in runs])),
        _pct(np.mean([r["accuracy"] == 1.0 for r in runs])),
        _pct(np.mean([r["learned_states"] == r["target_states"] for r in runs])),
        _sci(np.median(queries)) if queries else "--",
        _sci(np.quantile(queries, 0.9)) if queries else "--",
    ]


def summary_rows(results: Sequence[dict]) -> List[Union[List[str], str]]:
    return _by_size(results, _summary_cells)


def _difficulty_cells(runs: Sequence[dict]) -> List[str]:
    """Over the runs with an accuracy; -- where a metric is constant."""
    scored = [r for r in runs if r["accuracy"] is not None]
    queries = [r["queries_total"] for r in scored]
    cells = []
    for metric in DIFFICULTY_METRICS:
        values = [r["target"][metric] for r in scored]
        if len(set(values)) < 2 or len(set(queries)) < 2:
            cells.append("--")
            continue
        rho = scipy.stats.spearmanr(values, queries).statistic
        cells.append(f"${rho:+.2f}$")
    return cells


def difficulty_rows(results: Sequence[dict]) -> List[Union[List[str], str]]:
    return _by_size(results, _difficulty_cells)


def fragments(results: Sequence[dict]) -> Dict[str, str]:
    return {
        "summary": tabular(
            [
                "States",
                r"$\eta$",
                "Runs",
                "Success",
                "Exact",
                "Right size",
                "MQ median",
                "MQ p90",
            ],
            "llrrrrrr",
            summary_rows(results),
        ),
        "difficulty": tabular(
            ["States", r"$\eta$"] + list(DIFFICULTY_METRICS.values()),
            "ll" + "r" * len(DIFFICULTY_METRICS),
            difficulty_rows(results),
        ),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fragment in fragments(load_results(DATA_DIR)).items():
        tex_path = OUT_DIR / f"{name}.tex"
        tex_path.write_text(fragment + "\n")
        print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
