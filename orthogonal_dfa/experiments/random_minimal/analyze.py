"""
Tables over the results `run` caches: per target size and noise level, how
often E-L* learns the target and what it costs.

    python -m orthogonal_dfa.experiments.random_minimal.analyze

Writes ``summary.md`` and ``summary.csv`` next to the results.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .run import DATA_DIR, RESULT_SCHEMA_VERSION

LEARNED_ACCURACY = 0.99

COLUMNS = [
    "states",
    "eta",
    "runs",
    "learned",
    "exact",
    "right_size",
    "accuracy_min",
    "accuracy_median",
    "queries_median",
    "queries_p90",
    "seconds_median",
    "seconds_max",
    "errors",
]


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


def _quantile(values: Sequence[float], q: float):
    return None if not values else float(np.quantile(values, q))


def summarize(results: Sequence[dict]) -> List[Dict]:
    """One row per (states, eta). A run without an accuracy, from an error or
    a timeout, counts as not learned."""
    groups = defaultdict(list)
    for r in results:
        groups[r["target_states"], r["eta"]].append(r)
    rows = []
    for (states, eta), runs in sorted(groups.items()):
        accuracy = [r["accuracy"] for r in runs if r["accuracy"] is not None]
        queries = [r["queries_total"] for r in runs if r["accuracy"] is not None]
        seconds = [r["seconds"] for r in runs]
        rows.append(
            {
                "states": states,
                "eta": eta,
                "runs": len(runs),
                "learned": sum(a >= LEARNED_ACCURACY for a in accuracy) / len(runs),
                "exact": sum(a == 1.0 for a in accuracy) / len(runs),
                "right_size": np.mean([r["learned_states"] == states for r in runs]),
                "accuracy_min": min(accuracy, default=None),
                "accuracy_median": _quantile(accuracy, 0.5),
                "queries_median": _quantile(queries, 0.5),
                "queries_p90": _quantile(queries, 0.9),
                "seconds_median": _quantile(seconds, 0.5),
                "seconds_max": max(seconds),
                "errors": sum(r["error_type"] is not None for r in runs),
            }
        )
    return rows


def failures(results: Sequence[dict]) -> List[Dict]:
    """The runs short of `LEARNED_ACCURACY`, worst first."""
    failed = [
        r for r in results if r["accuracy"] is None or r["accuracy"] < LEARNED_ACCURACY
    ]
    failed.sort(key=lambda r: (-1 if r["accuracy"] is None else r["accuracy"]))
    return [
        {
            "target": r["benchmark"],
            "eta": r["eta"],
            "seed": r["seed"],
            "accuracy": r["accuracy"],
            "learned_states": r["learned_states"],
            "error": r["error_type"],
        }
        for r in failed
    ]


def _format(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{value:.3g}"
    return str(value)


def markdown_table(rows: Sequence[Dict], columns: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    results = load_results(args.data_dir)
    rows = summarize(results)
    failed = failures(results)
    with (args.data_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    report = (
        f"# E-L* on random minimal DFAs\n\n{len(results)} runs. Learned means "
        f"accuracy at least {LEARNED_ACCURACY}; right_size means the learned DFA "
        f"has the target's number of states.\n\n"
        + markdown_table(rows, COLUMNS)
        + f"\n\n## Runs not learned ({len(failed)})\n\n"
        + (markdown_table(failed, list(failed[0])) if failed else "None.")
        + "\n"
    )
    (args.data_dir / "summary.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
