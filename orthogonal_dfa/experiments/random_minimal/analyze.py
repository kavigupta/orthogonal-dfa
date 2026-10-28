"""
Render E-L* on random minimal DFAs for a paper, from the results `run` caches:

1. summary.tex -- per target size and noise level, how often E-L* learns the
   target and learns it exactly, the lowest accuracy, and what it costs in
   membership queries.
2. queries_vs_eta.pdf -- every run's queries against eta, with a power law in
   1 - 2 eta per size fit to log10 queries.
3. difficulty.pdf -- how far each run's log10 queries sits above its size's
   fit, against some of the target's `difficulty.target_metrics`.

    python -m orthogonal_dfa.experiments.random_minimal.analyze
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from matplotlib import ticker

from ..capal_comparison.core import REPO_ROOT
from ..capal_comparison.figures import SUCCESS_THRESHOLD, _pct, _sci, tabular
from .run import DATA_DIR, RESULT_SCHEMA_VERSION

OUT_DIR = REPO_ROOT / "figures" / "random-minimal"

# metric: (axis label, x scale), where an integer metric is jittered
DIFFICULTY_METRICS = {
    "endpoint_mass_min": ("rarest state's share of length-40 endpoints", "log"),
    "class_preserving_fraction": ("class-preserving fraction", "log"),
    "separating_length_max": ("longest shortest separating suffix", "integer"),
    "synchronizing_probability": ("probability a string synchronizes", "linear"),
}
# the paper's 1F5FA8, newwitness and darkdisagree
SIZE_COLORS = ["#1F5FA8", "#BF8330", "#A62A5A"]


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
    scored = [r for r in runs if r["accuracy"] is not None]
    queries = [r["queries_total"] for r in scored]
    return [
        _pct(np.mean([succeeded(r) for r in runs])),
        _pct(np.mean([r["accuracy"] == 1.0 for r in runs])),
        f"{min(r['accuracy'] for r in scored):.3f}" if scored else "--",
        _sci(np.median(queries)) if queries else "--",
        _sci(np.quantile(queries, 0.9)) if queries else "--",
    ]


def summary_rows(results: Sequence[dict]) -> List[Union[List[str], str]]:
    return _by_size(results, _summary_cells)


def _scored(results: Sequence[dict]) -> List[dict]:
    return [r for r in results if r["accuracy"] is not None]


def eta_fit(results: Sequence[dict]) -> Dict[int, Tuple[float, float]]:
    """Per size, the least-squares (scale, power) of

        log10 queries = scale - power * log10(1 - 2 eta)

    over the runs with an accuracy."""
    fits = {}
    for states in sorted({r["target_states"] for r in results}):
        runs = [r for r in _scored(results) if r["target_states"] == states]
        power, scale = np.polyfit(
            [-np.log10(1 - 2 * r["eta"]) for r in runs],
            np.log10([r["queries_total"] for r in runs]),
            1,
        )
        fits[states] = float(scale), float(power)
    return fits


def _fitted(fit: Tuple[float, float], eta):
    scale, power = fit
    return scale - power * np.log10(1 - 2 * eta)


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)


def plot_queries_vs_eta(results: Sequence[dict]):
    fits = eta_fit(results)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for color, (states, fit) in zip(SIZE_COLORS, fits.items()):
        runs = [r for r in _scored(results) if r["target_states"] == states]
        etas = np.array([r["eta"] for r in runs])
        ax.scatter(
            etas + rng.uniform(-0.006, 0.006, len(etas)),
            [r["queries_total"] for r in runs],
            s=4,
            color=color,
            alpha=0.35,
            linewidths=0,
        )
        line = np.linspace(etas.min(), etas.max(), 100)
        ax.plot(
            line,
            10 ** _fitted(fit, line),
            color=color,
            linewidth=1.5,
            label=f"$|Q| = {states}$",
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("MQs")
    ax.legend(frameon=False, fontsize=7)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_difficulty(results: Sequence[dict]):
    fits = eta_fit(results)
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.8), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (metric, (label, scale)) in zip(axes.flat, DIFFICULTY_METRICS.items()):
        for color, (states, fit) in zip(SIZE_COLORS, fits.items()):
            runs = [r for r in _scored(results) if r["target_states"] == states]
            values = np.array([r["target"][metric] for r in runs])
            excess = [
                np.log10(r["queries_total"]) - _fitted(fit, r["eta"]) for r in runs
            ]
            rho = scipy.stats.spearmanr(values, excess).statistic
            ax.scatter(
                values + (scale == "integer") * rng.uniform(-0.15, 0.15, len(values)),
                excess,
                s=4,
                color=color,
                alpha=0.35,
                linewidths=0,
                label=f"$|Q| = {states}$, $\\rho = {rho:+.2f}$",
            )
        if scale == "log":
            ax.set_xscale("log")
            decades = np.log10(ax.get_xlim()[1] / ax.get_xlim()[0])
            subs = (1, 2, 5) if decades < 1.5 else (1,)
            ax.xaxis.set_major_locator(ticker.LogLocator(subs=subs))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.axhline(0, color="0.5", linewidth=0.5)
        ax.set_xlabel(label)
        legend = ax.legend(frameon=False, fontsize=7, markerscale=2)
        for handle in legend.legend_handles:
            handle.set_alpha(1)
        _style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\log_{10}$ queries above fit")
    fig.tight_layout()
    return fig


def fragments(results: Sequence[dict]) -> Dict[str, str]:
    return {
        "summary": tabular(
            [
                "States",
                r"$\eta$",
                "Success",
                "Exact",
                "Lowest accuracy",
                "MQ median",
                "MQ p90",
            ],
            "llrrrrr",
            summary_rows(results),
        ),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results(DATA_DIR)
    for name, fragment in fragments(results).items():
        tex_path = OUT_DIR / f"{name}.tex"
        tex_path.write_text(fragment + "\n")
        print(f"wrote {tex_path}")
    for name, plot in [
        ("queries_vs_eta", plot_queries_vs_eta),
        ("difficulty", plot_difficulty),
    ]:
        fig = plot(results)
        pdf_path = OUT_DIR / f"{name}.pdf"
        # no creation date, so rerunning on the same results leaves the file alone
        fig.savefig(pdf_path, metadata={"CreationDate": None})
        plt.close(fig)
        print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
