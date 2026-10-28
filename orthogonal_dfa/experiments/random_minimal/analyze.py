"""
Render E-L* on random minimal DFAs for a paper, from the results `run` caches:

1. summary.tex -- per target size and noise level, how often E-L* learns the
   target and learns it exactly, the lowest accuracy, and what it costs in
   membership queries.
2. queries_vs_eta.pdf -- every run's queries against eta, with a power law in
   1 - 2 eta per size fit to log10 queries.
3. queries_vs_states.pdf -- every run's queries against the target's size,
   with a power law in the size per eta.
4. queries_vs_pap.pdf -- every run's queries against the target's
   class-preserving fraction p_AP, with a power law in p_AP per eta.
5. difficulty.pdf -- how far each run's log10 queries sits above its size's
   eta fit, against some of the target's `difficulty.target_metrics`.

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
    "class_preserving_fraction": (r"$p_{\mathrm{AP}}$", "log"),
    "separating_length_max": ("longest shortest separating suffix", "integer"),
    "synchronizing_probability": ("probability a string synchronizes", "linear"),
}
# the paper's 1F5FA8, newwitness, darkdisagree and violet!85!black
PAPER_COLORS = ["#1F5FA8", "#BF8330", "#A62A5A", "#6C006C"]
# Type 3 fonts, matplotlib's default, are rejected by ACM
PAPER_RC = {"font.size": 7, "legend.fontsize": 6, "pdf.fonttype": 42}


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


def _excess(fits: Dict[int, Tuple[float, float]], result: dict) -> float:
    """log10 queries over the eta fit of the result's size."""
    fit = fits[result["target_states"]]
    return np.log10(result["queries_total"]) - _fitted(fit, result["eta"])


def _power_fit_by_eta(results: Sequence[dict], x) -> Dict[float, Tuple[float, float]]:
    """Per eta, over every size, the least-squares (scale, power) of

        log10 queries = scale + power * log10 x(run)

    over the runs with an accuracy."""
    fits = {}
    for eta in sorted({r["eta"] for r in results}):
        runs = [r for r in _scored(results) if r["eta"] == eta]
        power, scale = np.polyfit(
            np.log10([x(r) for r in runs]),
            np.log10([r["queries_total"] for r in runs]),
            1,
        )
        fits[eta] = float(scale), float(power)
    return fits


def states_fit(results: Sequence[dict]) -> Dict[float, Tuple[float, float]]:
    return _power_fit_by_eta(results, lambda r: r["target_states"])


def pap_fit(results: Sequence[dict]) -> Dict[float, Tuple[float, float]]:
    return _power_fit_by_eta(
        results, lambda r: r["target"]["class_preserving_fraction"]
    )


def _log_ticks(axis):
    """Ticks at 1, 2 and 5 of each decade when the axis spans under 1.5 decades."""
    low, high = axis.get_view_interval()
    subs = (1, 2, 5) if np.log10(high / low) < 1.5 else (1,)
    axis.set_major_locator(ticker.LogLocator(subs=subs))
    axis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    axis.set_minor_formatter(ticker.NullFormatter())


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)


def _panel():
    """One of three side-by-side panels, all with the same axes box."""
    fig = plt.figure(figsize=(1.8, 2.5))
    return fig, fig.add_axes((0.24, 0.4, 0.72, 0.57))


def _finish_panel(fig, ax, xlabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MQs")
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.29),
        frameon=False,
        handlelength=1.2,
    )
    _style(ax)


def plot_queries_vs_eta(results: Sequence[dict]):
    fits = eta_fit(results)
    rng = np.random.default_rng(0)
    fig, ax = _panel()
    for color, (states, fit) in zip(PAPER_COLORS, fits.items()):
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
            label=f"$|Q| = {states}$: "
            f"$10^{{{fit[0]:.2f}}} (1 - 2\\eta)^{{-{fit[1]:.1f}}}$",
        )
    ax.set_yscale("log")
    ax.set_xticks(sorted({r["eta"] for r in results}))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    _finish_panel(fig, ax, r"$\eta$")
    return fig


def _plot_power_fits_by_eta(results: Sequence[dict], fits, x, jitter, symbol):
    """Log-log scatter of queries against ``x(run)`` per eta, with ``fits``; each
    x is scaled by up to 10 ** jitter either way."""
    rng = np.random.default_rng(0)
    fig, ax = _panel()
    for color, (eta, (scale, power)) in zip(PAPER_COLORS, fits.items()):
        runs = [r for r in _scored(results) if r["eta"] == eta]
        xs = np.array([x(r) for r in runs], dtype=float)
        ax.scatter(
            xs * 10 ** rng.uniform(-jitter, jitter, len(xs)),
            [r["queries_total"] for r in runs],
            s=4,
            color=color,
            alpha=0.35,
            linewidths=0,
        )
        line = np.geomspace(xs.min(), xs.max(), 2)
        ax.plot(
            line,
            10**scale * line**power,
            color=color,
            linewidth=1.5,
            label=f"$\\eta = {eta:g}$: $10^{{{scale:.2f}}} {symbol}^{{{power:.2f}}}$",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    _finish_panel(fig, ax, f"${symbol}$")
    return fig, ax


def plot_queries_vs_states(results: Sequence[dict]):
    fig, ax = _plot_power_fits_by_eta(
        results, states_fit(results), lambda r: r["target_states"], 0.03, "|Q|"
    )
    ax.set_xticks(sorted({r["target_states"] for r in results}))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    return fig


def plot_queries_vs_pap(results: Sequence[dict]):
    fig, ax = _plot_power_fits_by_eta(
        results,
        pap_fit(results),
        lambda r: r["target"]["class_preserving_fraction"],
        0,
        r"p_{\mathrm{AP}}",
    )
    _log_ticks(ax.xaxis)
    return fig


def plot_difficulty(results: Sequence[dict]):
    fits = eta_fit(results)
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.8), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (metric, (label, scale)) in zip(axes.flat, DIFFICULTY_METRICS.items()):
        for color, states in zip(PAPER_COLORS, fits):
            runs = [r for r in _scored(results) if r["target_states"] == states]
            values = np.array([r["target"][metric] for r in runs])
            excess = [_excess(fits, r) for r in runs]
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
            _log_ticks(ax.xaxis)
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
    with plt.rc_context(PAPER_RC):
        for name, plot in [
            ("queries_vs_eta", plot_queries_vs_eta),
            ("queries_vs_states", plot_queries_vs_states),
            ("queries_vs_pap", plot_queries_vs_pap),
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
