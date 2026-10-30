"""
Render E-L* on random minimal DFAs for a paper, from the results `run` caches:

1. summary.tex -- per target size and noise level, how often E-L* learns the
   target and learns it exactly, the lowest accuracy, and what it costs in
   membership queries and rounds, as a geometric mean and maximum.
2. queries_vs_eta.pdf -- every run's queries against eta, with a power law in
   the signal level 1/2 - eta per size fit to log10 queries.
3. queries_vs_states.pdf -- every run's queries against the target's size,
   with a power law in the size per eta.
4. queries_vs_pap.pdf -- every run's queries against the target's
   class-preserving fraction p_AP, with a power law in p_AP per eta.

    python -m orthogonal_dfa.experiments.random_minimal.analyze
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker

from ..capal_comparison.core import REPO_ROOT
from ..capal_comparison.figures import SUCCESS_THRESHOLD, _pct, tabular
from .run import DATA_DIR, RESULT_SCHEMA_VERSION

OUT_DIR = REPO_ROOT / "figures" / "random-minimal"

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


def _millions(count: float) -> str:
    return f"{count / 1e6:.2f}M"


def _geometric_mean(values: Sequence[float]) -> float:
    return float(np.exp(np.mean(np.log(values))))


def summary_cells(runs: Sequence[dict]) -> List[str]:
    """A run with an error or a timeout has no accuracy and counts as failed."""
    scored = [r for r in runs if r["accuracy"] is not None]
    queries = [r["queries_total"] for r in scored]
    rounds = [len(r["rounds"]) for r in scored]
    return [
        _pct(np.mean([succeeded(r) for r in runs])),
        _pct(np.mean([r["accuracy"] == 1.0 for r in runs])),
        f"{min(r['accuracy'] for r in scored):.3f}" if scored else "--",
        _millions(_geometric_mean(queries)) if queries else "--",
        _millions(max(queries)) if queries else "--",
        f"{_geometric_mean(rounds):.2f}" if rounds else "--",
        str(max(rounds)) if rounds else "--",
    ]


def summary_rows(results: Sequence[dict]) -> List[Union[List[str], str]]:
    return _by_size(results, summary_cells)


def _scored(results: Sequence[dict]) -> List[dict]:
    return [r for r in results if r["accuracy"] is not None]


@dataclass(frozen=True)
class Variable:
    symbol: str
    value: Callable[[dict], float]
    # fits are power laws in base(value)
    base: Callable[[np.ndarray], np.ndarray]
    base_symbol: str
    log: bool
    # a discrete variable is jittered by this much, in log10 units on a log axis,
    # and ticked at its values
    jitter: float


ETA = Variable(
    r"\eta",
    lambda r: r["eta"],
    lambda x: 1 / 2 - x,
    r"(1/2 - \eta)",
    False,
    0.006,
)
STATES = Variable("|Q|", lambda r: r["target_states"], lambda x: x, "|Q|", True, 0.03)
PAP = Variable(
    r"p_{\mathrm{AP}}",
    lambda r: r["target"]["class_preserving_fraction"],
    lambda x: x,
    r"p_{\mathrm{AP}}",
    True,
    0,
)


def _strata(results: Sequence[dict], z: Variable) -> Dict[float, List[dict]]:
    """The runs with an accuracy, by value of z in increasing order."""
    strata = defaultdict(list)
    for r in _scored(results):
        strata[z.value(r)].append(r)
    return dict(sorted(strata.items()))


def power_fit(
    results: Sequence[dict], x: Variable, z: Variable
) -> Dict[float, Tuple[float, float]]:
    """Per value of z, the least-squares (scale, power) of

        log10 queries = scale + power * log10 x.base(x)

    over the runs with an accuracy."""
    fits = {}
    for stratum, runs in _strata(results, z).items():
        power, scale = np.polyfit(
            np.log10(x.base(np.array([x.value(r) for r in runs], dtype=float))),
            np.log10([r["queries_total"] for r in runs]),
            1,
        )
        fits[stratum] = float(scale), float(power)
    return fits


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


def _power_of_two(log10_scale: float) -> str:
    return f"2^{{{log10_scale * np.log2(10):.2g}}}"


def plot_queries(results: Sequence[dict], x: Variable, z: Variable):
    """One of three side-by-side panels, all with the same axes box."""
    fits = power_fit(results, x, z)
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(1.8, 2.5))
    ax = fig.add_axes((0.24, 0.4, 0.72, 0.57))
    for color, (stratum, runs) in zip(PAPER_COLORS, _strata(results, z).items()):
        xs = np.array([x.value(r) for r in runs], dtype=float)
        noise = rng.uniform(-x.jitter, x.jitter, len(xs))
        ax.scatter(
            xs * 10**noise if x.log else xs + noise,
            [r["queries_total"] for r in runs],
            s=4,
            color=color,
            alpha=0.35,
            linewidths=0,
        )
        line = np.linspace(xs.min(), xs.max(), 100)
        scale, power = fits[stratum]
        ax.plot(
            line,
            10 ** (scale + power * np.log10(x.base(line))),
            color=color,
            linewidth=1.5,
            label=f"${z.symbol} = {stratum:g}$: "
            f"${_power_of_two(scale)} {x.base_symbol}^{{{power:.2g}}}$",
        )
    if x.log:
        ax.set_xscale("log")
    ax.set_yscale("log")
    if x.jitter:
        ax.set_xticks(sorted({x.value(r) for r in results}))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
    else:
        _log_ticks(ax.xaxis)
    ax.set_xlabel(f"${x.symbol}$")
    ax.set_ylabel("MQs")
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.29),
        frameon=False,
        handlelength=1.2,
    )
    _style(ax)
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
                "MQ geo.\\ mean",
                "MQ max",
                "Rounds geo.\\ mean",
                "Rounds max",
            ],
            "llrrrrrrr",
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
            ("queries_vs_eta", lambda results: plot_queries(results, ETA, STATES)),
            ("queries_vs_states", lambda results: plot_queries(results, STATES, ETA)),
            ("queries_vs_pap", lambda results: plot_queries(results, PAP, ETA)),
        ]:
            fig = plot(results)
            pdf_path = OUT_DIR / f"{name}.pdf"
            # no creation date, so rerunning on the same results leaves the file alone
            fig.savefig(pdf_path, metadata={"CreationDate": None})
            plt.close(fig)
            print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
