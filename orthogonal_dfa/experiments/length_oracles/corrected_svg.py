"""The length-oracle figure's (c) alone, as figures/length-oracles/corrected.svg.  Each
state is labeled with its phase and closed frames, and colored by the Correctly
Calibrated oracle's accept rate on the saved strings that end in it.  Each DFA's phi,
the residual's included, is on the flat labels, as figure scores them.  Needs pdflatex
and pdftocairo.

    python -m orthogonal_dfa.experiments.length_oracles.corrected_svg"""

import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from matplotlib import colormaps

from orthogonal_dfa.l_star.dfa_utils import table_end_states

from .dfa_structure import frame_states
from .figure import (
    FRAME_RULES,
    OUT_DIR,
    PAPER_NAMES,
    PRODUCT_SEED,
    RESIDUAL_PRODUCTS,
    Strings,
    flat_fit,
    phi,
    product_parts,
)
from .label import saved_labels
from .render_dfa import dfa_tikz

CMAP = colormaps["viridis"]
#: Viewers that rasterize an SVG at its declared size blur it when zoomed.
SIZE_SCALE = 16

# lstarex is the paper's style for the DFA drawings, copied from its commands.tex
TEX = r"""
\documentclass[border=4pt]{standalone}
\usepackage{libertine}
\usepackage[libertine]{newtxmath}
\usepackage{pgfplots}
\pgfplotsset{compat=1.16}
\usetikzlibrary{automata, arrows.meta}
\tikzset{
    lstarex/.style={
        node distance=2cm, >={Stealth[length=2mm]}, auto, font=\small,
        every state/.style={minimum size=6mm, inner sep=0pt},
        initial text={},
    },
}
\newcommand{\phifirst}{@PHI_first@}
\newcommand{\phiresidual}{@PHI_residual@}
\newcommand{\phiproduct}{@PHI_product@}
\newcommand{\dfa}[1]{\vcenter{\hbox{\begin{tabular}{c}
    \input{dfa_#1} \\ \footnotesize $\phi = \csname phi#1\endcsname$
\end{tabular}}}}
\begin{document}
$\vcenter{\hbox{\begin{tabular}{c}
$\dfa{first} \;\otimes\; \dfa{residual}$ \\[1em]
$=\; \dfa{product}$ \\[1em]
\footnotesize no-ORF rule: $\phi = @PHI_NOORF@$
\end{tabular}}}
\quad \vcenter{\hbox{\begin{tabular}{l}
\begin{tikzpicture}
\begin{axis}[
    hide axis, scale only axis, width=0pt, height=0pt,
    colormap={rate}{@COLORMAP@},
    point meta min=@LO@, point meta max=@HI@,
    colorbar, colorbar style={
        height=4cm, width=2.5mm, ylabel={oracle accept rate},
        ytick distance=0.1, yticklabel style={font=\footnotesize},
    },
]
\addplot[draw=none] coordinates {(0, 0)};
\end{axis}
\end{tikzpicture} \\[1em]
\footnotesize S = stop \\
\footnotesize W = any base \\[0.5em]
\footnotesize $x$/$yz$ = in phase $x$, \\
\footnotesize \quad frames $y, z$ closed \\
\footnotesize $*$ = phase varies
\end{tabular}}}$
\end{document}
"""


def tikz_color(rgb):
    r, g, b = rgb[:3]
    return f"rgb,1:red,{r:.4f};green,{g:.4f};blue,{b:.4f}"


def state_label(reached):
    """phase/closed: the phase, or * where the strings reaching the state differ in it,
    then the frames closed on every one of them."""
    phases = {phase for phase, _ in reached}
    phase = str(next(iter(phases))) if len(phases) == 1 else "$*$"
    closed = frozenset.intersection(*(closed for _, closed in reached))
    return phase + "/" + "".join(map(str, sorted(closed)))


def decoration(rgba, reached):
    r, g, b = rgba[:3]
    text = "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else "black"
    return (
        rf"fill={{{tikz_color(rgba)}}}, text={text}, font=\scriptsize",
        state_label(reached),
    )


def state_rates(dfa, data, y):
    transitions = np.array(
        [
            [dfa.transitions[q][c] for c in sorted(dfa.input_symbols)]
            for q in range(len(dfa.states))
        ]
    )
    end = table_end_states(transitions, dfa.initial_state, data.s)
    counts = np.bincount(end, minlength=len(transitions))
    accepts = np.bincount(end, weights=y, minlength=len(transitions))
    return {q: accepts[q] / counts[q] for q in range(len(transitions)) if counts[q]}


def main():
    strings, labels = saved_labels()
    data = Strings(np.array(strings, dtype=np.int64))
    y = {kind: labels[kind].astype(float) for kind in PAPER_NAMES}
    with open(RESIDUAL_PRODUCTS) as f:
        runs = {r["seed"]: r for r in json.load(f)}
    parts = product_parts(y, data, runs[PRODUCT_SEED])
    rates = {
        name: state_rates(d["minimal"], data, y["flat"]) for name, d in parts.items()
    }
    every = [r for by_state in rates.values() for r in by_state.values()]
    lo, hi = np.floor(min(every) * 10) / 10, np.ceil(max(every) * 10) / 10
    colormap = " ".join(
        "rgb=(" + ",".join(f"{v:.4f}" for v in CMAP(x)[:3]) + ")"
        for x in np.linspace(0, 1, 32)
    )
    rule = FRAME_RULES["NotAllFramesClosed"](data.closed)[data.read].astype(float)
    fills = {
        "COLORMAP": colormap,
        "LO": str(lo),
        "HI": str(hi),
        "PHI_first": f"{parts['first']['phi']:.3f}",
        "PHI_residual": f"{flat_fit(runs[PRODUCT_SEED]['residual'], y, data)[1]:.3f}",
        "PHI_product": f"{parts['product']['phi']:.3f}",
        "PHI_NOORF": f"{phi(rule, y['flat'][data.read]):.3f}",
    }
    tex = TEX
    for key, value in fills.items():
        tex = tex.replace(f"@{key}@", value)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, d in parts.items():
            reached = frame_states(d["minimal"])
            decorations = {
                q: decoration(CMAP((r - lo) / (hi - lo)), reached[q])
                for q, r in rates[name].items()
            }
            (tmp / f"dfa_{name}.tex").write_text(dfa_tikz(d["minimal"], decorations))
        (tmp / "corrected.tex").write_text(tex)
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "corrected.tex"],
            cwd=tmp,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            ["pdftocairo", "-svg", "corrected.pdf", "corrected.svg"],
            cwd=tmp,
            check=True,
        )
        svg = (tmp / "corrected.svg").read_text()
    # only the root's size, which comes first; the viewBox keeps the drawing's units
    svg = re.sub(
        r'(width|height)="([\d.]+)"',
        lambda m: f'{m[1]}="{float(m[2]) * SIZE_SCALE:g}"',
        svg,
        count=2,
    )
    (OUT_DIR / "corrected.svg").write_text(svg)


if __name__ == "__main__":
    main()
