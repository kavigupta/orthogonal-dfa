"""The length-oracle figure's panels, and one printed row per restart, from the
checked-in runs and labels alone.

    python -m orthogonal_dfa.experiments.length_oracles.figure

A DFA's state accepts iff its fit-half strings accept above the fit half's rate; phi and
the rules' agreements are on the read half.  Writes lengths.pdf, frames.tex and
dfa_{saw_1,saw_2,flat}.tex to figures/length-oracles/."""

import json
import shlex
import subprocess

import numpy as np
from automata.fa.dfa import DFA
from matplotlib import pyplot as plt
from matplotlib import transforms

from orthogonal_dfa.l_star.dfa_utils import table_end_states

from ..capal_comparison.core import REPO_ROOT
from ..random_minimal.analyze import PAPER_COLORS, PAPER_RC
from .label import saved_labels
from .oracles import SYMBOLS

RUNS = REPO_ROOT / "data" / "neural" / "length_oracles" / "runs.json"
RANDOM_STRINGS = (
    REPO_ROOT / "data" / "neural" / "length_oracles" / "random_strings.json"
)
OUT_DIR = REPO_ROOT / "figures" / "length-oracles"
#: TAG, TAA, TGA and the two wildcards, in the paper's symbols.
NAMES = ["S_1", "S_0", "S_2", "X", "Y"]
PAPER_NAMES = {"saw": "Length-Sawtooth", "flat": "Length-Flat"}
#: The DFAs drawn, by (oracle, seed, restart).  Only the restart a run kept may be drawn.
REPRESENTATIVES = {
    "saw_1": ("saw", 0, 2),
    "saw_2": ("saw", 1, 1),
    "flat": ("flat", 0, 3),
}
WORD = {".": "open", "C": "closed"}
#: TikZ cm per graphviz inch; graphviz's nodes and labels are sized to match the TikZ's
#: 6mm states (lstarex) and 8pt labels.
CM_PER_INCH = 1.15
NODE_INCHES = 0.6 / CM_PER_INCH
LABEL_POINTS = 8 * 2.54 / CM_PER_INCH
#: A ring is drawn as two rows, this far apart and with its states this far apart.
RING_ROWS_CM, RING_STEP_CM = 1.1, 1.4
#: Which of frames 0, 1, 2 (base position mod 3) a stop starts in, "C" closed.
PATTERNS = ["...", "..C", ".C.", "C..", ".CC", "C.C", "CC.", "CCC"]


class Strings:
    def __init__(self, s):
        self.s = s
        self.read = np.arange(len(s)) >= len(s) // 2
        self.stops = (s < 3).sum(1)
        closed = np.zeros((len(s), 3), bool)
        position = np.zeros(len(s), int)
        for t in range(s.shape[1]):
            stop = s[:, t] < 3
            closed[stop, position[stop] % 3] = True
            position += np.where(stop, 3, 1)
        names = ["".join("C" if c else "." for c in row) for row in closed]
        self.pattern = np.array([PATTERNS.index(p) for p in names])
        self.rules = {
            "frame0 open": ~closed[:, 0],
            "frames01 not both": ~(closed[:, 0] & closed[:, 1]),
            "odd stops": self.stops % 2 == 1,
            "one stop": self.stops == 1,
        }


def phi(v, y):
    if v.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(v, y)[0, 1])


def nmi(a, b):
    """Normalised by the smaller entropy."""
    joint = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(joint, (a, b), 1)
    p = joint / joint.sum()

    def entropy(q):
        q = q[q > 0]
        return -(q * np.log(q)).sum()

    ha, hb = entropy(p.sum(1)), entropy(p.sum(0))
    return float((ha + hb - entropy(p.ravel())) / max(1e-12, min(ha, hb)))


def describe(restart, y, data):
    transitions = np.array(restart["dfa"]["transitions"])
    start = restart["dfa"]["start"]
    end = table_end_states(transitions, start, data.s)
    fit, read = ~data.read, data.read
    count = np.bincount(end[fit], minlength=len(transitions))
    accepts = np.bincount(end[fit], weights=y[fit], minlength=len(transitions))
    verdict = accepts > y[fit].mean() * count
    v = verdict[end]
    by_modulus = {k: nmi(end, (SYMBOLS - data.stops) % k) for k in range(2, 8)}
    # The divisors and the multiples of a counter's period tie with it: a mod-6
    # counter ties at 2, 3 and 6, a mod-2 one at 2, 4 and 6.
    tied = [k for k, v in by_modulus.items() if v > max(by_modulus.values()) - 1e-6]
    modulus = next(
        (k for k in tied if all(k % j == 0 or j % k == 0 for j in tied)), min(tied)
    )
    return dict(
        minimal=DFA(
            states=set(range(len(transitions))),
            input_symbols=set(range(len(NAMES))),
            transitions={
                q: {c: int(t) for c, t in enumerate(row)}
                for q, row in enumerate(transitions)
            },
            initial_state=start,
            final_states={q for q in range(len(transitions)) if verdict[q]},
        ).minify(),
        phi=phi(v[read], y[read]),
        agreement={
            name: float((v[read] == rule[read]).mean())
            for name, rule in data.rules.items()
        },
        by_modulus=by_modulus,
        modulus=modulus,
        nmi_frame=nmi(end, data.pattern),
        stop_self_loops=float(
            (transitions[:, :3] == np.arange(len(transitions))[:, None]).mean()
        ),
    )


def edge_label(symbols, tex):
    """Graphviz gets the plain text of the TikZ label, so it leaves room for it."""
    symbols, parts = set(symbols), []
    if {0, 1, 2} <= symbols:
        parts.append("S")
        symbols -= {0, 1, 2}
    if {3, 4} <= symbols:
        parts.append("W")
        symbols -= {3, 4}
    names = [f"${n}$" if tex else n.replace("_", "") for n in NAMES]
    return ", ".join(parts + [names[c] for c in sorted(symbols)])


def is_ring(dfa):
    """A single cycle of at least 3 states, every other edge a self-loop: a counter."""
    successors = {
        q: {t for t in dfa.transitions[q].values() if t != q} for q in dfa.states
    }
    if len(dfa.states) < 3 or any(len(t) != 1 for t in successors.values()):
        return False
    q, seen = dfa.initial_state, set()
    while q not in seen:
        seen.add(q)
        (q,) = successors[q]
    return q == dfa.initial_state and seen == set(dfa.states)


def ring_positions(dfa):
    """Along the cycle from the start state: the top row left to right, then the
    bottom row right to left, in graphviz inches."""
    order = [dfa.initial_state]
    while len(order) < len(dfa.states):
        (q,) = {t for t in dfa.transitions[order[-1]].values() if t != order[-1]}
        order.append(q)
    top = (len(order) + 1) // 2
    xs = list(range(top)) + [
        top - 1 - j - (len(order) % 2) / 2 for j in range(len(order) - top)
    ]
    return {
        f"q{q}": (
            x * RING_STEP_CM / CM_PER_INCH,
            RING_ROWS_CM / CM_PER_INCH * (i < top),
        )
        for i, (q, x) in enumerate(zip(order, xs))
    }


def layout(dfa):
    return "circo" if is_ring(dfa) else "dot"


def dot_source(dfa):
    lines = [
        "digraph{rankdir=LR; margin=0;",
        f"node[shape=circle,fontname=Helvetica,width={NODE_INCHES:.2f},"
        'fixedsize=true,label=""];',
        f"edge[fontname=Helvetica,fontsize={LABEL_POINTS:.1f},arrowsize=.6];",
        f"start[shape=point,width=.06]; start->q{dfa.initial_state};",
    ]
    for q in sorted(dfa.states):
        if q in dfa.final_states:
            lines.append(f"q{q}[shape=doublecircle];")
        row = dfa.transitions[q]
        for t in sorted(set(row.values())):
            label = edge_label([c for c in row if row[c] == t], tex=False)
            lines.append(f'q{q}->q{t}[label="{label}"];')
    return "\n".join(lines + ["}"])


def ring_edges(dfa, xy):
    """On the two rows' outside: a row's loops and edge labels above the top row and
    below the bottom one, and the end edges' labels beside them."""
    center = np.mean([xy[f"q{q}"][0] for q in dfa.states])
    top = {q: xy[f"q{q}"][1] > 0 for q in dfa.states}
    lines = [rf"\draw[<-] (q{dfa.initial_state}.west) -- ++(-0.5cm, 0);"]
    for q in sorted(dfa.states):
        row = dfa.transitions[q]
        for t in sorted(set(row.values())):
            label = edge_label([c for c in row if row[c] == t], tex=True)
            if top[q] == top[t]:
                d = 90 if top[q] else 270
            else:
                d = 180 if xy[f"q{q}"][0] < center else 0
            node = rf"node[font=\footnotesize, anchor={(d + 180) % 360}] {{{label}}}"
            if t != q:
                lines.append(rf"\draw[->] (q{q}) -- {node} (q{t});")
            else:
                lines.append(
                    rf"\path[->] (q{q}) edge[out={d + 20}, in={d - 20},"
                    rf" looseness=8, min distance=5mm] {node} (q{q});"
                )
    return lines


def spline_edges(dfa, rows, xy):
    """Along graphviz's edge curves, with its label positions."""
    lines = [rf"\draw[->] {xy['start']} -- (q{dfa.initial_state});"]
    for f in rows:
        if f[0] != "edge" or f[1] == "start":
            continue
        n = int(f[3])
        points = [f"({f[4 + 2 * i]}, {f[5 + 2 * i]})" for i in range(n)]
        # cubic B-spline segments; the ends attach to the nodes' borders
        points[0], points[-1] = f"({f[1]})", f"({f[2]})"
        path = points[0]
        for i in range(1, n, 3):
            path += f" .. controls {points[i]} and {points[i + 1]} .. {points[i + 2]}"
        row = dfa.transitions[int(f[1][1:])]
        label = edge_label([c for c in row if row[c] == int(f[2][1:])], tex=True)
        lines += [
            rf"\draw[->] {path};",
            rf"\node[font=\footnotesize] at ({f[5 + 2 * n]}, {f[6 + 2 * n]})"
            rf" {{{label}}};",
        ]
    return lines


def dfa_tikz(dfa):
    """In the paper's lstarex style, at graphviz's node positions.  A ring is drawn as
    two rows instead, with its edges on its outside."""
    plain = subprocess.run(
        [layout(dfa), "-Tplain"],
        input=dot_source(dfa),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = [shlex.split(line) for line in plain.splitlines()]
    ring = is_ring(dfa)
    xy = (
        ring_positions(dfa)
        if ring
        else {f[1]: (float(f[2]), float(f[3])) for f in rows if f[0] == "node"}
    )
    lines = [rf"\begin{{tikzpicture}}[lstarex, x={CM_PER_INCH}cm, y={CM_PER_INCH}cm]"]
    for q in sorted(dfa.states):
        accepting = ", accepting" if q in dfa.final_states else ""
        lines.append(rf"\node[state{accepting}] (q{q}) at {xy[f'q{q}']} {{}};")
    lines += ring_edges(dfa, xy) if ring else spline_edges(dfa, rows, xy)
    return "\n".join(lines + [r"\end{tikzpicture}", ""])


def plot_lengths(data, random_strings):
    """Each oracle's accept rate on uniform random strings of the compiled lengths up to
    their 99th percentile, drawn at the size it is printed."""
    lengths = 36 + 2 * data.stops
    rows = [
        r
        for r in random_strings
        if lengths.min() <= r["length"] <= np.percentile(lengths, 99)
        and (r["length"] - 36) % 2 == 0
    ]
    xs = [r["length"] for r in rows]
    fig = plt.figure(figsize=(2.2, 2.1))
    ax = fig.add_axes((0.22, 0.5, 0.74, 0.47))
    for color, (kind, name) in zip(PAPER_COLORS, PAPER_NAMES.items()):
        ys = [r[f"{kind}_accepts"] / r["n"] for r in rows]
        ax.plot(xs, ys, "o-", color=color, label=name, ms=3, linewidth=1.5)
    ax.set_xticks(
        xs,
        [f"{x}\n{(x - 36) // 2}\n{SYMBOLS - (x - 36) // 2}" for x in xs],
    )
    # row headers for the three lines of each tick label
    ax.text(
        0,
        0,
        "length\nstops\nwilds",
        ha="right",
        va="top",
        transform=transforms.offset_copy(
            ax.transAxes,
            fig,
            x=-6,
            y=-plt.rcParams["xtick.major.size"] - plt.rcParams["xtick.major.pad"],
            units="points",
        ),
    )
    ax.set_xlabel("uniformly random string")
    ax.set_ylabel("oracle accept rate")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.2),
        frameon=False,
        handlelength=1.2,
        ncol=2,
    )
    return fig


def frame_grids(data, y):
    """TikZ: the Length-Flat oracle's accept rate, a grid of frame 1 by frame 2 for each
    state of frame 0, shaded with the paper's agree above .5 and disagree below."""
    lines = [
        r"\begin{tikzpicture}[x=0.9cm, y=0.55cm, font=\footnotesize,"
        r" every node/.style={text height=1.5ex, text depth=0.4ex, inner sep=1pt}]",
        r"\node[rotate=90] at (-1.45, -1) {Frame 1};",
    ]
    for f1 in range(2):
        lines.append(
            rf"\node[anchor=east] at (-0.1, {-f1 - 0.5}) {{{WORD['.C'[f1]]}}};"
        )
    for f0 in range(2):
        x0 = 2.5 * f0
        lines += [
            rf"\node at ({x0 + 1}, 2.2) {{Frame 0 {WORD['.C'[f0]]}}};",
            rf"\draw[black!50] ({x0 + 0.05}, 1.85) -- ({x0 + 1.95}, 1.85);",
            rf"\node at ({x0 + 1}, 1.35) {{Frame 2}};",
            rf"\draw[black!50] ({x0 + 0.05}, 1) -- ({x0 + 1.95}, 1);",
        ]
        for f2 in range(2):
            lines.append(rf"\node at ({x0 + f2 + 0.5}, 0.5) {{{WORD['.C'[f2]]}}};")
        for f1 in range(2):
            for f2 in range(2):
                v = y[
                    data.pattern == PATTERNS.index(".C"[f0] + ".C"[f1] + ".C"[f2])
                ].mean()
                color = "agree" if v >= 0.5 else "disagree"
                shade = round(min(abs(v - 0.5) * 300, 100))
                cell = f"({x0 + f2}, {-f1 - 1}) rectangle +(1, 1)"
                lines += [
                    rf"\fill[{color}!{shade}!white] {cell};",
                    rf"\draw[white, line width=0.8pt] {cell};",
                    rf"\node at ({x0 + f2 + 0.5}, {-f1 - 0.5}) {{{v:.2f}}};",
                ]
    return "\n".join(lines + [r"\end{tikzpicture}", ""])


def caption(kind, d):
    if kind == "saw":
        return f"wildcards mod {d['modulus']}: NMI {d['by_modulus'][d['modulus']]:.2f}"
    return f"frame0 open: agrees {d['agreement']['frame0 open']:.2f}"


def write_rates(data, flat):
    with open(RANDOM_STRINGS) as f:
        random_strings = json.load(f)["rows"]
    with plt.rc_context(PAPER_RC):
        fig = plot_lengths(data, random_strings)
        # no creation date, so rerunning on the same data leaves the file alone
        fig.savefig(
            OUT_DIR / "lengths.pdf",
            bbox_inches="tight",
            pad_inches=0.02,
            metadata={"CreationDate": None},
        )
        plt.close(fig)
    (OUT_DIR / "frames.tex").write_text(frame_grids(data, flat))


def describe_all(restarts, y, data):
    """Each restart's description, printed as a row."""
    described = {}
    for key, r in restarts.items():
        d = described[key] = describe(r, y[r["oracle"]], data)
        picks = [
            s["pick_score"] for k, s in sorted(restarts.items()) if k[:2] == key[:2]
        ]
        # As counterexample_synthesis keeps it: the first best.
        d["kept"] = key[2] == int(np.argmax(picks))
        print(
            f"{key[0]:4s} s{key[1]} r{key[2]}{'*' if d['kept'] else ' '} "
            f"pick {r['pick_score']:.3f} phi {d['phi']:+.3f} "
            f"states {len(r['dfa']['transitions']):2d} "
            f"(minimal {len(d['minimal'].states):2d})  "
            f"NMI(wildcards mod {d['modulus']}) {d['by_modulus'][d['modulus']]:.2f} "
            f"NMI(frame) {d['nmi_frame']:.2f}  "
            f"stop self-loops {d['stop_self_loops']:.2f}  "
            + "  ".join(f"{k} {a:.2f}" for k, a in d["agreement"].items())
        )
    return described


def main():
    strings, labels = saved_labels()
    data = Strings(np.array(strings, dtype=np.int64))
    y = {kind: labels[kind].astype(float) for kind in PAPER_NAMES}
    with open(RUNS) as f:
        restarts = {
            (r["oracle"], r["seed"], r["restart"]): r for r in json.load(f)["restarts"]
        }

    read = data.read
    for kind in PAPER_NAMES:
        cells = [
            f"{name} {phi(rule[read].astype(float), y[kind][read]):+.3f}"
            for name, rule in data.rules.items()
        ]
        print(kind, "rule phi:", "  ".join(cells))
    described = describe_all(restarts, y, data)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, key in REPRESENTATIVES.items():
        d = described[key]
        if not d["kept"]:
            raise ValueError(f"{key} is not the restart its run kept")
        (OUT_DIR / f"dfa_{name}.tex").write_text(dfa_tikz(d["minimal"]))
        print(
            f"dfa_{name}: {key[0]} s{key[1]} r{key[2]}: {caption(key[0], d)}, "
            f"{len(d['minimal'].states)} states, phi {d['phi']:.3f}"
        )
    write_rates(data, y["flat"])


if __name__ == "__main__":
    main()
