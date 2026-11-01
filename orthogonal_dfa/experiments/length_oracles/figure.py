"""The length-oracle figure, and one printed row per restart, from the checked-in
runs and labels alone.

    python -m orthogonal_dfa.experiments.length_oracles.figure

A DFA's state accepts iff its fit-half strings accept above the fit half's rate; phi and
the rules' agreements are on the read half.  Writes figures/length-oracles/figure.pdf, and
the paper's panels: lengths.pdf, frames.tex and dfa_{saw_1,saw_2,flat}.tex."""

import json
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from automata.fa.dfa import DFA
from matplotlib import image as mpimg
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
WORD = {".": "open", "C": "closed"}
#: Which of frames 0, 1, 2 (base position mod 3) a stop starts in, "C" closed.
PATTERNS = ["...", "..C", ".C.", "C..", ".CC", "C.C", "CC.", "CCC"]
KINDS = {
    "saw": "Sawtooth oracle (one threshold, at length 40)",
    "flat": "Per-length oracle (each length at its own median)",
}


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
        parts.append("stop")
        symbols -= {0, 1, 2}
    if {3, 4} <= symbols:
        parts.append("wild")
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


def layout(dfa):
    return "circo" if is_ring(dfa) else "dot"


def draw(dfa, path):
    lines = [
        "digraph{rankdir=LR; bgcolor=transparent;",
        'node[shape=circle,fontname=Helvetica,width=.45,fixedsize=true,label=""];',
        # the TikZ's 8pt labels, at 1.33cm per graphviz inch
        "edge[fontname=Helvetica,fontsize=15,arrowsize=.6];",
        f"start[shape=point,width=.06]; start->q{dfa.initial_state};",
    ]
    for q in sorted(dfa.states):
        style = (
            'shape=doublecircle,width=.45,style=filled,fillcolor="#cfe8cf"'
            if q in dfa.final_states
            else 'style=filled,fillcolor="#eeeeee"'
        )
        lines.append(f"q{q}[{style}];")
        row = dfa.transitions[q]
        for t in sorted(set(row.values())):
            label = edge_label([c for c in row if row[c] == t], tex=False)
            lines.append(f'q{q}->q{t}[label="{label}"];')
    lines[0] += " margin=0;"
    path.with_suffix(".dot").write_text("\n".join(lines + ["}"]))
    subprocess.run(
        [layout(dfa), "-Tpng", "-Gdpi=200", path.with_suffix(".dot"), "-o", path],
        check=True,
    )


def dfa_tikz(dfa, dot_path):
    """In the paper's lstarex style, along graphviz's node positions, edge curves and
    label positions.  On a ring, self-loops point away from its center instead, where
    graphviz would draw them across the ring's edges."""
    plain = subprocess.run(
        [layout(dfa), "-Tplain", dot_path], capture_output=True, text=True, check=True
    ).stdout
    rows = [shlex.split(line) for line in plain.splitlines()]
    xy = {f[1]: (float(f[2]), float(f[3])) for f in rows if f[0] == "node"}
    lines = [r"\begin{tikzpicture}[lstarex, x=1.33cm, y=1.33cm]"]
    for q in sorted(dfa.states):
        accepting = ", accepting" if q in dfa.final_states else ""
        lines.append(rf"\node[state{accepting}] (q{q}) at {xy[f'q{q}']} {{}};")
    ring = is_ring(dfa)
    if ring:
        center = np.mean([xy[f"q{q}"] for q in dfa.states], axis=0)
        outward = {
            q: np.degrees(np.arctan2(*(np.array(xy[f"q{q}"]) - center)[::-1]))
            for q in dfa.states
        }
        # the start state's loop and arrow share its outward side
        start = outward[dfa.initial_state] - 40
        outward[dfa.initial_state] += 40
        lines.append(
            rf"\draw[<-] (q{dfa.initial_state}.{start:.0f})"
            rf" -- ++({start:.0f}:0.5cm);"
        )
    else:
        lines.append(rf"\draw[->] {xy['start']} -- (q{dfa.initial_state});")
    for f in rows:
        if f[0] != "edge" or f[1] == "start":
            continue
        if ring and f[1] == f[2]:
            q = int(f[1][1:])
            row = dfa.transitions[q]
            label = edge_label([c for c in row if row[c] == q], tex=True)
            d = outward[q]
            lines.append(
                rf"\path[->] (q{q}) edge[out={d + 20:.0f}, in={d - 20:.0f}, looseness=8, min distance=5mm]"
                rf" node[font=\footnotesize, anchor={(d + 180) % 360:.0f}] {{{label}}} (q{q});"
            )
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
            rf"\node[font=\footnotesize] at ({f[5 + 2 * n]}, {f[6 + 2 * n]}) {{{label}}};",
        ]
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
    fig = plt.figure(figsize=(2.2, 2.5))
    ax = fig.add_axes((0.22, 0.45, 0.74, 0.52))
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
        bbox_to_anchor=(0.5, 0.17),
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


def representatives(described):
    """The two sawtooth restarts, from different seeds, whose end state best tracks the
    wildcard count modulo some k (kept ones first among ties), and the kept per-length
    restart that best agrees with "frame0 open"."""

    def counting(key):
        d = described[key]
        return d["by_modulus"][d["modulus"]]

    saw = sorted(
        (key for key in described if key[0] == "saw"),
        key=lambda key: (round(counting(key), 6), described[key]["kept"]),
        reverse=True,
    )
    picked = [saw[0], next(key for key in saw if key[1] != saw[0][1])]
    out = [
        (
            key,
            f"wildcards mod {described[key]['modulus']}: NMI {counting(key):.2f}",
        )
        for key in picked
    ]
    key = max(
        (key for key in described if key[0] == "flat" and described[key]["kept"]),
        key=lambda key: described[key]["agreement"]["frame0 open"],
    )
    caption = f"frame0 open: agrees {described[key]['agreement']['frame0 open']:.2f}"
    return out + [(key, caption)]


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


def main():
    strings, labels = saved_labels()
    data = Strings(np.array(strings, dtype=np.int64))
    y = {kind: labels[kind].astype(float) for kind in KINDS}
    with open(RUNS) as f:
        restarts = {
            (r["oracle"], r["seed"], r["restart"]): r for r in json.load(f)["restarts"]
        }

    read = data.read
    for kind in KINDS:
        cells = [
            f"{name} {phi(rule[read].astype(float), y[kind][read]):+.3f}"
            for name, rule in data.rules.items()
        ]
        print(kind, "rule phi:", "  ".join(cells))
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 8.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.3, wspace=0.18)
    colors = plt.cm.tab10(np.arange(len(PATTERNS)))  # pylint: disable=no-member
    for col, (kind, title) in enumerate(KINDS.items()):
        ax = fig.add_subplot(grid[0, col])
        for j, pattern in enumerate(PATTERNS):
            xs, ys = [], []
            for stops in range(6):
                cell = (data.pattern == j) & (data.stops == stops)
                if cell.sum() >= 400:
                    xs.append(36 + 2 * stops)
                    ys.append(y[kind][cell].mean())
            ax.plot(xs, ys, "o-", color=colors[j], label=pattern, ms=4)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("compiled length (bases) = 36 + 2 x stops")
        ax.set_xticks([36 + 2 * k for k in range(6)])
        ax.set_ylim(0.15, 0.95)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.set_ylabel("oracle accept rate")
        else:
            ax.legend(
                title="frame pattern",
                fontsize=8,
                title_fontsize=8,
                ncol=2,
                loc="lower left",
            )

    saw_rows = grid[1, 0].subgridspec(2, 1, hspace=0.35)
    with tempfile.TemporaryDirectory() as drawings:
        for k, (name, (key, caption)) in enumerate(
            zip(["saw_1", "saw_2", "flat"], representatives(described))
        ):
            d = described[key]
            path = Path(drawings) / f"{key[0]}_s{key[1]}_r{key[2]}.png"
            draw(d["minimal"], path)
            (OUT_DIR / f"dfa_{name}.tex").write_text(
                dfa_tikz(d["minimal"], path.with_suffix(".dot"))
            )
            print(
                f"dfa_{name}: {key[0]} s{key[1]} r{key[2]}: {caption}, "
                f"{len(d['minimal'].states)} states, phi {d['phi']:.3f}"
            )
            ax = fig.add_subplot(saw_rows[k] if key[0] == "saw" else grid[1, 1])
            ax.imshow(mpimg.imread(path))
            ax.axis("off")
            ax.set_title(
                f"{KINDS[key[0]].split(' (')[0]}, seed {key[1]} restart {key[2]}"
                f"{' (kept)' if d['kept'] else ''}: {caption}\n"
                f"({len(d['minimal'].states)} states, phi {d['phi']:.3f})",
                fontsize=9,
            )
    with plt.rc_context({"pdf.fonttype": PAPER_RC["pdf.fonttype"]}):
        fig.savefig(OUT_DIR / "figure.pdf", bbox_inches="tight")
    plt.close(fig)
    write_rates(data, y["flat"])


if __name__ == "__main__":
    main()
