"""The length-oracle figure, and one printed row per restart, from the runs and the
labelled strings.

    python -m orthogonal_dfa.experiments.length_oracles.figure

A DFA's state accepts iff its fit-half strings accept above the fit half's rate; phi and
the rules' agreements are on the read half.  Writes figures/length_oracles/."""

import json
import subprocess

import numpy as np
from automata.fa.dfa import DFA
from matplotlib import image as mpimg
from matplotlib import pyplot as plt

from orthogonal_dfa.l_star.dfa_utils import table_end_states

from ..capal_comparison.core import REPO_ROOT
from ..random_minimal.analyze import PAPER_RC
from .label import COUNT, labels, strings
from .oracles import SYMBOLS

RUNS = REPO_ROOT / "data" / "neural" / "length_oracles" / "runs.json"
OUT_DIR = REPO_ROOT / "figures" / "length_oracles"
NAMES = ["TAG", "TAA", "TGA", "N", "N"]
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
    modulus = max(by_modulus, key=by_modulus.get)
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


def edge_label(symbols):
    symbols, parts = set(symbols), []
    if {0, 1, 2} <= symbols:
        parts.append("stop")
        symbols -= {0, 1, 2}
    if {3, 4} <= symbols:
        parts.append("N")
        symbols -= {3, 4}
    return ", ".join(parts + [NAMES[c] for c in sorted(symbols)])


def draw(dfa, path):
    lines = [
        "digraph{rankdir=LR; bgcolor=transparent;",
        'node[shape=circle,fontname=Helvetica,width=.35,fixedsize=true,label=""];',
        "edge[fontname=Helvetica,fontsize=10,arrowsize=.6];",
        f"start[shape=point,width=.06]; start->q{dfa.initial_state};",
    ]
    for q in sorted(dfa.states):
        style = (
            'shape=doublecircle,width=.3,style=filled,fillcolor="#cfe8cf"'
            if q in dfa.final_states
            else 'style=filled,fillcolor="#eeeeee"'
        )
        lines.append(f"q{q}[{style}];")
        row = dfa.transitions[q]
        for t in sorted(set(row.values())):
            label = edge_label([c for c in row if row[c] == t])
            lines.append(f'q{q}->q{t}[label="{label}"];')
    path.with_suffix(".dot").write_text("\n".join(lines + ["}"]))
    subprocess.run(
        ["dot", "-Tpng", "-Gdpi=200", path.with_suffix(".dot"), "-o", path],
        check=True,
    )


def representatives(described):
    """The two sawtooth restarts, from different seeds, whose end state best tracks the
    wildcard count modulo some k, and the kept per-length restart that best agrees
    with "frame0 open"."""

    def counting(key):
        d = described[key]
        return d["by_modulus"][d["modulus"]]

    saw = sorted((key for key in described if key[0] == "saw"), key=counting)[::-1]
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


def main():
    data = Strings(np.array(strings(COUNT), dtype=np.int64))
    y = {kind: labels(kind, COUNT).astype(float) for kind in KINDS}
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
        picks = [s["pick_score"] for k, s in restarts.items() if k[:2] == key[:2]]
        d["kept"] = r["pick_score"] == max(picks)
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
    for k, (key, caption) in enumerate(representatives(described)):
        d = described[key]
        path = OUT_DIR / f"dfa_{key[0]}_s{key[1]}_r{key[2]}.png"
        draw(d["minimal"], path)
        ax = fig.add_subplot(saw_rows[k] if key[0] == "saw" else grid[1, 1])
        ax.imshow(mpimg.imread(path))
        ax.axis("off")
        ax.set_title(
            f"{KINDS[key[0]].split(' (')[0]}, seed {key[1]} restart {key[2]}"
            f"{' (kept)' if d['kept'] else ''}: {caption}\n"
            f"({len(d['minimal'].states)} states, phi {d['phi']:.3f})",
            fontsize=9,
        )
    fig.savefig(OUT_DIR / "figure.png", dpi=150, bbox_inches="tight")
    with plt.rc_context({"pdf.fonttype": PAPER_RC["pdf.fonttype"]}):
        fig.savefig(OUT_DIR / "figure.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
