"""A DFA over the length-oracle vocabulary as TikZ in the paper's lstarex style, laid out
by graphviz."""

import shlex
import subprocess

import numpy as np

from .dfa_structure import is_ring

#: TAG, TAA, TGA and the two wildcards, in the paper's symbols.
NAMES = ["S_1", "S_0", "S_2", "X", "Y"]
#: TikZ cm per graphviz inch; graphviz's nodes and labels are sized to match the TikZ's
#: 6mm states (lstarex) and 8pt labels.
CM_PER_INCH = 1.05
NODE_INCHES = 0.6 / CM_PER_INCH
LABEL_POINTS = 8 * 2.54 / CM_PER_INCH
#: A ring is drawn as two rows, this far apart and with its states this far apart.
RING_ROWS_CM, RING_STEP_CM = 1.1, 1.25


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
