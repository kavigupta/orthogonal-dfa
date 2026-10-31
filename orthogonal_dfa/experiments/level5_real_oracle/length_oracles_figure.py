# pylint: disable=invalid-name,missing-function-docstring,too-many-locals,consider-using-with,modified-iterating-list
"""The length-oracle figure and per-restart table, from length_oracles_extract.py's JSON
alone (numpy, matplotlib and graphviz's dot; nothing from this repo).

  python orthogonal_dfa/experiments/level5_real_oracle/length_oracles_figure.py DATA.json OUT

A DFA's state accepts iff its fit-half strings accept above the fit half's rate; phi and
agreements are on the read half.  Writes OUT/figure.{png,pdf} and prints one row per restart.
"""

import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # pylint: disable=wrong-import-position
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position

NAMES = ["TAG", "TAA", "TGA", "N", "N"]
PATTERNS = ["...", "..C", ".C.", "C..", ".CC", "C.C", "CC.", "CCC"]
RULES = {
    "frame0 open": lambda stops, p: p[0] == ".",
    "frames01 not both": lambda stops, p: p[:2] != "CC",
    "odd stops": lambda stops, p: stops % 2 == 1,
    "one stop": lambda stops, p: stops == 1,
}
REPRESENTATIVES = [
    (
        ("saw", 1, 3),
        "sawtooth, seed 1 restart 3: wildcards mod 2\n(accepts an odd number of stops)",
    ),
    (
        ("saw", 2, 2),
        "sawtooth, seed 2 restart 2: wildcards mod 3\n(accepts 1 or 4 stops)",
    ),
    (
        ("flat", 1, 0),
        "per-length, seed 1 (kept): tracks phase\n(rejects a stop in frame 0)",
    ),
]


def phi(n, v, y, vy):
    """From the totals n, sum v, sum y and sum vy of two 0/1 variables."""
    den = np.sqrt(v * (n - v) * y * (n - y))
    return float((n * vy - v * y) / den) if den else 0.0


def nmi(joint):
    """Normalised by the smaller entropy, from a 2-D table of counts."""
    p = joint / joint.sum()

    def H(q):
        q = q[q > 0]
        return -(q * np.log(q)).sum()

    a, b = H(p.sum(1)), H(p.sum(0))
    return float((a + b - H(p.ravel())) / max(1e-12, min(a, b)))


def minimise(T, init, out):
    """Moore minimisation of the machine reachable from init."""
    seen, order = {init}, [init]
    for q in order:
        for s in T[q]:
            if s not in seen:
                seen.add(s)
                order.append(s)
    pos = {q: i for i, q in enumerate(order)}
    T = np.array([[pos[s] for s in T[q]] for q in order])
    out = np.asarray(out)[order]
    block = np.unique(out, return_inverse=True)[1].ravel()
    while True:
        new = np.unique(
            np.column_stack([block, block[T]]), axis=0, return_inverse=True
        )[1].ravel()
        if new.max() == block.max():
            break
        block = new
    m = block.max() + 1
    Tm, outm = np.zeros((m, T.shape[1]), int), np.zeros(m, int)
    Tm[block] = block[T]
    outm[block] = out
    return Tm, int(block[0]), outm


def total(rows, key="n", keep=lambda x: True):
    return sum(x[key] for x in rows if keep(x))


def describe(r):
    """Fitted labels, minimal machine and statistics of one restart."""
    T = np.array(r["dfa"]["transitions"])
    rows = r["end_counts"]
    fit = [x for x in rows if x["half"] == "fit"]
    read = [x for x in rows if x["half"] == "read"]
    n_s, a_s = np.zeros(len(T)), np.zeros(len(T))
    for x in fit:
        n_s[x["state"]] += x["n"]
        a_s[x["state"]] += x["accepts"]
    label = (n_s > 0) & (a_s > a_s.sum() / n_s.sum() * n_s)
    Tm, im, om = minimise(T, r["dfa"]["start"], label.astype(int))

    def v(x):
        return label[x["state"]]

    n = total(read)
    d = dict(
        Tm=Tm,
        im=im,
        om=om,
        phi=phi(
            n,
            total(read, keep=v),
            total(read, "accepts"),
            total(read, "accepts", v),
        ),
        agreement={
            name: total(
                read, keep=lambda x, rule=rule: v(x) == rule(x["stops"], x["pattern"])
            )
            / n
            for name, rule in RULES.items()
        },
        stop_self_loops=float(
            np.mean([T[q, c] == q for q in range(len(T)) for c in range(3)])
        ),
    )
    counts = {}
    for k in range(2, 8):
        joint = np.zeros((len(T), k))
        for x in rows:
            joint[x["state"], (36 - x["stops"]) % k] += x["n"]
        counts[k] = nmi(joint)
    d["count_modulus"] = max(counts, key=counts.get)
    d["nmi_count"] = counts[d["count_modulus"]]
    joint = np.zeros((len(T), len(PATTERNS)))
    for x in rows:
        joint[x["state"], PATTERNS.index(x["pattern"])] += x["n"]
    d["nmi_frame"] = nmi(joint)
    return d


def edge_label(cs):
    cs, parts = set(cs), []
    if {0, 1, 2} <= cs:
        parts.append("stop")
        cs -= {0, 1, 2}
    if {3, 4} <= cs:
        parts.append("N")
        cs -= {3, 4}
    return ", ".join(parts + [NAMES[c] for c in sorted(cs)])


def draw(Tm, im, om, path):
    lines = [
        "digraph{rankdir=LR; bgcolor=transparent;",
        'node[shape=circle,fontname=Helvetica,width=.35,fixedsize=true,label=""];',
        "edge[fontname=Helvetica,fontsize=10,arrowsize=.6];",
        f"start[shape=point,width=.06]; start->q{im};",
    ]
    for u, row in enumerate(Tm):
        style = (
            'shape=doublecircle,width=.3,style=filled,fillcolor="#cfe8cf"'
            if om[u]
            else 'style=filled,fillcolor="#eeeeee"'
        )
        lines.append(f"q{u}[{style}];")
        for t in sorted(set(row)):
            lab = edge_label([c for c in range(5) if row[c] == t])
            lines.append(f'q{u}->q{t}[label="{lab}"];')
    open(path + ".dot", "w").write("\n".join(lines + ["}"]))
    subprocess.run(
        ["dot", "-Tpng", "-Gdpi=200", path + ".dot", "-o", path + ".png"], check=True
    )


def print_tables(data):
    read = [x for x in data["oracle_counts"] if x["half"] == "read"]
    for kind in ("saw", "flat"):
        y = f"{kind}_accepts"
        cells = []
        for name, rule in RULES.items():

            def keep(x, rule=rule):
                return rule(x["stops"], x["pattern"])

            value = phi(
                total(read),
                total(read, keep=keep),
                total(read, y),
                total(read, y, keep),
            )
            cells.append(f"{name} {value:+.3f}")
        print(kind, "rule phi:", "  ".join(cells))
    described = {}
    for r in data["restarts"]:
        d = described[(r["oracle"], r["seed"], r["restart"])] = describe(r)
        print(
            f"{r['oracle']:4s} s{r['seed']} r{r['restart']}{'*' if r['kept'] else ' '} "
            f"pick {r['pick_score']:.3f} phi {d['phi']:+.3f} "
            f"states {len(r['dfa']['transitions']):2d} (minimal {len(d['Tm']):2d})  "
            f"NMI(wildcards mod {d['count_modulus']}) {d['nmi_count']:.2f} "
            f"NMI(frame) {d['nmi_frame']:.2f}  stop self-loops {d['stop_self_loops']:.2f}  "
            + "  ".join(f"{k} {a:.2f}" for k, a in d["agreement"].items())
        )
    return described


def main():
    data_path, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    data = json.load(open(data_path))
    described = print_tables(data)

    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.3, wspace=0.18)
    colors = plt.cm.tab10(np.arange(8))  # pylint: disable=no-member
    oc = data["oracle_counts"]
    for col, (kind, title) in enumerate(
        (
            ("saw", "Sawtooth oracle (one threshold, at length 40)"),
            ("flat", "Per-length oracle (each length at its own median)"),
        )
    ):
        ax = fig.add_subplot(gs[0, col])
        for j, p in enumerate(PATTERNS):
            xs, ys = [], []
            for stops in range(6):
                cell = [x for x in oc if x["pattern"] == p and x["stops"] == stops]
                if total(cell) >= 400:
                    xs.append(36 + 2 * stops)
                    ys.append(total(cell, f"{kind}_accepts") / total(cell))
            ax.plot(xs, ys, "o-", color=colors[j], label=p, ms=4)
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

    sub = gs[1, 0].subgridspec(2, 1, hspace=0.35)
    for k, (key, caption) in enumerate(REPRESENTATIVES):
        d = described[key]
        path = f"{out}/dfa_{key[0]}_s{key[1]}_r{key[2]}"
        draw(d["Tm"], d["im"], d["om"], path)
        ax = fig.add_subplot(sub[k]) if key[0] == "saw" else fig.add_subplot(gs[1, 1])
        ax.imshow(mpimg.imread(path + ".png"))
        ax.axis("off")
        ax.set_title(
            f"{caption}  ({len(d['Tm'])} states, phi {d['phi']:.3f})", fontsize=9
        )
    fig.savefig(f"{out}/figure.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{out}/figure.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
