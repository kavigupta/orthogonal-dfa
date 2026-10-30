# pylint: disable=invalid-name,missing-function-docstring,too-many-locals,too-many-statements,consider-using-with,consider-using-enumerate,modified-iterating-list,cell-var-from-loop
"""Every restart DFA of length_oracles.py runs, read on length_oracles_label.py's strings
with its own oracle's labels: a state accepts iff its strings in the first half beat that
half's accept rate; phi is read on the second half.  Prints one row per DFA, writes
OUT/decode.json and OUT/figure.{png,pdf}.

  python orthogonal_dfa/experiments/level5_real_oracle/length_oracles_figure.py \
      LABELS.npz OUT RUN_DIR...

RUN_DIR names start with its oracle (saw_s0, flat_s3, ...)."""

import json
import os
import pickle
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # pylint: disable=wrong-import-position
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position

NAMES = ["TAG", "TAA", "TGA", "N", "N"]


def pattern(w):
    pos, closed = 0, set()
    for c in w:
        if c < 3:
            closed.add(pos % 3)
            pos += 3
        else:
            pos += 1
    return "".join("C" if i in closed else "." for i in range(3))


def nmi(a, b):
    def H(x):
        _, c = np.unique(x, return_counts=True)
        p = c / c.sum()
        return -(p * np.log(p)).sum()

    _, ab = np.unique(np.stack([a, b], 1).astype(str), axis=0, return_inverse=True)
    return (H(a) + H(b) - H(ab.ravel())) / max(1e-12, min(H(a), H(b)))


def table(dfa):
    states = sorted(dfa.states, key=str)
    index = {q: i for i, q in enumerate(states)}
    T = np.array([[index[dfa.transitions[q][c]] for c in range(5)] for q in states])
    return T, index[dfa.initial_state]


def ends(T, init, S):
    q = np.full(len(S), init)
    for t in range(S.shape[1]):
        q = T[q, S[:, t]]
    return q


def minimise(T, init, out):
    """Moore minimisation of the machine reachable from init."""
    seen, order = {init}, [init]
    for q in order:
        for p in T[q]:
            if p not in seen:
                seen.add(p)
                order.append(p)
    pos = {q: i for i, q in enumerate(order)}
    T = np.array([[pos[p] for p in T[q]] for q in order])
    out = np.asarray(out)[order]
    block = np.unique(out, return_inverse=True)[1].ravel()
    while True:
        new = np.unique(
            np.column_stack([block, block[T]]), axis=0, return_inverse=True
        )[1].ravel()
        if new.max() == block.max():
            break
        block = new
    n = block.max() + 1
    Tm, outm = np.zeros((n, T.shape[1]), int), np.zeros(n, int)
    Tm[block] = block[T]
    outm[block] = out
    return Tm, int(block[0]), outm


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
    for u in range(len(Tm)):
        style = (
            'shape=doublecircle,width=.3,style=filled,fillcolor="#cfe8cf"'
            if om[u]
            else 'style=filled,fillcolor="#eeeeee"'
        )
        lines.append(f"q{u}[{style}];")
        for v in sorted(set(Tm[u])):
            lab = edge_label([c for c in range(5) if Tm[u][c] == v])
            lines.append(f'q{u}->q{v}[label="{lab}"];')
    open(path + ".dot", "w").write("\n".join(lines + ["}"]))
    subprocess.run(
        ["dot", "-Tpng", "-Gdpi=200", path + ".dot", "-o", path + ".png"], check=True
    )


def main():
    labels_path, out, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(out, exist_ok=True)
    z = np.load(labels_path)
    S = z["s"].astype(np.int64)
    pat = np.array([pattern(w) for w in S])
    order = sorted(set(pat), key=lambda k: (k.count("C"), k))
    stops = (S < 3).sum(1)
    wild = (S >= 3).sum(1)
    length = 3 * stops + wild
    half = len(S) // 2
    fit, read = slice(0, half), slice(half, None)
    sub = slice(0, min(40_000, len(S)))
    rules = {
        "frame0 open": np.array([p[0] == "." for p in pat]),
        "frames01 not both": np.array([p[:2] != "CC" for p in pat]),
        "odd stops": stops % 2 == 1,
        "one stop": stops == 1,
    }
    for kind in ("saw", "flat"):
        y = z[kind][read].astype(float)
        print(
            kind,
            "rule phi:",
            "  ".join(
                f"{k} {np.corrcoef(v[read], y)[0, 1]:+.3f}" for k, v in rules.items()
            ),
        )

    rows, machines = [], {}
    for run in runs:
        name = os.path.basename(run.rstrip("/"))
        kind, seed = re.match(r"(saw|flat)_s(\d+)", name).groups()
        y = z[kind].astype(float)
        log = open(f"{run}/log").read().replace("\r", "\n")
        pick = [
            float(x) for x in re.findall(r"restart \d: .*held-out phi ([0-9.]+)", log)
        ]
        for i, p in enumerate(pick):
            dfa = pickle.load(open(f"{run}/search_{i}.pkl", "rb"))["dfa"]
            T, init = table(dfa)
            q = ends(T, init, S)
            accept = np.array(
                [
                    (q[fit] == u).any() and y[fit][q[fit] == u].mean() > y[fit].mean()
                    for u in range(len(T))
                ]
            )
            Tm, im, om = minimise(T, init, accept.astype(int))
            v = accept[q]
            k = max(range(2, 8), key=lambda m: nmi(q[sub], wild[sub] % m))
            row = dict(
                oracle=kind,
                seed=int(seed),
                restart=i,
                kept=p == max(pick),
                pick=p,
                phi=(
                    float(np.corrcoef(v[read], y[read])[0, 1]) if v[read].std() else 0.0
                ),
                states=len(T),
                minimal_states=len(Tm),
                count_modulus=k,
                nmi_count=float(nmi(q[sub], wild[sub] % k)),
                nmi_frame=float(nmi(q[sub], pat[sub])),
                stop_self_loops=float(
                    np.mean([T[u, c] == u for u in range(len(T)) for c in range(3)])
                ),
                agreement={r: float(np.mean(v == rule)) for r, rule in rules.items()},
                accept_by_pattern={o: float(v[pat == o].mean()) for o in order},
            )
            rows.append(row)
            machines[(kind, int(seed), i)] = (Tm, im, om, row)
            print(
                f"{kind:4s} s{seed} r{i}{'*' if row['kept'] else ' '} pick {p:.3f} phi {row['phi']:+.3f} "
                f"states {len(T):2d} (minimal {len(Tm):2d})  NMI(wildcards mod {k}) {row['nmi_count']:.2f} "
                f"NMI(frame) {row['nmi_frame']:.2f}  stop self-loops {row['stop_self_loops']:.2f}  "
                + "  ".join(f"{r} {a:.2f}" for r, a in row["agreement"].items()),
                flush=True,
            )
    json.dump(rows, open(f"{out}/decode.json", "w"), indent=1)

    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.3, wspace=0.18)
    lens = [36, 38, 40, 42, 44, 46]
    colors = plt.cm.tab10(np.arange(8))  # pylint: disable=no-member
    for col, (kind, title) in enumerate(
        (
            ("saw", "Sawtooth oracle (one threshold, at length 40)"),
            ("flat", "Per-length oracle (each length at its own median)"),
        )
    ):
        ax = fig.add_subplot(gs[0, col])
        y = z[kind].astype(float)
        for j, p in enumerate(order):
            xs = [l for l in lens if ((length == l) & (pat == p)).sum() >= 400]
            ax.plot(
                xs,
                [y[(length == l) & (pat == p)].mean() for l in xs],
                "o-",
                color=colors[j],
                label=p,
                ms=4,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("compiled length (bases) = 36 + 2 x stops")
        ax.set_xticks(lens)
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

    reps = [
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
    sub_gs = gs[1, 0].subgridspec(2, 1, hspace=0.35)
    for k, (key, caption) in enumerate(reps):
        if key not in machines:
            continue
        Tm, im, om, row = machines[key]
        path = f"{out}/dfa_{key[0]}_s{key[1]}_r{key[2]}"
        draw(Tm, im, om, path)
        ax = (
            fig.add_subplot(sub_gs[k]) if key[0] == "saw" else fig.add_subplot(gs[1, 1])
        )
        ax.imshow(mpimg.imread(path + ".png"))
        ax.axis("off")
        ax.set_title(f"{caption}  ({len(Tm)} states, phi {row['phi']:.3f})", fontsize=9)
    fig.savefig(f"{out}/figure.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{out}/figure.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
