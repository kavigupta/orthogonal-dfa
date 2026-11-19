"""Prints what the runs found, from their checked-in results, and writes
data/neural/raw_acgt/residual_products.json and figures/raw-acgt/stage_one.png.

    python -m orthogonal_dfa.experiments.raw_acgt.report"""

import io
import json
import subprocess

import numpy as np
from matplotlib import pyplot as plt

from ..capal_comparison.core import REPO_ROOT
from ..length_oracles.residual_report import minimize
from .common import (
    DATA,
    EVAL,
    FRAMES,
    RESIDUAL,
    RUNS,
    SEEDS,
    H,
    ResidualRaw,
    accepting,
    corr,
    end_states,
    features,
    kept_stage_one,
    load_flat,
    phi,
    saved_centred,
    top_features,
    verdict,
)


def stage_one_restarts():
    y = saved_centred("eval") > 0
    feats = features(EVAL)
    print(
        f"oracle accept {y.mean():.3f}",
        " ".join(f"{k} {corr(feats[k][H:], y[H:]):+.3f}" for k in FRAMES),
    )
    for s in SEEDS:
        with open(RUNS / f"s{s}" / "restarts.json") as f:
            restarts = json.load(f)
        kept = max(restarts, key=lambda r: r["pick_score"])["restart"]
        for r in restarts:
            d = r["dfa"]
            q = end_states(d, EVAL)
            v = verdict(q, y)
            own = np.array(d["accepting"], bool)[q]
            frames = " ".join(f"{k} {corr(v[H:], feats[k][H:]):+.2f}" for k in FRAMES)
            print(
                f"s{s} r{r['restart']}{'*' if r['restart'] == kept else ' '} "
                f"{r['ending']:>10s} probes {r['probes']:6d} states "
                f"{len(d['transitions']):3d} reached {len(np.unique(q)):3d} | pick "
                f"{r['pick_score']:+.3f} held-out {corr(v[H:], y[H:]):+.3f} own "
                f"{corr(own[H:], y[H:]):+.3f} accepts {v.mean():.2f} | {frames}"
            )


def listed(pairs):
    return ", ".join(f"{k} {v:+.2f}" for k, v in pairs)


def stage_one_features():
    y = saved_centred("eval") > 0
    feats = {k: v for k, v in features(EVAL).items() if k not in FRAMES}
    print("oracle:", listed(top_features(y, feats, 8)))
    for s in SEEDS:
        r = kept_stage_one(s)
        v = verdict(end_states(r["dfa"], EVAL), y)
        print(
            f"s{s} r{r['restart']} ({len(r['dfa']['transitions'])} states):",
            listed(top_features(v, feats)),
        )


def stage_one_states():
    y = saved_centred("eval") > 0
    for s in SEEDS:
        d = kept_stage_one(s)["dfa"]
        q = end_states(d, EVAL)
        own = [i for i, a in enumerate(d["accepting"]) if a]
        print(
            f"s{s} start {d['start']} own-accepting {own} "
            f"fit-accepting {sorted(accepting(q, y))}"
        )
        for st, row in enumerate(d["transitions"]):
            m = q == st
            rate = f"{y[m].mean():.3f}" if m.any() else "nan"
            print(f"  q{st:2d} -> {row}  share {m.mean():.3f}  oracle-accept {rate}")


def residual_sanity():
    flat = load_flat()
    c_fit, c_ev = saved_centred("fit"), saved_centred("eval")
    y = c_ev > 0
    direct = flat.membership_queries([r.tobytes() for r in EVAL[:2000]])
    print(
        f"flat accept {y.mean():.3f}; agrees with the oracle {(direct == y[:2000]).mean()}"
    )
    feats = features(EVAL)
    for s in SEEDS:
        st = kept_stage_one(s)["dfa"]
        p1 = minimize(st, accepting(end_states(st, EVAL), y))
        res = ResidualRaw(flat, p1, c_fit)
        q = end_states(p1, EVAL)
        r = c_ev > res.offsets[q]
        direct = res.membership_queries([x.tobytes() for x in EVAL[:2000]])
        rates = [
            round(float(r[q == k].mean()), 3) for k in range(len(p1["transitions"]))
        ]
        print(
            f"s{s}: stage 1 {len(st['transitions'])} st, minimal "
            f"{len(p1['transitions'])}; residual accept {r.mean():.3f} (per state "
            f"{rates}); phi with stage 1 {corr(verdict(q, y)[H:], r[H:]):+.3f}; with "
            f"flat {corr(r, y):+.3f}; agrees with the oracle {np.mean(direct == r[:2000])}"
        )
        print(
            "   residual vs:",
            ", ".join(f"{n} {v:+.3f}" for n, v in top_features(r, feats, 8)),
        )


def show(r):
    if r["stage"] == 1:
        print(
            f"s{r['seed']} stage 1: {r['stage_states']:3d} st | product "
            f"{r['product_states']:3d} min phi {r['product_phi']:.3f}"
        )
        return
    frames = " ".join(f"{k} {v:+.2f}" for k, v in r["stage_frames"].items())
    print(
        f"s{r['seed']} stage {r['stage']}: {r['stage_states']:3d} st residual phi "
        f"{r['residual_phi']:.3f} (flat {r['stage_flat_phi']:+.3f}) | product "
        f"{r['reachable']:3d} reach {r['product_states']:3d} min phi "
        f"{r['product_phi']:.3f} | {r['hours']:.2f} h | frames {frames}"
    )
    restarts = " ".join(
        f"r{x['restart']}:{x['states']}st/{x['residual_phi']:.3f}/{x['ending']}"
        for x in r["restarts"]
    )
    print(
        "      stage tracks:",
        ", ".join(f"{n} {v:+.2f}" for n, v in r["stage_features"]),
        "| product:",
        ", ".join(f"{n} {v:+.2f}" for n, v in r["product_features"][:3]),
        "| restarts:",
        restarts,
    )


def residual_stages():
    out = []
    for s in SEEDS:
        with open(RESIDUAL / f"s{s}" / "stages.jsonl") as f:
            for line in f:
                r = json.loads(line)
                show(r)
                out.append(
                    dict(
                        seed=s,
                        stage=r["stage"],
                        restart=r["restart"],
                        stage_dfa=r["dfa"],
                        product=r["product"],
                        product_phi=r["product_phi"],
                        residual_phi=r.get("residual_phi"),
                    )
                )
    with open(DATA / "residual_products.json", "w") as f:
        json.dump(out, f, indent=1)


def dot_png(table, rates):
    """Each state is labelled with the flat oracle's accept rate on the strings ending
    there; edges are labelled with their bases."""
    lines = [
        "digraph{rankdir=LR; dpi=150;",
        "node[shape=circle,fontname=Helvetica,fontsize=10];",
        "edge[fontname=Helvetica,fontsize=10];",
        "start[shape=point]; start->q0;",
    ]
    for q, row in enumerate(table["transitions"]):
        shape = "doublecircle" if table["accepting"][q] else "circle"
        lines.append(f'q{q}[shape={shape},label="{rates[q]:.2f}"];')
        for t in sorted(set(row)):
            bases = "".join(b for b, n in zip("ACGT", row) if n == t)
            lines.append(f'q{q}->q{t}[label="{"*" if len(bases) == 4 else bases}"];')
    png = subprocess.run(
        ["dot", "-Tpng"],
        input="\n".join(lines + ["}"]).encode(),
        capture_output=True,
        check=True,
    ).stdout
    return plt.imread(io.BytesIO(png))


def draw_stage_one():
    """The kept stage-1 DFAs, minimized under their held-out accepting states."""
    y = saved_centred("eval") > 0
    panels = []
    for s in SEEDS:
        r = kept_stage_one(s)
        q = end_states(r["dfa"], EVAL)
        table = minimize(r["dfa"], accepting(q, y))
        qm = end_states(table, EVAL)
        rates = [y[qm == k].mean() for k in range(len(table["transitions"]))]
        title = f"seed {s}: {len(table['transitions'])} states, held-out phi {phi(q, y):.3f}"
        panels.append((title, dot_png(table, rates)))
    # padded to one size so every panel is drawn at the same scale
    h, w = (max(img.shape[i] for _, img in panels) for i in (0, 1))
    fig, axes = plt.subplots(2, 2, figsize=(2 * w / 150, 2 * h / 150 + 0.6))
    for (title, img), ax in zip(panels, axes.ravel()):
        canvas = np.ones((h, w, img.shape[2]), img.dtype)
        top, left = (h - img.shape[0]) // 2, (w - img.shape[1]) // 2
        canvas[top : top + img.shape[0], left : left + img.shape[1]] = img
        ax.imshow(canvas)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    out = REPO_ROOT / "figures" / "raw-acgt" / "stage_one.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)


def main():
    stage_one_restarts()
    stage_one_features()
    stage_one_states()
    residual_sanity()
    residual_stages()
    draw_stage_one()


if __name__ == "__main__":
    main()
