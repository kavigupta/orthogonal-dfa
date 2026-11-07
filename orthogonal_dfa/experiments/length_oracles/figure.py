"""The length-oracle figure's panels, and the numbers its caption uses, from the
checked-in runs and labels alone.

    python -m orthogonal_dfa.experiments.length_oracles.figure

A run's DFA's state accepts iff its fit-half strings accept above the fit half's rate; phi
is on the read half.  A residual run's first stage and product are scored the same way
on the flat labels, as are rules on which frames hold a stop; its residual DFA keeps the
accepting states saved with it.  Writes to figures/length-oracles/
the panels (lengths.pdf, dfa_saw.tex, and dfa_first.tex, dfa_residual.tex and
dfa_product.tex for the product), values.tex (macros for the caption's numbers) and
phis.tex (macros for the phis the text compares)."""

import json

import numpy as np
from automata.fa.dfa import DFA
from matplotlib import pyplot as plt
from matplotlib import transforms

from orthogonal_dfa.l_star.dfa_utils import phi_maximizing_states, table_end_states

from ..capal_comparison.core import REPO_ROOT
from ..random_minimal.analyze import PAPER_COLORS, PAPER_RC
from .dfa_structure import counts_wildcards, dfa_accepts, tracks_phase
from .label import saved_labels
from .oracles import SYMBOLS, VOCAB
from .render_dfa import dfa_tikz

RUNS = REPO_ROOT / "data" / "neural" / "length_oracles" / "runs.json"
RANDOM_STRINGS = (
    REPO_ROOT / "data" / "neural" / "length_oracles" / "random_strings.json"
)
RESIDUAL_PRODUCTS = (
    REPO_ROOT / "data" / "neural" / "length_oracles" / "residual_products.json"
)
OUT_DIR = REPO_ROOT / "figures" / "length-oracles"
PAPER_NAMES = {"saw": "Improperly Calibrated", "flat": "Correctly Calibrated"}
#: The DFAs drawn, by (oracle, seed, restart).  Only the restart a run kept may be drawn.
REPRESENTATIVES = {"saw": ("saw", 1, 0)}
#: The residual seed whose first stage, residual DFA and product are drawn.
PRODUCT_SEED = 1
#: Each drawn part of that run, by its key in residual_products.json.  The residual DFA
#: reads the residual oracle, whose labels are not saved, so it keeps the accepting states
#: saved with it and has no phi.
PRODUCT_PARTS = {"first_stage": "first", "residual": "residual", "product": "product"}
#: The parts with a phi, by their name in phis.tex.
PHI_NAMES = {"first": "Original", "product": "Product"}
#: Rules on which frames are closed, by (string, frame), scored on the flat labels.
FRAME_RULES = {
    "NotFramesZeroAndOneClosed": lambda closed: ~(closed[:, 0] & closed[:, 1]),
    "NotAllFramesClosed": lambda closed: ~closed.all(1),
}


class Strings:
    def __init__(self, s):
        self.s = s
        self.read = np.arange(len(s)) >= len(s) // 2
        self.stops = (s < 3).sum(1)
        # A stop closes the frame of the base it starts at, mod 3; a stop is 3 bases, a
        # wildcard 1.
        self.closed = np.zeros((len(s), 3), bool)
        position = np.zeros(len(s), int)
        for t in range(s.shape[1]):
            stop = s[:, t] < 3
            self.closed[stop, position[stop] % 3] = True
            position += np.where(stop, 3, 1)


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
        minimal=minimal_dfa(restart["dfa"], verdict),
        phi=phi(v[read], y[read]),
        modulus=modulus,
    )


def minimal_dfa(table, accepting):
    """States numbered breadth-first from the start: minify's numbering changes with the
    hash seed, and graphviz's layout with the numbering."""
    dfa = DFA(
        states=set(range(len(table["transitions"]))),
        input_symbols=set(range(VOCAB.alphabet_size)),
        transitions={
            q: {c: int(t) for c, t in enumerate(row)}
            for q, row in enumerate(table["transitions"])
        },
        initial_state=table["start"],
        final_states={q for q, a in enumerate(accepting) if a},
    ).minify()
    order = [dfa.initial_state]
    i = 0
    while i < len(order):
        for c in sorted(dfa.input_symbols):
            t = dfa.transitions[order[i]][c]
            if t not in order:
                order.append(t)
        i += 1
    number = {q: i for i, q in enumerate(order)}
    return DFA(
        states=set(number.values()),
        input_symbols=dfa.input_symbols,
        transitions={
            number[q]: {c: number[t] for c, t in dfa.transitions[q].items()}
            for q in order
        },
        initial_state=0,
        final_states={number[q] for q in dfa.final_states},
    )


def product_parts(y, data, run):
    """Each part's minimal DFA, and its phi on the flat labels where it reads them, with
    accepting states as held_out_phi fits them."""
    parts = {}
    for key, name in PRODUCT_PARTS.items():
        table = run[key]
        if key == "residual":
            parts[name] = dict(minimal=minimal_dfa(table, table["accepting"]))
            continue
        transitions = np.array(table["transitions"])
        end = table_end_states(transitions, table["start"], data.s)
        fit, read = ~data.read, data.read
        counts = np.bincount(end[fit], minlength=len(transitions))
        accepts = np.bincount(
            end[fit], weights=y["flat"][fit], minlength=len(transitions)
        )
        chosen = phi_maximizing_states(
            {q: c for q, c in enumerate(counts) if c},
            {q: a for q, a in enumerate(accepts) if counts[q]},
        )
        accepting = np.isin(np.arange(len(transitions)), list(chosen))
        parts[name] = dict(
            minimal=minimal_dfa(table, accepting),
            phi=phi(accepting[end][read].astype(float), y["flat"][read]),
        )
    return parts


def compiled_lengths(data):
    """From the shortest to the 99th percentile."""
    lengths = 36 + 2 * data.stops
    return list(range(lengths.min(), int(np.percentile(lengths, 99)) + 1, 2))


def plot_lengths(data, random_strings):
    """Each oracle's accept rate on uniform random strings of the compiled lengths,
    drawn at the size it is printed."""
    rows = [r for r in random_strings if r["length"] in compiled_lengths(data)]
    xs = [r["length"] for r in rows]
    fig = plt.figure(figsize=(1.85, 2.1))
    ax = fig.add_axes((0.26, 0.5, 0.7, 0.47))
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
        ncol=1,
    )
    return fig


def write_lengths(data):
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


def describe_all(restarts, y, data):
    described = {}
    for key, r in restarts.items():
        d = described[key] = describe(r, y[r["oracle"]], data)
        picks = [
            s["pick_score"] for k, s in sorted(restarts.items()) if k[:2] == key[:2]
        ]
        # As counterexample_synthesis keeps it: the first best.
        d["kept"] = key[2] == int(np.argmax(picks))
    return described


def join_words(items, conjunction):
    items = [str(x) for x in items]
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" {conjunction} " + items[-1]


def values(described, data, parts):
    """Macros for the caption: each drawn DFA's numbers, named after it (saw gives
    \\loSawPhi), and how many kept DFAs of each oracle count wildcards or track the
    phase."""

    def macro(name, value):
        return rf"\newcommand{{\lo{name}}}{{{value}}}"

    lines = []
    for name, key in REPRESENTATIVES.items():
        d, prefix = described[key], name.title()
        lines += [
            macro(f"{prefix}Phi", f"{d['phi']:.3f}"),
            macro(f"{prefix}States", len(d["minimal"].states)),
        ]
        if key[0] == "saw":
            k, dfa = d["modulus"], d["minimal"]
            lines += [
                macro(f"{prefix}Modulus", k),
                macro(
                    f"{prefix}Wildcards",
                    join_words(
                        [w for w in range(k) if dfa_accepts(dfa, [3] * w)], "or"
                    ),
                ),
            ]
            lengths = [
                L
                for L in compiled_lengths(data)
                if dfa_accepts(dfa, [0] * ((L - 36) // 2) + [3] * ((108 - L) // 2))
            ]
            # "lengths 38 or 42" for which lengths it accepts, "38 and 42" for both
            lines += [
                macro(f"{prefix}Lengths", join_words(lengths, "or")),
                macro(f"{prefix}LengthsAnd", join_words(lengths, "and")),
            ]
    for name, d in parts.items():
        lines.append(macro(f"{name.title()}States", len(d["minimal"].states)))
    for kind in PAPER_NAMES:
        kept = [d for key, d in described.items() if key[0] == kind and d["kept"]]
        counting = sum(counts_wildcards(d["minimal"]) for d in kept)
        phase = sum(tracks_phase(d["minimal"]) for d in kept)
        lines += [
            macro(f"{kind.title()}Counting", f"{counting}/{len(kept)}"),
            macro(f"{kind.title()}Phase", f"{phase}/{len(kept)}"),
        ]
    return "\n".join(lines + [""])


def phis(data, parts, product_phis, y):
    """Macros for the product's parts' and the frame rules' phis, e.g. \\phiProduct, and
    \\phiMin, the lowest product phi over the residual seeds."""
    found = {phi_name: parts[name]["phi"] for name, phi_name in PHI_NAMES.items()}
    found["Min"] = min(product_phis)
    for name, rule in FRAME_RULES.items():
        v = rule(data.closed)[data.read].astype(float)
        found[name] = phi(v, y["flat"][data.read])
    return "".join(
        rf"\newcommand{{\phi{name}}}{{{value:.3f}}}" + "\n"
        for name, value in found.items()
    )


def main():
    strings, labels = saved_labels()
    data = Strings(np.array(strings, dtype=np.int64))
    y = {kind: labels[kind].astype(float) for kind in PAPER_NAMES}
    with open(RUNS) as f:
        restarts = {
            (r["oracle"], r["seed"], r["restart"]): r
            for r in json.load(f)["restarts"]
            if r["oracle"] in PAPER_NAMES
        }
    described = describe_all(restarts, y, data)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, key in REPRESENTATIVES.items():
        if not described[key]["kept"]:
            raise ValueError(f"{key} is not the restart its run kept")
        (OUT_DIR / f"dfa_{name}.tex").write_text(dfa_tikz(described[key]["minimal"]))
    with open(RESIDUAL_PRODUCTS) as f:
        runs = {r["seed"]: r for r in json.load(f)}
    parts = product_parts(y, data, runs[PRODUCT_SEED])
    product_phis = [
        product_parts(y, data, run)["product"]["phi"] for run in runs.values()
    ]
    if parts["product"]["phi"] != max(product_phis):
        raise ValueError(
            f"seed {PRODUCT_SEED}'s product is not the best of {product_phis}"
        )
    for name, d in parts.items():
        (OUT_DIR / f"dfa_{name}.tex").write_text(dfa_tikz(d["minimal"]))
    (OUT_DIR / "values.tex").write_text(values(described, data, parts))
    (OUT_DIR / "phis.tex").write_text(phis(data, parts, product_phis, y))
    write_lengths(data)


if __name__ == "__main__":
    main()
