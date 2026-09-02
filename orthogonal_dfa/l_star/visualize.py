"""Diagnostics renderer for a :class:`TransitionResolver` against a known DFA.

Produces three panels, all keyed by one colour per learned Myhill-Nerode class:

  1. the *true* DFA, each state drawn as a pie over the classes that random
     strings reaching that state actually sift to (plus an indecisive slice --
     the family's false negatives, which sift to ``None``);
  2. the discrimination tree, every internal node annotated with its midfix
     (the ``prepend`` that sits between a prefix and each base suffix);
  3. the DFA over the learned classes.

``dot`` does the graph layout (via its ``plain`` output, which is just node and
spline coordinates) and matplotlib does the drawing, so the panels keep an equal
aspect and stay sharp at any dpi.
"""

import shlex
import subprocess
from collections import Counter
from typing import Dict, List, Optional

import matplotlib
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Wedge
from matplotlib.path import Path

# Categorical slots, in the fixed order that clears the CVD gates.  Classes past
# the eighth fold into OTHER rather than inventing a hue.
_PALETTE = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
_OTHER = "#8a8a86"
_INDECISIVE = "#d9d9d4"  # the family can't place these -- the FN slice
_INK = "#1a1a1a"
_MUTED = "#6d6d68"


def _fmt(seq) -> str:
    """A midfix/access string as text; the empty string renders as epsilon."""
    return "".join(str(c) for c in seq) if len(seq) else "ε"


def _class_colors(classes) -> Dict[int, str]:
    return {
        c: (_PALETTE[i] if i < len(_PALETTE) else _OTHER)
        for i, c in enumerate(sorted(classes))
    }


def sample_class_distribution(
    classify, true_dfa, *, pst, rng, num_samples=500, per_state=60, prefill=None
):
    """For each true state, the distribution of learned classes that random
    strings *reaching* that state classify to.  ``None`` keys the indecisive
    (false-negative) share.  Returns ``{true_state: Counter}``.

    ``classify`` maps a string to a class id or ``None``, and ``pst`` supplies the
    sampler and alphabet, so this works for any learner: pass
    ``learner.sift`` for direct-L*, or ``lambda s: dt.classify(s, oracle)`` for a
    resolver-built decision tree.  ``prefill`` is an optional hook that warms a
    whole batch of strings at once (e.g. ``learner._sift_prefill``); without it
    the classifications simply cost more oracle calls.

    Every prefix is bucketed by the state it reaches, not just each sampled
    string's end state: the sampler draws one fixed length, so a transient state
    -- one no string of that length can end in -- would otherwise draw no samples
    at all and its classes would vanish from the picture.  Buckets are deduped and
    capped so a rare state is represented as well as a common one."""
    buckets: Dict[int, List[bytes]] = {s: [] for s in true_dfa.states}
    seen: Dict[int, set] = {s: set() for s in true_dfa.states}
    for _ in range(num_samples):
        w = pst.sampler.sample(rng, pst.alphabet_size)
        state = true_dfa.initial_state
        for i, c in enumerate([None, *w]):
            if c is not None:
                state = true_dfa.transitions[state][c]
            key = w[:i]
            if len(buckets[state]) < per_state and key not in seen[state]:
                seen[state].add(key)
                buckets[state].append(key)
    if prefill is not None:
        prefill([p for ps in buckets.values() for p in ps])
    dist: Dict[int, Counter] = {s: Counter() for s in true_dfa.states}
    for state, ps in buckets.items():
        for p in ps:
            dist[state][classify(p)] += 1
    return dist


def _breakdown(counter: Counter) -> str:
    total = sum(counter.values())
    if not total:
        return "no samples"
    parts = "  ".join(
        ("ind" if cls is None else f"q{cls}") + f" {100 * n / total:.0f}%"
        for cls, n in counter.most_common()
    )
    return f"{parts}  n={total}"


# ---------------------------------------------------------------------------
# layout: dot computes it, we only read the coordinates back
# ---------------------------------------------------------------------------


def _dot_layout(nodes, edges, *, rankdir="LR", nodesep=0.45, ranksep=0.6) -> dict:
    """Run ``dot`` over a node/edge list and parse its ``plain`` output.

    ``nodes`` is ``{name: (width, height)}`` in inches, ``edges`` a list of
    ``(tail, head, label)``.  Returns node centres/sizes and edge splines in
    inches, y pointing up -- ready to draw at equal aspect."""
    lines = [
        "digraph {",
        f"  rankdir={rankdir}; nodesep={nodesep}; ranksep={ranksep};",
        '  node [shape=circle, fixedsize=true, label=""];',
    ]
    for name, (w, h) in nodes.items():
        lines.append(f'  "{name}" [width={w:.4f}, height={h:.4f}];')
    for tail, head, label in edges:
        lab = f' [label="{label}"]' if label else ""
        lines.append(f'  "{tail}" -> "{head}"{lab};')
    lines.append("}")
    plain = subprocess.run(
        ["dot", "-Tplain"],
        input="\n".join(lines),
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    out = {"nodes": {}, "edges": [], "size": (1.0, 1.0)}
    for line in plain.splitlines():
        f = shlex.split(line)
        if not f:
            continue
        if f[0] == "graph":
            out["size"] = (float(f[2]), float(f[3]))
        elif f[0] == "node":
            out["nodes"][f[1]] = {
                "xy": (float(f[2]), float(f[3])),
                "wh": (float(f[4]), float(f[5])),
            }
        elif f[0] == "edge":
            n = int(f[3])
            pts = [(float(f[4 + 2 * i]), float(f[5 + 2 * i])) for i in range(n)]
            rest = f[4 + 2 * n :]
            label = None
            if rest and rest[0] not in ("solid", "dashed", "dotted", "bold"):
                label = (rest[0], (float(rest[1]), float(rest[2])))
            out["edges"].append(
                {"tail": f[1], "head": f[2], "pts": pts, "label": label}
            )
    return out


def _draw_edges(ax, layout, *, fontsize):
    for e in layout["edges"]:
        pts = e["pts"]
        # dot emits a cubic B-spline: one anchor then triplets of control points.
        verts, codes = [pts[0]], [Path.MOVETO]
        for i in range(1, len(pts) - 2, 3):
            verts += pts[i : i + 3]
            codes += [Path.CURVE4] * 3
        ax.add_patch(
            PathPatch(
                Path(verts, codes), fill=False, edgecolor=_MUTED, lw=0.9, zorder=1
            )
        )
        tip, prev = np.array(verts[-1]), np.array(verts[-2])
        d = tip - prev
        if np.hypot(*d):
            d = d / np.hypot(*d)
            ax.add_patch(
                FancyArrowPatch(
                    tuple(tip - d * 0.06),
                    tuple(tip),
                    arrowstyle="-|>",
                    mutation_scale=8,
                    color=_MUTED,
                    lw=0,
                    zorder=1,
                )
            )
        if e["label"]:
            text, (lx, ly) = e["label"]
            ax.text(
                lx,
                ly,
                text,
                fontsize=fontsize,
                color=_MUTED,
                ha="center",
                va="center",
                zorder=2,
            )


def _pie_node(ax, xy, r, shares, *, double=False, lw=1.0):
    """A node drawn as a pie over ``shares`` -- a list of (colour, fraction)."""
    start = 90.0
    for colour, frac in shares:
        if frac <= 0:
            continue
        end = start - 360.0 * frac
        ax.add_patch(
            Wedge(xy, r, end, start, facecolor=colour, edgecolor="none", zorder=3)
        )
        start = end
    ax.add_patch(Circle(xy, r, fill=False, edgecolor=_INK, lw=lw, zorder=4))
    if double:
        ax.add_patch(
            Circle(xy, r * 0.84, fill=False, edgecolor=_INK, lw=lw * 0.8, zorder=4)
        )


def _finish(ax, layout, pad=0.35):
    w, h = layout["size"]
    ax.set_xlim(-pad, w + pad)
    ax.set_ylim(-pad, h + pad)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------------
# the three panels
# ---------------------------------------------------------------------------


def _panel_true_dfa(ax, true_dfa, dist, colors):
    r = 0.30
    nodes = {str(s): (2 * r, 2 * r) for s in sorted(true_dfa.states)}
    nodes["__start"] = (0.06, 0.06)
    edges = [("__start", str(true_dfa.initial_state), None)]
    for s in sorted(true_dfa.states):
        merged: Dict[int, List[int]] = {}
        for c, t in sorted(true_dfa.transitions[s].items()):
            merged.setdefault(t, []).append(c)
        for t, syms in merged.items():
            edges.append((str(s), str(t), ",".join(str(c) for c in syms)))
    layout = _dot_layout(nodes, edges, rankdir="LR", nodesep=0.9, ranksep=1.3)
    _draw_edges(ax, layout, fontsize=6.5)
    for s in sorted(true_dfa.states):
        counter = dist.get(s, Counter())
        total = sum(counter.values()) or 1
        shares = [
            (_INDECISIVE if cls is None else colors.get(cls, _OTHER), n / total)
            for cls, n in counter.most_common()
        ] or [(_INDECISIVE, 1.0)]
        impure = len([c for c in counter if c is not None]) > 1 or counter.get(None)
        xy = layout["nodes"][str(s)]["xy"]
        _pie_node(
            ax,
            xy,
            r,
            shares,
            double=s in true_dfa.final_states,
            lw=2.0 if impure else 1.0,
        )
        ax.text(
            xy[0],
            xy[1],
            str(s),
            fontsize=8.5,
            weight="bold",
            color="white",
            ha="center",
            va="center",
            zorder=5,
            path_effects=_halo(),
        )
        ax.text(
            xy[0],
            xy[1] - r - 0.13,
            _breakdown(counter),
            fontsize=5.6,
            color=_MUTED,
            ha="center",
            va="top",
            zorder=5,
        )
    _finish(ax, layout, pad=0.5)


def _walk_tree(node, colors, nodes, edges, labels, *, uid=None):
    uid = [0] if uid is None else uid
    uid[0] += 1
    name = f"n{uid[0]}"
    if isinstance(node, int):
        nodes[name] = (0.44, 0.30)
        labels[name] = ("leaf", f"q{node}", colors.get(node, _OTHER))
        return name
    prepend, lookup = node
    text = _fmt(prepend)
    nodes[name] = (max(0.52, 0.16 + 0.085 * len(text)), 0.30)
    labels[name] = ("mid", text, None)
    for side in (True, False):
        child = _walk_tree(lookup[side], colors, nodes, edges, labels, uid=uid)
        edges.append((name, child, "A" if side else "R"))
    return name


def _tree_root(learner):
    """The learner's discrimination-tree root, however it stores its tree."""
    tree = getattr(learner, "tree", None)
    return learner.dt if tree is None else tree.root


def _panel_tree(ax, dt, colors):
    nodes, edges, labels = {}, [], {}
    _walk_tree(dt, colors, nodes, edges, labels)
    layout = _dot_layout(nodes, edges, rankdir="TB", nodesep=0.3, ranksep=0.45)
    _draw_edges(ax, layout, fontsize=6)
    for name, (kind, text, colour) in labels.items():
        xy = layout["nodes"][name]["xy"]
        w, h = layout["nodes"][name]["wh"]
        if kind == "leaf":
            ax.add_patch(
                matplotlib.patches.FancyBboxPatch(
                    (xy[0] - w / 2, xy[1] - h / 2),
                    w,
                    h,
                    boxstyle="round,pad=0,rounding_size=0.08",
                    facecolor=colour,
                    edgecolor="none",
                    zorder=3,
                )
            )
            ax.text(
                *xy,
                text,
                fontsize=7.5,
                color="white",
                weight="bold",
                ha="center",
                va="center",
                zorder=4,
            )
        else:
            ax.add_patch(
                matplotlib.patches.FancyBboxPatch(
                    (xy[0] - w / 2, xy[1] - h / 2),
                    w,
                    h,
                    boxstyle="square,pad=0",
                    facecolor="#ffffff",
                    edgecolor=_INK,
                    lw=0.9,
                    zorder=3,
                )
            )
            ax.text(
                *xy,
                text,
                fontsize=7.5,
                color=_INK,
                family="monospace",
                ha="center",
                va="center",
                zorder=4,
            )
    _finish(ax, layout, pad=0.3)


def _sift_fn(learner):
    """The learner's classifier, wherever it keeps it.

    A sifter reports the boundary along with the leaf, which every other caller
    wants and a render does not, so the discarding happens here rather than as a
    second method on the sifter for this one caller to find.
    """
    fn = getattr(learner, "sift", None)
    if fn is not None:
        return fn
    sifter = getattr(learner, "sifter", None)
    if sifter is not None:
        return lambda seq: sifter.sift_and_boundary(seq)[0]
    raise AttributeError("learner exposes no sift")


def _prefill_fn(learner):
    """The batched warm-up for a whole set of strings, if there is one."""
    for holder, name in ((learner, "_sift_prefill"), (learner, "sifter")):
        if name == "sifter":
            sifter = getattr(learner, "sifter", None)
            if sifter is not None:
                return getattr(sifter, "prefill", None)
        else:
            fn = getattr(holder, name, None)
            if fn is not None:
                return fn
    return None


def _resolved_edges(learner):
    """The learner's transition function, wherever it keeps it."""
    for attr in ("dfa", None):
        holder = learner if attr is None else getattr(learner, attr, None)
        edges = getattr(holder, "transitions", None)
        if edges is not None:
            return edges
    raise AttributeError("learner exposes no transition function")


def _panel_class_dfa(ax, learner, colors, final_states, flipped):
    # Whatever edges the learner has resolved so far, which is all a diagnostic
    # needs -- however it happens to store them.
    transitions = _resolved_edges(learner)
    finals = set(final_states or ())
    r = 0.26
    nodes = {str(s): (2 * r, 2 * r) for s in sorted(transitions)}
    nodes["__start"] = (0.06, 0.06)
    edges = [("__start", "0", None)]
    for s in sorted(transitions):
        merged: Dict[int, List[int]] = {}
        for c, t in sorted(transitions[s].items()):
            merged.setdefault(t, []).append(c)
        for t, syms in merged.items():
            edges.append((str(s), str(t), ",".join(str(c) for c in syms)))
    layout = _dot_layout(nodes, edges, rankdir="LR", nodesep=0.7, ranksep=1.0)
    _draw_edges(ax, layout, fontsize=6.5)
    for s in sorted(transitions):
        xy = layout["nodes"][str(s)]["xy"]
        ax.add_patch(
            Circle(xy, r, facecolor=colors.get(s, _OTHER), edgecolor="none", zorder=3)
        )
        ax.add_patch(
            Circle(
                xy,
                r,
                fill=False,
                zorder=4,
                edgecolor="#c0392f" if s in flipped else _INK,
                lw=2.4 if s in flipped else 1.0,
            )
        )
        if s in finals:
            ax.add_patch(
                Circle(xy, r * 0.84, fill=False, edgecolor=_INK, lw=0.8, zorder=4)
            )
        ax.text(
            xy[0],
            xy[1],
            f"q{s}",
            fontsize=8,
            weight="bold",
            color="white",
            ha="center",
            va="center",
            zorder=5,
            path_effects=_halo(),
        )
        access = learner.access.get(s)
        if access is not None:
            ax.text(
                xy[0],
                xy[1] - r - 0.11,
                _fmt(access)[:16],
                fontsize=5.4,
                color=_MUTED,
                ha="center",
                va="top",
                zorder=5,
            )
    _finish(ax, layout, pad=0.4)


def _halo():
    from matplotlib import patheffects  # pylint: disable=import-outside-toplevel

    return [patheffects.withStroke(linewidth=1.6, foreground="#00000055")]


def render_diagnostics(
    learner,
    true_dfa,
    *,
    rng,
    path,
    pst=None,
    num_samples: int = 500,
    per_state: int = 60,
    final_states: Optional[set] = None,
    flipped=(),
    scale: float = 1.0,
    dpi: int = 220,
) -> str:
    """Write the three panels, stacked, to ``path``.  Returns the path.

    ``pst`` defaults to the learner's own; pass it explicitly when the learner
    does not carry one."""
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    dist = sample_class_distribution(
        _sift_fn(learner),
        true_dfa,
        pst=pst if pst is not None else learner.pst,
        rng=rng,
        num_samples=num_samples,
        per_state=per_state,
        prefill=_prefill_fn(learner),
    )
    colors = _class_colors(set(range(learner.num_states)))
    panels = [
        (
            "true DFA — shaded by the classes its strings sift to",
            lambda a: _panel_true_dfa(a, true_dfa, dist, colors),
        ),
        (
            "discrimination tree — every internal node is a midfix",
            lambda a: _panel_tree(a, _tree_root(learner), colors),
        ),
        (
            "learned Myhill–Nerode classes",
            lambda a: _panel_class_dfa(a, learner, colors, final_states, flipped),
        ),
    ]

    # Measure every panel first.  Each is then drawn at the *same* scale (one
    # layout inch to one figure inch) rather than stretched to a shared width,
    # so nodes are the same size in all three and nothing is distorted.
    probe = plt.figure()
    extents = []
    for _, draw in panels:
        ax = probe.add_subplot(111)
        draw(ax)
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        extents.append((x1 - x0, y1 - y0))
        probe.clf()
    plt.close(probe)

    sheet_w = max(w for w, _ in extents) * scale
    heights = [h * scale for _, h in extents]
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(sheet_w, sum(heights) + 1.0),
        gridspec_kw={"height_ratios": heights, "hspace": 0.10},
    )
    fig.patch.set_facecolor("white")
    for ax, (title, draw), (w, _h) in zip(axes, panels, extents):
        draw(ax)
        # Widen the view to the sheet so every panel shares one scale, centred.
        x0, x1 = ax.get_xlim()
        slack = (sheet_w / scale - w) / 2
        ax.set_xlim(x0 - slack, x1 + slack)
        ax.set_title(title, fontsize=8, color=_MUTED, loc="left", pad=6)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
