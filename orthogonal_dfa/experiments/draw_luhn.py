"""
figures/luhn/luhn_dfa.pdf: `luhn_dfa`, sized to fill half the width of a
full-width figure.

    python -m orthogonal_dfa.experiments.draw_luhn
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from orthogonal_dfa.experiments.capal_comparison.core import REPO_ROOT
from orthogonal_dfa.experiments.random_minimal.analyze import PAPER_RC
from orthogonal_dfa.l_star.examples.luhn import luhn_dfa

OUT_PATH = REPO_ROOT / "figures" / "luhn" / "luhn_dfa.pdf"
# the paper's disagree, agree and 1F5FA8!50!white
REJECT, ACCEPT, EDGE = "#E63A7C", "#2BCC4B", "#8FAFD3"
# inches: half the textwidth of a two-column ACM paper
WIDTH = 3.5
# per ring d = a - b, its rank outward from the center and its rotation in
# degrees, annealed to minimize edge crossings
RING_RANK = [7, 3, 2, 8, 5, 0, 6, 9, 1, 4]
RING_ROTATION = [245.7, 276.6, 96.3, 134.4, 135.4, 339.9, 351.3, 26.6, 34.9, 244.1]


def position(state: int) -> np.ndarray:
    """State 10 a + b on ring a - b, at angle 36 b plus the ring's rotation.

    (a, b) -> (a + 1, b + 1) maps each digit's transitions onto themselves, and
    here it is a rotation by 36 degrees.
    """
    a, b = divmod(state, 10)
    d = (a - b) % 10
    theta = np.deg2rad(36 * b + RING_ROTATION[d])
    return (3 + RING_RANK[d]) * np.array([np.cos(theta), np.sin(theta)])


def draw():
    dfa = luhn_dfa()
    pos = {q: position(q) for q in dfa.states}
    edges = [
        (pos[q], pos[r]) for q, row in dfa.transitions.items() for r in row.values()
    ]
    fig = plt.figure(figsize=(WIDTH, WIDTH))
    ax = fig.add_axes((0.01, 0.01, 0.98, 0.98))
    ax.add_collection(LineCollection(edges, colors=EDGE, linewidths=0.3))
    for accept, color in [(False, REJECT), (True, ACCEPT)]:
        states = [q for q in dfa.states if (q in dfa.final_states) == accept]
        ax.scatter(
            *np.array([pos[q] for q in states]).T,
            s=30,
            c=color,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PAPER_RC):
        fig = draw()
        # no creation date, so redrawing leaves the file alone
        fig.savefig(OUT_PATH, metadata={"CreationDate": None})
        plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
