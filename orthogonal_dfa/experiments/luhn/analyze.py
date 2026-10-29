"""
summary.tex, the `random_minimal.analyze` summary of the results `run` caches,
as a two-column table of measures.

    python -m orthogonal_dfa.experiments.luhn.analyze
"""

from typing import Sequence

from ..capal_comparison.core import REPO_ROOT
from ..random_minimal.analyze import load_results, summary_cells
from .run import DATA_DIR

OUT_DIR = REPO_ROOT / "figures" / "luhn"


def summary(results: Sequence[dict]) -> str:
    etas = {r["eta"] for r in results}
    assert len(etas) == 1, etas
    _, exact, lowest, mq_mean, mq_max, rounds_mean, rounds_max = summary_cells(results)
    rows = [
        ["Exact", exact, r"MQ geo.\ mean", mq_mean],
        ["Lowest accuracy", lowest, "MQ max", mq_max],
        [r"Rounds geo.\ mean", rounds_mean, "Rounds max", rounds_max],
    ]
    return "\n".join(
        [r"\begin{tabular}{lr@{\hspace{2em}}lr}"]
        + ["    " + " & ".join(row) + r" \\" for row in rows]
        + [r"\end{tabular}"]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = OUT_DIR / "summary.tex"
    tex_path.write_text(summary(load_results(DATA_DIR)) + "\n")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
