"""
summary.tex, the `random_minimal.analyze` summary of the results `run` caches,
as a two-column table of measures, and examples.tex, sampled strings with their
true label and the seed-0 oracle's answer.

    python -m orthogonal_dfa.experiments.luhn.analyze
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.examples.luhn import luhn_dfa
from orthogonal_dfa.l_star.structures import NoisyOracle, SymmetricBernoulli

from ..capal_comparison.core import REPO_ROOT, eta_to_signal_strength
from ..random_minimal.analyze import load_results, summary_cells
from .run import DATA_DIR, ETAS, SAMPLE_LENGTH

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


# (true label, oracle answer) -> how many to show: wrong on one of four in each
# class, near the oracle's real error rate, though the classes are balanced
EXAMPLE_COUNTS = {(0, 0): 3, (0, 1): 1, (1, 0): 1, (1, 1): 3}


def example_rows(eta: float, seed: int) -> List[Tuple[str, int, int]]:
    """The first uniform draws filling `EXAMPLE_COUNTS`, read by the oracle
    `learn_dfa` builds for `seed`."""
    dfa = luhn_dfa()
    oracle = NoisyOracle(
        DFAOracle(dfa),
        SymmetricBernoulli(p_correct=0.5 + eta_to_signal_strength(eta)),
        seed,
    )
    rng = np.random.default_rng(0)
    found: Dict[Tuple[int, int], List[str]] = {k: [] for k in EXAMPLE_COUNTS}
    while any(len(found[k]) < n for k, n in EXAMPLE_COUNTS.items()):
        digits = [int(d) for d in rng.integers(0, 10, size=SAMPLE_LENGTH)]
        key = (
            int(dfa.accepts_input(digits)),
            int(oracle.membership_query(bytes(digits))),
        )
        if len(found[key]) < EXAMPLE_COUNTS[key]:
            found[key].append("".join(map(str, digits)))
    return [(x, *key) for key, xs in found.items() for x in xs]


def examples(rows: Sequence[Tuple[str, int, int]]) -> str:
    """Uses the paper's `\\luhnlabel{color}{bit}`."""

    def label(bit: int) -> str:
        return r"\luhnlabel{agree}{1}" if bit else r"\luhnlabel{disagree}{0}"

    return "\n".join(
        [r"\begin{tabular}{lcc}", r"    $x$ & $A(x)$ & $O(x)$ \\", r"    \midrule"]
        + [
            rf"    \texttt{{{x}}} & {label(truth)} & {label(answer)} \\"
            for x, truth, answer in rows
        ]
        + [r"\end{tabular}"]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = OUT_DIR / "summary.tex"
    tex_path.write_text(summary(load_results(DATA_DIR)) + "\n")
    print(f"wrote {tex_path}")
    (eta,) = ETAS
    examples_path = OUT_DIR / "examples.tex"
    examples_path.write_text(examples(example_rows(eta, seed=0)) + "\n")
    print(f"wrote {examples_path}")


if __name__ == "__main__":
    main()
