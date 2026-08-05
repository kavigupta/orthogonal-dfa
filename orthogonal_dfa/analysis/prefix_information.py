"""How much of the oracle's accept decision the prefix carries -- measured directly,
independent of E-L*.

For an oracle over prefix+suffix strings, form the membership matrix
``M[i, j] = oracle accepts prefixes[i] + suffixes[j]``.  If the decision is a function
of the prefix, the rows are constant and the prefix explains all the variance; if accept
is independent of the prefix, the estimate falls to the ~1/n_suffixes sampling floor.
The prefix-explained fraction is thus a ceiling on what E-L* could ever recover as DFA
state -- the "is there anything to learn" question, separate from whether E-L* learns it.

The per-oracle measurement is permacached, so a notebook re-runs instantly on hot caches.
"""

import numpy as np
from permacache import permacache

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.composition_residual import fit_composition_residual
from orthogonal_dfa.l_star.examples.set_difference import SetDifferenceOracle
from orthogonal_dfa.l_star.examples.spliceai_oracle import (
    SpliceModelOracle,
    median_threshold,
)
from orthogonal_dfa.spliceai.exon_score import SpliceAIExonScore
from orthogonal_dfa.spliceai.load_model import load_fm, load_spliceai

# Oracle variants shown as bars, ordered by how much composition each strips out.
ORACLES = ["spliceai", "composition_residual", "spliceai_minus_fm"]
LABELS = {
    "spliceai": "SpliceAI",
    "composition_residual": "composition\nresidual",
    "spliceai_minus_fm": "SpliceAI \\ FM",
}


def prefix_explained_variance(matrix):
    """Fraction of the accept matrix's variance explained by the prefix (the row).

    One-way ANOVA with prefix as the factor: ``SS_between / SS_total``.  1.0 when every
    row is constant (accept is a function of the prefix), ~1/n_suffixes when accept is
    independent of the prefix (see :func:`independence_floor`)."""
    grand = matrix.mean()
    ss_total = ((matrix - grand) ** 2).sum()
    if ss_total == 0:
        return 0.0
    row_means = matrix.mean(axis=1)
    ss_between = matrix.shape[1] * ((row_means - grand) ** 2).sum()
    return float(ss_between / ss_total)


def independence_floor(matrix, seed=0):
    """``prefix_explained_variance`` under the null that accept is prefix-independent:
    permute each suffix column across prefixes (keeps per-suffix accept rates, destroys
    prefix structure).  Lands at ~1/n_suffixes -- the sampling floor to compare against.
    """
    rng = np.random.default_rng(seed)
    shuffled = np.stack([rng.permutation(col) for col in matrix.T], axis=1)
    return prefix_explained_variance(shuffled)


# SpliceAI and the FM trace are both permacache-hashable, so median_threshold and
# fit_composition_residual cache directly -- no .function / no_cache_global needed.
def _balanced(model, exon, length):
    """Oracle thresholding the exon score at its median over random length-``length``."""
    score_model = SpliceAIExonScore(model).eval()
    threshold = median_threshold(score_model, exon, length)
    return SpliceModelOracle(exon, score_model, threshold)


def _residual(model, exon, len_hi):
    """Oracle on the per-length composition residual, median-thresholded."""
    residual = fit_composition_residual(
        SpliceAIExonScore(model).eval(), exon, len_lo=90, len_hi=len_hi
    )
    threshold = median_threshold(residual, exon, exon.random_text_length)
    return SpliceModelOracle(exon, residual, threshold)


def _build_oracle(name, length):
    exon, calib = default_exon, 2 * length
    model = load_spliceai(400, 0)
    if name == "spliceai":
        return _balanced(model, exon, calib)
    if name == "composition_residual":
        return _residual(model, exon, calib + 5)
    if name == "spliceai_minus_fm":
        return SetDifferenceOracle(
            _balanced(model, exon, calib), _balanced(load_fm(1), exon, calib)
        )
    raise ValueError(f"unknown oracle {name!r}")


def _membership_matrix(oracle, prefixes, suffixes, prefix_batch=32):
    rows = []
    for i in range(0, len(prefixes), prefix_batch):
        batch = prefixes[i : i + prefix_batch]
        flat = oracle.membership_queries([p + s for p in batch for s in suffixes])
        rows.append(np.asarray(flat).reshape(len(batch), len(suffixes)))
    return np.concatenate(rows, axis=0)


@permacache("orthogonal_dfa/analysis/prefix_information/measure_v1")
def measure_prefix_information(
    oracle_name, *, length=95, n_prefixes=512, n_suffixes=256, seed=0
):
    """Prefix-explained variance (with its independence floor and accept rate) for
    ``oracle_name`` over random length-``length`` prefixes and suffixes."""
    oracle = _build_oracle(oracle_name, length)
    rng = np.random.default_rng(seed)
    prefixes = rng.integers(0, 4, (n_prefixes, length)).tolist()
    suffixes = rng.integers(0, 4, (n_suffixes, length)).tolist()
    matrix = _membership_matrix(oracle, prefixes, suffixes)
    return dict(
        prefix_explained=prefix_explained_variance(matrix),
        noise_floor=independence_floor(matrix),
        accept_rate=float(matrix.mean()),
    )


def prefix_information_bars(**kw):
    """``measure_prefix_information`` for every oracle in :data:`ORACLES` (each cached)."""
    return {name: measure_prefix_information(name, **kw) for name in ORACLES}


def plot_prefix_information(results, ax=None):
    """Bar chart of prefix-explained variance per oracle, with the noise floor marked."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    names = list(results)
    explained = [results[n]["prefix_explained"] for n in names]
    floor = float(np.mean([results[n]["noise_floor"] for n in names]))
    ax.bar([LABELS.get(n, n) for n in names], explained, color="#4c72b0")
    ax.axhline(floor, linestyle="--", color="0.4", linewidth=1)
    ax.text(
        len(names) - 0.5, floor, "noise floor", va="bottom", ha="right", color="0.4"
    )
    ax.set_ylabel("prefix-explained variance")
    ax.set_ylim(0, max(explained) * 1.15)
    return ax
