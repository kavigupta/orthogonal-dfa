"""Fit strings (rng 777) set each product state's median centred score for the residual
oracles; eval strings (rng 12345) score everything, accepting states fit on the first
half and phi read on the second."""

import json
from collections import Counter

import numpy as np
from permacache import permacache

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.dfa_utils import phi_maximizing_states, table_end_states
from orthogonal_dfa.l_star.structures import Oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai

from ..capal_comparison.core import REPO_ROOT
from ..length_oracles.oracles import oracle
from .frames import closed_frames

DATA = REPO_ROOT / "data" / "neural" / "raw_acgt"
RUNS = DATA / "runs"
RESIDUAL = DATA / "residual"
SEEDS = range(4)
LENGTH = 40
N_FIT, N_EVAL = 100_000, 40_000
H = N_EVAL // 2


def uniform(seed, n):
    return np.random.default_rng(seed).integers(0, 4, size=(n, LENGTH)).astype(np.uint8)


FIT, EVAL = uniform(777, N_FIT), uniform(12345, N_EVAL)


def load_flat():
    return oracle("flat", default_exon, load_spliceai(400, 0))


def centred(flat, queries):
    queries = list(queries)
    return flat.scores(queries) - flat.thresholds_of(queries)


@permacache("orthogonal_dfa/experiments/raw_acgt/common/saved_centred")
def saved_centred(name):
    """``centred`` of FIT or EVAL."""
    return centred(load_flat(), [r.tobytes() for r in dict(fit=FIT, eval=EVAL)[name]])


def kept_stage_one(seed):
    with open(RUNS / f"s{seed}" / "restarts.json") as f:
        return max(json.load(f), key=lambda r: r["pick_score"])


def end_states(table, queries):
    """Of a uint8 array, or of a list of strings of any lengths."""
    t = np.array(table["transitions"])
    if isinstance(queries, np.ndarray):
        return table_end_states(t, table["start"], queries)
    lengths = np.array([len(s) for s in queries])
    out = np.empty(len(queries), int)
    for length in np.unique(lengths):
        idx = np.nonzero(lengths == length)[0]
        rows = np.array([list(queries[i]) for i in idx], dtype=int)
        out[idx] = table_end_states(t, table["start"], rows.reshape(len(idx), length))
    return out


def offsets(table, c_fit):
    q = end_states(table, FIT)
    return np.array(
        [
            np.median(c_fit[q == s]) if (q == s).any() else 0.0
            for s in range(len(table["transitions"]))
        ]
    )


class ResidualRaw(Oracle):
    """Accepts iff a string's centred score beats the median of its state of ``table``."""

    def __init__(self, flat, table, c_fit):
        self._flat, self._table = flat, table
        self.offsets = offsets(table, c_fit)

    @property
    def alphabet_size(self):
        return 4

    def membership_queries(self, strings):
        strings = list(strings)
        if not strings:
            return np.array([], dtype=bool)
        return (
            centred(self._flat, strings)
            > self.offsets[end_states(self._table, strings)]
        )

    def membership_query(self, string):
        return bool(self.membership_queries([string])[0])

    def target_dfa(self):
        return None


def accepting(cells, y):
    counts, accepts = Counter(cells[:H].tolist()), Counter()
    for c, v in zip(cells[:H].tolist(), y[:H]):
        accepts[c] += int(v)
    return phi_maximizing_states(counts, accepts)


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else 0.0


def verdict(cells, y):
    return np.isin(cells, list(accepting(cells, y)))


def phi(cells, y):
    return corr(verdict(cells, y)[H:], y[H:])


FRAMES = ("f0 open", "f1 open", "f2 open", "f0|1 open")


def features(arr):
    c = closed_frames(arr)
    out = dict(zip(FRAMES, (~c[:, 0], ~c[:, 1], ~c[:, 2], ~c[:, 0] | ~c[:, 1])))
    out.update({f"#{b}": (arr == i).sum(1) for i, b in enumerate("ACGT")})
    for p in (0, 1, 2, 37, 38, 39):
        for i, b in enumerate("ACGT"):
            out[f"pos{p}={b}"] = arr[:, p] == i
    return out


def top_features(v, feats, k=5):
    scored = ((n, corr(v[H:], f[H:])) for n, f in feats.items())
    return sorted(scored, key=lambda x: -abs(x[1]))[:k]
