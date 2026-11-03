"""The flat oracle with a first-stage DFA's effect taken out: a super-string is accepted
iff its centred score beats the median centred score of the random strings ending in
its first-stage state, where

    centred score = score - flat threshold[compiled length]

    python -u -m orthogonal_dfa.experiments.length_oracles.residual

fits the centred scores into the cache before the runs read them."""

import json

import numpy as np
from permacache import permacache

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.dfa_utils import table_end_states
from orthogonal_dfa.l_star.structures import Oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import _compilation_seed

from ..capal_comparison.core import REPO_ROOT
from . import label
from .oracles import VOCAB, oracle

RUNS = REPO_ROOT / "data" / "neural" / "length_oracles" / "runs.json"


def end_states(table, strings):
    """End state of each super-string, of any lengths."""
    transitions = np.array(table["transitions"])
    out = np.empty(len(strings), int)
    lengths = np.array([len(s) for s in strings])
    for length in np.unique(lengths):
        idx = np.nonzero(lengths == length)[0]
        rows = np.array([list(strings[i]) for i in idx], dtype=int)
        out[idx] = table_end_states(transitions, table["start"], rows)
    return out


def centred_scores(flat, strings, seed):
    """Of super-strings compiled as ``LiftedOracle(flat, VOCAB, seed=seed)`` compiles them."""
    out = []
    for i in range(0, len(strings), label.CHUNK):
        chunk = strings[i : i + label.CHUNK]
        rngs = [np.random.default_rng(_compilation_seed(s, seed)) for s in chunk]
        compiled = [bytes(c) for c in VOCAB.compile_many(chunk, rngs)]
        out.append(flat.scores(compiled) - flat.thresholds_of(compiled))
    return np.concatenate(out) if out else np.array([])


@permacache("orthogonal_dfa/experiments/length_oracles/residual/saved_centred_scores")
def saved_centred_scores(seed, n):
    """``centred_scores`` of ``label.strings(n)``."""
    flat = oracle("flat", default_exon, load_spliceai(400, 0))
    return centred_scores(flat, label.strings(n), seed)


def first_stage(seed):
    """Flat seed ``seed``'s kept DFA in runs.json."""
    with open(RUNS) as f:
        restarts = json.load(f)["restarts"]
    rows = [r for r in restarts if (r["oracle"], r["seed"]) == ("flat", seed)]
    return max(rows, key=lambda r: r["pick_score"])["dfa"]


def offsets(table):
    """Per state of ``table``, the median centred score over the first half of the
    saved strings; 0 for a state none of them reach."""
    c = saved_centred_scores(0, label.COUNT)[: label.COUNT // 2]
    q = end_states(table, label.strings(label.COUNT)[: label.COUNT // 2])
    return np.array(
        [
            np.median(c[q == s]) if (q == s).any() else 0.0
            for s in range(len(table["transitions"]))
        ]
    )


class ResidualOracle(Oracle):
    def __init__(self, flat, stage, *, seed):
        """``flat`` is a PerLengthOracle and ``stage`` a dfa_table dict."""
        self._flat, self._stage, self._seed = flat, stage, seed
        self.offsets = offsets(stage)

    @property
    def alphabet_size(self):
        return VOCAB.alphabet_size

    def membership_queries(self, strings):
        strings = list(strings)
        if not strings:
            return np.array([], dtype=bool)
        c = centred_scores(self._flat, strings, self._seed)
        return c > self.offsets[end_states(self._stage, strings)]

    def membership_query(self, string):
        return bool(self.membership_queries([string])[0])

    def target_dfa(self):
        return None


if __name__ == "__main__":
    saved_centred_scores(0, label.COUNT)
