# pylint: disable=wrong-import-position,wrong-import-order,protected-access,invalid-name,missing-function-docstring,import-error,consider-using-with
"""Round 0 with 4 restarts on the sawtooth oracle (`saw`: one threshold, the median at
compiled length 40) or the per-length one (`flat`: each compiled length 35-85 thresholded
at its own median).  Each counterexample search writes <out>/search_<i>.pkl: its DFA and
its hypothesis after every change.

  python -u orthogonal_dfa/experiments/level5_real_oracle/length_oracles.py \
      --oracle saw|flat --seed S --out OUT
"""

import os
import pickle
import runpy
import sys

import numpy as np

EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP)
import fullloop_mem

from orthogonal_dfa.l_star import counterexample_synthesis as cs
from orthogonal_dfa.l_star.examples.gate_composition_residual import (
    fit_gate_composition_residual,
    gate_residual_oracle,
)
from orthogonal_dfa.l_star.examples.spliceai_oracle import (
    SpliceModelOracle,
    median_threshold,
    run_over_middles,
)
from orthogonal_dfa.l_star.fallbacks.score_patience import snapshot
from orthogonal_dfa.spliceai.exon_score import SpliceAIExonScore

LEN_LO, LEN_HI = 35, 85


class PerLengthOracle(SpliceModelOracle):
    def __init__(self, exon, score_model, thresholds, **kw):
        super().__init__(exon, score_model, float("nan"), **kw)
        self._lo = min(thresholds)
        self._thresholds = np.array([thresholds[l] for l in sorted(thresholds)])

    def membership_queries(self, strings):
        scores = run_over_middles(
            self._score_model,
            self._flank_l,
            self._flank_r,
            strings,
            device=self._device,
            chunk=self._chunk,
        )
        idx = np.clip(
            np.array([len(s) for s in strings]) - self._lo, 0, len(self._thresholds) - 1
        )
        return scores > self._thresholds[idx]


def oracle(kind, exon, model, device=None):
    if kind == "saw":
        return gate_residual_oracle(
            exon, model, length=40, device=device, len_lo=LEN_LO, len_hi=LEN_HI
        )
    gscore = fit_gate_composition_residual(
        SpliceAIExonScore(model).eval(),
        exon,
        device=device,
        len_lo=LEN_LO,
        len_hi=LEN_HI,
    )
    thresholds = {
        l: median_threshold(gscore, exon, l, device=device)
        for l in range(LEN_LO, LEN_HI + 1)
    }
    return PerLengthOracle(exon, gscore, thresholds, device=device)


class LoggedSearch(cs.CounterexampleSearch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectory = []

    def step(self, x):
        version = self.hypothesis.version
        ending = super().step(x)
        if self.hypothesis.version != version:
            self.trajectory.append((self.probes, snapshot(self.hypothesis)))
        return ending


def main():
    i = sys.argv.index("--oracle")
    kind = sys.argv[i + 1]
    del sys.argv[i : i + 2]
    out = sys.argv[sys.argv.index("--out") + 1]
    fullloop_mem.gate_residual_oracle = lambda exon, model, *, device=None, **_: oracle(
        kind, exon, model, device
    )
    cs.CounterexampleSearch = LoggedSearch
    search_once, count = cs._counterexample_search, [0]

    def logged(pst, vs, **kwargs):
        search, ending = search_once(pst, vs, **kwargs)
        with open(f"{out}/search_{count[0]}.pkl", "wb") as f:
            pickle.dump(
                dict(
                    ending=ending.kind,
                    probes=search.probes,
                    best_score=getattr(search.score_patience, "best_score", None),
                    dfa=cs._hypothesis_dfa(pst, search.hypothesis),
                    trajectory=search.trajectory,
                ),
                f,
            )
        count[0] += 1
        return search, ending

    cs._counterexample_search = logged
    sys.argv += ["--plateau", "--restarts", "4", "--max-rounds", "1"]
    runpy.run_path(os.path.join(EXP, "harvest_diag.py"), run_name="__main__")


if __name__ == "__main__":
    main()
