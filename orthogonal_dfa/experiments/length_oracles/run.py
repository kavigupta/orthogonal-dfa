# pylint: disable=protected-access
"""Round 0 of synthesis with 4 restarts on one oracle.

    python -u -m orthogonal_dfa.experiments.length_oracles.run --oracle saw|flat \
        --seed S --out OUT

Writes OUT/restarts.json as each restart ends, and OUT/run.json at the end."""

import argparse
import json
import os
import subprocess
import time

import numpy as np

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star import cluster
from orthogonal_dfa.l_star import counterexample_synthesis as cs
from orthogonal_dfa.l_star.dfa_utils import held_out_phi
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.tracker import SynthesisTracker
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler

from .oracles import SYMBOLS, VOCAB, oracle

RESTARTS = 4
#: The test halves: strings drawn with these seeds, labelled with the run's lift seed.
TEST_SEEDS, TEST_HALF = (1111, 2222), 6000


def configure(pst):
    config = pst.config
    config.counterexample_member_fallback = True
    config.counterexample_skip_undecided = True
    config.counterexample_restarts = RESTARTS
    config.fnr_limit = 0.10
    config.fnr_plateau_patience = 3
    config.fnr_plateau_tol = 0.01
    config.counterexample_score_patience = 500
    config.counterexample_harvest_tests = False
    config.rank_rounds_by_held_out = True


def test_half(lifted, seed):
    rng = np.random.default_rng(seed)
    sampler = SuperSampler(VOCAB, SYMBOLS)
    s = [sampler.sample(rng, VOCAB.alphabet_size) for _ in range(TEST_HALF)]
    return s, np.asarray(lifted.membership_queries(s), dtype=float)


def dfa_table(dfa):
    states = sorted(dfa.states, key=str)
    index = {q: i for i, q in enumerate(states)}
    return dict(
        transitions=[
            [index[dfa.transitions[q][c]] for c in range(VOCAB.alphabet_size)]
            for q in states
        ],
        start=index[dfa.initial_state],
        accepting=[int(q in dfa.final_states) for q in states],
    )


def record(out):
    """Patches the synthesis to keep every family judged, the selection sample, and
    every restart."""
    judged, selections, restarts = [], [], []

    judge = cluster.judge_family

    def judge_family(*args):
        judged.append(judge(*args))
        return judged[-1]

    sample = cs._selection_sample

    def selection_sample(pst):
        selections.append(sample(pst))
        return selections[-1]

    search_once = cs._counterexample_search

    def counterexample_search(pst, vs, **kwargs):
        search, ending = search_once(pst, vs, **kwargs)
        dfa = cs._hypothesis_dfa(pst, search.hypothesis)
        family = [j for j in judged if list(j.vs) == list(vs)][-1]
        restarts.append(
            dict(
                restart=len(restarts),
                pick_score=held_out_phi(dfa, *selections[-1].pick),
                search_score=search.score_patience.best_score,
                ending=ending.kind,
                probes=search.probes,
                family_size=len(vs),
                family_fnr=family.fnr,
                dfa=dfa_table(dfa),
            )
        )
        with open(f"{out}/restarts.json", "w") as f:
            json.dump(restarts, f, indent=1)
        return search, ending

    cluster.judge_family = judge_family
    cs._selection_sample = selection_sample
    cs._counterexample_search = counterexample_search
    return restarts


def git(*args):
    here = os.path.dirname(os.path.abspath(__file__))
    return subprocess.check_output(["git", *args], text=True, cwd=here).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True, choices=("saw", "flat"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    started = time.time()
    base = oracle(a.oracle, default_exon, load_spliceai(400, 0))
    lifted = LiftedOracle(base, VOCAB, seed=a.seed)
    fit, read = (test_half(lifted, seed) for seed in TEST_SEEDS)
    pst = build_pst(
        lambda _noise, s: LiftedOracle(base, VOCAB, seed=s),
        min_signal_strength=0.05,
        seed=a.seed,
        sampler=SuperSampler(VOCAB, SYMBOLS),
    )
    configure(pst)
    restarts = record(a.out)
    best = cs.counterexample_driven_synthesis(
        pst,
        acc_threshold=0.98,
        tracker=SynthesisTracker(),
        max_rounds=1,
        indecisive_fraction=0.1,
        min_indecisive=200,
    )
    assert len(restarts) == RESTARTS, len(restarts)
    with open(f"{a.out}/run.json", "w") as f:
        json.dump(
            dict(
                oracle=a.oracle,
                seed=a.seed,
                commit=git("rev-parse", "--short=8", "HEAD"),
                dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
                hours=(time.time() - started) / 3600,
                family_size=restarts[0]["family_size"],
                family_fnr=restarts[0]["family_fnr"],
                test_halves_phi=held_out_phi(best.dfa, fit, read),
            ),
            f,
            indent=1,
        )


if __name__ == "__main__":
    main()
