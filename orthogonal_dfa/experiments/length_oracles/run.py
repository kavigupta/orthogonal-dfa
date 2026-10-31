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
from orthogonal_dfa.l_star import counterexample_synthesis as cs
from orthogonal_dfa.l_star.dfa_utils import dfa_table, held_out_phi
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


class RestartRecorder(SynthesisTracker):
    """Writes ``out``/restarts.json as each restart ends."""

    def __init__(self, out):
        self.out, self.restarts, self.family = out, [], None

    def on_family_resolved(self, suffixes, boundary, fnr, round_index):
        self.family = dict(family_size=len(suffixes), family_fnr=fnr)

    # pylint: disable=too-many-positional-arguments
    def on_restart_ended(self, restart, search, ending, dfa, pick_score, round_index):
        self.restarts.append(
            dict(
                restart=restart,
                pick_score=pick_score,
                search_score=search.score_patience.best_score,
                ending=ending.kind,
                probes=search.probes,
                **self.family,
                dfa=dfa_table(dfa),
            )
        )
        with open(f"{self.out}/restarts.json", "w") as f:
            json.dump(self.restarts, f, indent=1)


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
    tracker = RestartRecorder(a.out)
    best = cs.counterexample_driven_synthesis(
        pst,
        acc_threshold=0.98,
        tracker=tracker,
        max_rounds=1,
        indecisive_fraction=0.1,
        min_indecisive=200,
    )
    restarts = tracker.restarts
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
