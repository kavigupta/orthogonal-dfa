"""Round 0 of synthesis with 4 restarts on one oracle.

    python -u -m orthogonal_dfa.experiments.length_oracles.run --oracle saw|flat|residual \
        --seed S --out OUT

Writes OUT/restarts.json as each restart ends, and OUT/run.json at the end."""

import argparse
import json
import os
import subprocess
import time

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star import counterexample_synthesis as cs
from orthogonal_dfa.l_star.dfa_utils import dfa_table
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.tracker import SynthesisTracker
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage.oracle import LiftedOracle
from orthogonal_dfa.superlanguage.sampler import SuperSampler

from .oracles import SYMBOLS, VOCAB, oracle
from .residual import ResidualOracle, first_stage

RESTARTS = 4


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
    config.counterexample_prepend_trie = False


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
    """None outside a git checkout, such as the artifact."""
    here = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=here, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True, choices=("saw", "flat", "residual"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    started = time.time()
    if a.oracle == "residual":
        flat, stage = oracle("flat", default_exon, load_spliceai(400, 0)), first_stage(
            a.seed
        )
        creator = lambda _noise, s: ResidualOracle(flat, stage, seed=s)
    else:
        base = oracle(a.oracle, default_exon, load_spliceai(400, 0))
        creator = lambda _noise, s: LiftedOracle(base, VOCAB, seed=s)
    pst = build_pst(
        creator,
        min_signal_strength=0.05,
        seed=a.seed,
        sampler=SuperSampler(VOCAB, SYMBOLS),
    )
    configure(pst)
    tracker = RestartRecorder(a.out)
    cs.counterexample_driven_synthesis(
        pst,
        acc_threshold=0.98,
        tracker=tracker,
        max_rounds=1,
        indecisive_fraction=0.1,
        min_indecisive=200,
    )
    restarts = tracker.restarts
    assert len(restarts) == RESTARTS, len(restarts)
    status = git("status", "--porcelain", "--untracked-files=no")
    with open(f"{a.out}/run.json", "w") as f:
        json.dump(
            dict(
                oracle=a.oracle,
                seed=a.seed,
                commit=git("rev-parse", "--short=8", "HEAD"),
                dirty=None if status is None else bool(status),
                hours=(time.time() - started) / 3600,
                family_size=restarts[0]["family_size"],
                family_fnr=restarts[0]["family_fnr"],
            ),
            f,
            indent=1,
        )


if __name__ == "__main__":
    main()
