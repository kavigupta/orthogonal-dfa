"""Round 0 of synthesis with 4 restarts on the flat oracle over length-40 ACGT strings.

    python -u -m orthogonal_dfa.experiments.raw_acgt.run --seed S

Writes data/neural/raw_acgt/runs/sS/restarts.json as each restart ends, then run.json.
"""

import argparse
import json
import os
import time

import numpy as np

from orthogonal_dfa.l_star import counterexample_synthesis as cs
from orthogonal_dfa.l_star.learn import NUM_PREFIXES
from orthogonal_dfa.l_star.prefix_suffix_tracker import (
    PrefixSuffixTracker,
    SearchConfig,
)
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.statistics import compute_suffix_size_counterexample_gen

from ..length_oracles import run as length_run
from .common import LENGTH, RUNS, load_flat


def search_config():
    """``build_pst``'s config at signal 0.05."""
    return SearchConfig(
        suffix_size_counterexample_gen=compute_suffix_size_counterexample_gen(
            0.01, 0.55
        ),
        min_signal_strength=0.05,
        num_addtl_prefixes=NUM_PREFIXES,
        min_suffix_frequency=0.02,
        require_accept_preserving=True,
        cross_limit=1.5e-7,
        acceptable_fnr=0.01,
        max_coverage_error=1 / 3,
    )


def synthesize(oracle, seed, out):
    """The restarts of one round-0 run against ``oracle``, recorded in ``out``."""
    # build_pst seeds the learner with 0 whatever its seed, which a deterministic
    # oracle turns into one run for every seed.
    pst = PrefixSuffixTracker.create(
        UniformSampler(LENGTH),
        np.random.default_rng(seed),
        oracle,
        search_config(),
        num_prefixes=NUM_PREFIXES,
    )
    length_run.configure(pst)
    tracker = length_run.RestartRecorder(out)
    cs.counterexample_driven_synthesis(
        pst,
        acc_threshold=0.98,
        tracker=tracker,
        max_rounds=1,
        indecisive_fraction=0.1,
        min_indecisive=200,
    )
    return tracker.restarts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    out = RUNS / f"s{a.seed}"
    os.makedirs(out, exist_ok=True)
    started = time.time()
    restarts = synthesize(load_flat(), a.seed, out)
    status = length_run.git("status", "--porcelain", "--untracked-files=no")
    with open(out / "run.json", "w") as f:
        json.dump(
            dict(
                oracle="flat-raw",
                length=LENGTH,
                seed=a.seed,
                commit=length_run.git("rev-parse", "--short=8", "HEAD"),
                dirty=None if status is None else bool(status),
                hours=(time.time() - started) / 3600,
                restarts=len(restarts),
                family_size=restarts[0]["family_size"] if restarts else None,
                family_fnr=restarts[0]["family_fnr"] if restarts else None,
            ),
            f,
            indent=1,
        )


if __name__ == "__main__":
    main()
