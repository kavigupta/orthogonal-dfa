"""Every run of the experiment and its data files that are not saved yet, from source.

    python -m orthogonal_dfa.experiments.length_oracles.reproduce OUT --gpus 0 1

Runs go round robin over the GPUs, and logs go to OUT.  GPUs are needed only if something
is missing."""

import argparse
import itertools
import os
import subprocess
import sys
import time

from .residual import RUNS, SEEDS

MODULE = "orthogonal_dfa.experiments.length_oracles"
#: Starting every run at once failed CUDA initialisation.
STAGGER_SECONDS = 240


def launch(step, gpu, log, *args):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    with open(log, "w") as f:
        return subprocess.Popen(
            [sys.executable, "-u", "-m", f"{MODULE}.{step}", *args],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )


def any_missing(oracles):
    return any(
        not (RUNS / f"{oracle}_s{seed}" / "run.json").exists()
        for oracle, seed in itertools.product(oracles, SEEDS)
    )


def run_all(out, gpus, oracles):
    procs = []
    for oracle, seed in itertools.product(oracles, SEEDS):
        run = RUNS / f"{oracle}_s{seed}"
        if (run / "run.json").exists():
            continue
        os.makedirs(run, exist_ok=True)
        if procs:
            time.sleep(STAGGER_SECONDS)
        procs.append(
            launch(
                "run",
                next(gpus),
                f"{out}/{run.name}.log",
                *("--oracle", oracle, "--seed", str(seed), "--out", str(run)),
            )
        )
    failed = [p.args for p in procs if p.wait() != 0]
    assert not failed, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--gpus", nargs="*", type=int, default=[])
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    def run_step(step):
        gpu = args.gpus[0] if args.gpus else ""
        assert launch(step, gpu, f"{args.out}/{step}.log").wait() == 0

    if any_missing(("saw", "flat", "residual")):
        run_step("oracles")
    run_step("label")
    gpus = itertools.cycle(args.gpus)
    run_all(args.out, gpus, ("saw", "flat"))
    if any_missing(("residual",)):
        # The residual runs read flat's first stage back from the runs just saved.
        run_step("residual")
        run_all(args.out, gpus, ("residual",))
    run_step("random_strings")
    run_step("residual_report")


if __name__ == "__main__":
    main()
