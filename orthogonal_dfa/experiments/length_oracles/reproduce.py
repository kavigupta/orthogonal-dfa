"""Every run of the experiment and its data files, from source.

    python -m orthogonal_dfa.experiments.length_oracles.reproduce OUT --gpus 0 1

Runs go round robin over the GPUs; a run with an OUT/<run>/run.json is not rerun."""

import argparse
import itertools
import os
import subprocess
import sys
import time

from .extract import extract

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Also fits both oracles into the cache before the runs read it.
    assert launch("label", args.gpus[0], f"{args.out}/label.log").wait() == 0

    runs, procs = [], []
    gpus = itertools.cycle(args.gpus)
    for oracle, seed in itertools.product(("saw", "flat"), range(4)):
        run = f"{args.out}/{oracle}_s{seed}"
        runs.append(run)
        if os.path.exists(f"{run}/run.json"):
            continue
        os.makedirs(run, exist_ok=True)
        if procs:
            time.sleep(STAGGER_SECONDS)
        procs.append(
            launch(
                "run",
                next(gpus),
                f"{run}/log",
                *("--oracle", oracle, "--seed", str(seed), "--out", run),
            )
        )
    failed = [p.args for p in procs if p.wait() != 0]
    assert not failed, failed

    extract(runs)
    assert launch("random_strings", args.gpus[0], f"{args.out}/random.log").wait() == 0


if __name__ == "__main__":
    main()
