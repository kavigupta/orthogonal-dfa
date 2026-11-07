"""Every run of the experiment and its data files that are not saved yet, from source.

    python -m orthogonal_dfa.experiments.length_oracles.reproduce OUT --gpus 0 1

Runs go round robin over the GPUs.  A run in runs.json is not rerun; one with an
OUT/<run>/run.json only has that file extracted.  GPUs are needed only if something is
missing."""

import argparse
import itertools
import os
import subprocess
import sys
import time

from . import label, random_strings, residual_report
from .extract import extract, saved_runs

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


def pending_runs(oracles):
    saved = {(r["oracle"], r["seed"]) for r in saved_runs()["runs"]}
    return [p for p in itertools.product(oracles, range(4)) if p not in saved]


def run_all(out, gpus, oracles):
    runs, procs = [], []
    for oracle, seed in pending_runs(oracles):
        run = f"{out}/{oracle}_s{seed}"
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
    if runs:
        extract(runs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--gpus", nargs="*", type=int, default=[])
    args = parser.parse_args()

    first_stage = pending_runs(("saw", "flat"))
    second_stage = pending_runs(("residual",))
    steps = [
        step
        for step, path in (
            ("label", label.SAVED),
            ("random_strings", random_strings.OUT),
            ("residual_report", residual_report.PRODUCTS),
        )
        if not path.exists()
    ]
    if not (first_stage or second_stage or steps):
        print("Every neural output is saved; nothing to run.", flush=True)
        return
    if not args.gpus:
        raise SystemExit(
            f"--gpus is required: missing runs {first_stage + second_stage} "
            f"and steps {steps}"
        )
    os.makedirs(args.out, exist_ok=True)

    def run_step(step):
        log = f"{args.out}/{step}.log"
        assert launch(step, args.gpus[0], log).wait() == 0

    if first_stage or second_stage:
        run_step("oracles")
    if "label" in steps:
        run_step("label")
    gpus = itertools.cycle(args.gpus)
    run_all(args.out, gpus, ("saw", "flat"))
    if second_stage:
        # The residual runs read flat's first stage back from the runs.json just written.
        run_step("residual")
        run_all(args.out, gpus, ("residual",))
    for step in steps:
        if step != "label":
            run_step(step)


if __name__ == "__main__":
    main()
