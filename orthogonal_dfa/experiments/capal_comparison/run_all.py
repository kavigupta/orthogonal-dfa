#!/usr/bin/env python3
"""Run every CAPAL-comparison experiment, in order.

Each experiment reuses whatever its JSON already holds, so re-running this is
cheap when nothing has changed and resumes where an interrupted sweep stopped.
That reuse is keyed on cell identity, not on the code that produced the cell:
after changing either learner, delete `data/capal/` first.

    python -m orthogonal_dfa.experiments.capal_comparison.run_all
"""

from __future__ import annotations

import time

from . import run_capal_bench, run_our_bench

EXPERIMENTS = [run_capal_bench, run_our_bench]


def main() -> None:
    started = time.time()
    for i, experiment in enumerate(EXPERIMENTS, start=1):
        name = experiment.__name__.rsplit(".", 1)[-1]
        print(f"\n===== [{i}/{len(EXPERIMENTS)}] {name} =====", flush=True)
        t0 = time.time()
        experiment.main()
        print(f"===== {name} took {time.time() - t0:.0f}s =====", flush=True)
    print(f"\nAll {len(EXPERIMENTS)} experiments done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
