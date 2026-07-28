#!/usr/bin/env python3
"""Experiment 2: both learners on this repo's benchmarks.

Replaces the old split where CAPAL's numbers came from one sweep and E-L*'s
from `scripts/count_queries.py`, with the head-to-head table assembled by hand
-- which is how the published comparison drifted out of date across three
refactors without anything noticing. Both learners now run in one process,
against one benchmark list, scored on one word list.

Example:
    python -m orthogonal_dfa.experiments.capal_comparison.run_our_bench \
        --etas 0.05 0.30
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .core import REPO_ROOT
from .our_targets import our_benchmarks
from .sweep import add_common_args, run_sweep, select_targets

DEFAULT_OUT = REPO_ROOT / "data" / "capal" / "our_benchmarks.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(ap)
    args = ap.parse_args()

    benchmarks = select_targets(our_benchmarks(), args.targets)

    run_sweep(
        benchmarks,
        experiment="our_benchmarks",
        description=(
            "CAPAL and E-L* on this repo's oracle benchmarks (modulo-9 and the "
            "regex family from tests/test_lstar.py) under persistent noise."
        ),
        generated_by="orthogonal_dfa.experiments.capal_comparison.run_our_bench",
        out_path=Path(args.out) if args.out else DEFAULT_OUT,
        etas=args.etas,
        seeds=args.seeds,
        learners=args.learners,
    )


if __name__ == "__main__":
    main()
