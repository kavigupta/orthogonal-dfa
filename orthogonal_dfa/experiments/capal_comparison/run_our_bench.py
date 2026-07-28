#!/usr/bin/env python3
"""Experiment 2: both learners on this repo's benchmarks.

Replaces the old split where CAPAL's numbers came from one sweep and E-L*'s
from `scripts/count_queries.py`, with the head-to-head table assembled by hand
-- which is how the published comparison drifted out of date across three
refactors without anything noticing. Both learners now run in one process,
against one benchmark list, scored on one word list.

    python -m orthogonal_dfa.experiments.capal_comparison.run_our_bench
"""

from __future__ import annotations

from .our_targets import our_benchmarks
from .sweep import run_sweep


def main() -> None:
    run_sweep(
        our_benchmarks(),
        experiment="our_benchmarks",
        description=(
            "CAPAL and E-L* on this repo's oracle benchmarks (modulo-9 and the "
            "regex family from tests/test_lstar.py) under persistent noise."
        ),
        generated_by="orthogonal_dfa.experiments.capal_comparison.run_our_bench",
    )


if __name__ == "__main__":
    main()
