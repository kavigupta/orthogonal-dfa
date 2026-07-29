#!/usr/bin/env python3
"""Experiment 1: both learners on CAPAL's own benchmark suite.

Runs CAPAL and E-L* over the 28 `.taf` targets shipped in the upstream
`dataset/` directory (Simple/Normal/Difficult), at each noise level -- the one
experiment on CAPAL's home turf rather than ours.

    python -m orthogonal_dfa.experiments.capal_comparison.run_capal_bench
"""

from __future__ import annotations

from .capal_targets import capal_benchmarks
from .sweep import run_sweep


def main() -> None:
    run_sweep(
        capal_benchmarks(),
        experiment="capal_benchmarks",
        description=(
            "CAPAL and E-L* on CAPAL's own 28 .taf benchmark targets "
            "(Simple/Normal/Difficult) under persistent noise."
        ),
        generated_by="orthogonal_dfa.experiments.capal_comparison.run_capal_bench",
    )


if __name__ == "__main__":
    main()
