#!/usr/bin/env python3
"""Experiment 1: both learners on CAPAL's own benchmark suite.

Runs CAPAL and E-L* over the 28 `.taf` targets shipped in the upstream
`dataset/` directory (Simple/Normal/Difficult), at each noise level. This is
the only experiment on *CAPAL's* home turf rather than ours, so it is the
fairest read on the two learners' relative strengths.

Example:
    python -m orthogonal_dfa.experiments.capal_comparison.run_capal_bench \
        --etas 0.05 0.20 --targets Simple01 Normal01
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .core import REPO_ROOT
from .sweep import add_common_args, run_sweep
from .targets import capal_benchmarks

DEFAULT_OUT = REPO_ROOT / "data" / "capal" / "capal_benchmarks.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(ap)
    args = ap.parse_args()

    if args.capal_dir:
        os.environ["ORTHO_CAPAL_DIR"] = args.capal_dir

    try:
        benchmarks = capal_benchmarks(args.targets)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None

    run_sweep(
        benchmarks,
        experiment="capal_benchmarks",
        description=(
            "CAPAL and E-L* on CAPAL's own 28 .taf benchmark targets "
            "(Simple/Normal/Difficult) under persistent noise."
        ),
        generated_by="orthogonal_dfa.experiments.capal_comparison.run_capal_bench",
        out_path=Path(args.out) if args.out else DEFAULT_OUT,
        etas=args.etas,
        seeds=args.seeds,
        learners=args.learners,
    )


if __name__ == "__main__":
    main()
