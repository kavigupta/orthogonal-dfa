"""CAPAL vs E-L* comparison experiments.

Two experiments, one JSON schema:

1. `run_capal_bench` -- both learners on CAPAL's own 28 `.taf` benchmarks.
2. `run_our_bench`   -- both learners on this repo's oracle benchmarks.

Each writes a self-contained JSON under `data/capal/` carrying config and one
record per (benchmark, learner, eta, seed) cell, so that analysis reads those
JSONs alone and never needs to re-run a learner. The JSONs are not checked in;
regenerate them with the drivers above.
"""

from .core import (
    LEARNER_CAPAL,
    LEARNER_ELSTAR,
    SCHEMA_VERSION,
    Cell,
    run_capal_cell,
    run_elstar_cell,
    write_experiment,
)

__all__ = [
    "LEARNER_CAPAL",
    "LEARNER_ELSTAR",
    "SCHEMA_VERSION",
    "Cell",
    "run_capal_cell",
    "run_elstar_cell",
    "write_experiment",
]
