"""CAPAL vs E-L* comparison experiments.

Three experiments, one JSON schema:

1. `run_capal_bench`  -- both learners on CAPAL's own 28 `.taf` benchmarks.
2. `run_our_bench`    -- both learners on this repo's oracle benchmarks.
3. `run_modulo_wall`  -- CAPAL config sweep on the modulo-9 eta=0.30 wall.

Each writes a self-contained JSON under `data/capal/` carrying provenance,
config, and one record per (benchmark, learner, eta, seed) cell. The report is
generated from those JSONs alone -- analysis must never need to re-run a
learner.
"""

from .core import (
    LEARNER_CAPAL,
    LEARNER_ELSTAR,
    SCHEMA_VERSION,
    Cell,
    eta_to_signal_strength,
    run_capal_cell,
    run_elstar_cell,
    write_experiment,
)

__all__ = [
    "LEARNER_CAPAL",
    "LEARNER_ELSTAR",
    "SCHEMA_VERSION",
    "Cell",
    "eta_to_signal_strength",
    "run_capal_cell",
    "run_elstar_cell",
    "write_experiment",
]
