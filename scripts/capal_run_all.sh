#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
M=orthogonal_dfa.experiments.capal_comparison
python -m $M.run_all
# Reads E-L*'s spend from our_benchmarks.json, so must follow run_all.
python -m $M.run_wall_sweep --matched-budget
python -m $M.run_wall_sweep
python -m $M.figures
