#!/usr/bin/env bash
# Usage: scripts/capal_run_all.sh PROCESSES
set -euo pipefail
cd "$(dirname "$0")/.."
M=orthogonal_dfa.experiments.capal_comparison
python -m $M.run_all --processes "$1"
# Reads E-L*'s spend from our_benchmarks.json, so must follow run_all.
python -m $M.run_wall_sweep --processes "$1" --matched-budget
python -m $M.run_wall_sweep --processes "$1"
python -m $M.figures
