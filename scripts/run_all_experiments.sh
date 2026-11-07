#!/usr/bin/env bash
# Usage: scripts/run_all_experiments.sh PROCESSES [GPU...]
# GPUs are needed only if a neural output is missing.
set -euo pipefail
cd "$(dirname "$0")/.."
PROCESSES=$1
shift
E=orthogonal_dfa.experiments

scripts/capal_run_all.sh

python -m $E.random_minimal.run --processes "$PROCESSES"
python -m $E.random_minimal.capal_random --processes "$PROCESSES"
python -m $E.random_minimal.analyze
python -m $E.random_minimal.capal_random_analyze

python -m $E.luhn.run --processes "$PROCESSES"
python -m $E.luhn.analyze
python -m $E.draw_luhn

python -m $E.length_oracles.reproduce scratchpad/length_oracle_runs --gpus "$@"
python -m $E.length_oracles.figure
