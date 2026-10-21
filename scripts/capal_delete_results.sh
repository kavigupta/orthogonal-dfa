#!/usr/bin/env bash
# The sweeps reuse any cell already in these files, so delete them after changing a learner.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -f data/capal/{capal_benchmarks,our_benchmarks,matched_budget,wall_sweep}.json
