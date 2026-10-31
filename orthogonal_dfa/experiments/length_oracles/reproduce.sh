#!/usr/bin/env bash
# Every run and the JSON, from source:
#
#   orthogonal_dfa/experiments/length_oracles/reproduce.sh OUT [GPU...]
#
# Runs go round robin over the GPUs (default 0).  A run with an OUT/<run>/run.json is
# not rerun.
set -euo pipefail
OUT=$(realpath -m "$1")
shift
GPUS=("${@:-0}")
PYTHON=${PYTHON:-python}
M=orthogonal_dfa.experiments.length_oracles

# The oracle fits are cached by name and arguments only, so a cache from older code
# would serve an older fit.
export XDG_CACHE_HOME="$OUT/cache"
export PYTHONPATH=.
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
mkdir -p "$OUT"

# Also fits both oracles into the cache before the runs read it.
[ -f "$OUT/labels.npz" ] ||
    CUDA_VISIBLE_DEVICES=${GPUS[0]} $PYTHON -u -m $M.label "$OUT/labels.npz" \
        > "$OUT/label.log" 2>&1

runs=() pids=() i=0
for oracle in saw flat; do
    for seed in 0 1 2 3; do
        run="$OUT/${oracle}_s$seed"
        runs+=("$run")
        [ -f "$run/run.json" ] && continue
        mkdir -p "$run"
        # Starting every run at once failed CUDA initialisation.
        [ ${#pids[@]} -eq 0 ] || sleep 240
        CUDA_VISIBLE_DEVICES=${GPUS[$((i % ${#GPUS[@]}))]} $PYTHON -u -m $M.run \
            --oracle $oracle --seed $seed --out "$run" > "$run/log" 2>&1 &
        pids+=($!)
        i=$((i + 1))
    done
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ $failed -eq 0 ] || { echo "a run failed; see $OUT/*/log" >&2; exit 1; }

$PYTHON -m $M.extract "$OUT/labels.npz" "$OUT/length_oracles_data.json" "${runs[@]}"
CUDA_VISIBLE_DEVICES=${GPUS[0]} $PYTHON -u -m $M.random_strings "$OUT/length_oracles_data.json"
echo "$OUT/length_oracles_data.json"
