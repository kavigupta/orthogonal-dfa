# Experiments

This script runs every experiment in the paper, and produces the figures and tables.

    scripts/run_all_experiments.sh PROCESSES [GPU...]

This is idepmotent, so you can run it multiple times and it will only rerun experiments with
missing results. To re-run an experiment, delete its results (see below).

## Prerequisites

    pip install -r requirements.txt
    sudo apt install graphviz
    # for the CAPAL experiments, clone CAPAL at the commit pinned in `orthogonal_dfa/capal_official/adapter.py`:
    cd ..; git clone https://github.com/lkwargs/CAPAL capal; cd capal; git checkout 57d877f6a083d58852660fac388ff49c052dc2d2; cd -

Rerunning the neural experiments needs a GPU.


## What produces what

Times are the saved runs' totals, before dividing by `PROCESSES` or GPUs.

| Paper | Results | Tables and figures | Time |
|---|---|---|---|
| Comparison to CAPAL | `data/capal/{capal_benchmarks,our_benchmarks}.json` | `figures/capal-comparison/{benchmark_grid,query_cost,success_rates}.tex` | 1.5 h |
| CAPAL hyperparameter sweep | `data/capal/wall_sweep.json` | `figures/capal-comparison/hyperparameter_sweep.tex` | 21 h |
| CAPAL with matched query budget | `data/capal/matched_budget.json` | `figures/capal-comparison/matched_budget.tex` | 1 h |
| Random suite | `data/random_minimal/{targets,results}/` | `figures/random-minimal/{summary.tex,queries_vs_*.pdf}` | 28 h |
| Random suite difficulty check, CAPAL | `data/random_minimal/capal_results/` | `figures/random-minimal/capal_successes.tex` | 43 h |
| Luhn | `data/luhn/results/` | `figures/luhn/{summary.tex,examples.tex,luhn_dfa.pdf}` | 15 min |
| Neural | `data/neural/length_oracles/` | `figures/length-oracles/` | 51 GPU-h |

All other paper figures are illustrations not arising from experiments.

## Rerunning experiments

### CAPAL comparison

The matched budget is the queries E-L* spent in `our_benchmarks.json`, so its cells rerun
on their own when that spend changes.

```bash
# everything
scripts/capal_delete_results.sh

# CAPAL's benchmarks, both learners
rm data/capal/capal_benchmarks.json

# our benchmarks, both learners
rm data/capal/our_benchmarks.json

# hyperparameter sweep
rm data/capal/wall_sweep.json

# matched budget
rm data/capal/matched_budget.json

# only E-L*'s cells (needs jq), after changing E-L* or its preconditions
for f in data/capal/capal_benchmarks.json data/capal/our_benchmarks.json; do
    jq '.cells |= map(select(.learner != "E-L*"))' "$f" > "$f.new"
    mv "$f.new" "$f"
done
```

### Random suite

```bash
# E-L*
rm -r data/random_minimal/results

# E-L* at one noise level
rm data/random_minimal/results/*_eta0.30_seed0.json

# E-L* on one size of target
rm data/random_minimal/results/n18_*

# CAPAL
rm -r data/random_minimal/capal_results

# new targets, after changing the sampler: both learners' results are on the old ones
rm -r data/random_minimal/targets data/random_minimal/results data/random_minimal/capal_results
```

### Luhn

```bash
# every seed
rm -r data/luhn/results

# one seed
rm data/luhn/results/luhn_len16_eta0.30_seed0.json
```

### Neural

The residual runs start from the flat runs' DFAs, and `residual_products.json` is computed
from the residual runs and the labels, so each command below also deletes what depends on
what it deletes.

```bash
D=data/neural/length_oracles

# all the runs
rm -rf $D/runs $D/residual_products.json

# the improperly calibrated runs
rm -rf $D/runs/saw_s*

# the correctly calibrated runs, and the residual runs after them
rm -rf $D/runs/flat_s* $D/runs/residual_s* $D/residual_products.json

# the residual runs
rm -rf $D/runs/residual_s* $D/residual_products.json

# the products of the residual runs
rm $D/residual_products.json

# the labels, and the products scored on them
rm $D/labels.npz $D/residual_products.json
rm -r ~/.cache/permacache/orthogonal_dfa/experiments/length_oracles/label

# accept rates on random strings
rm $D/random_strings.json
```

The neural steps also keep intermediate values in permacache, keyed on their arguments and
the model rather than on the code. After changing how the oracles are fit or how centred
scores are computed, also delete these, which takes the oracles' fits with them:

```bash
# every neural cache
rm -r ~/.cache/permacache/orthogonal_dfa/l_star/examples/gate_composition_residual \
    ~/.cache/permacache/orthogonal_dfa/l_star/examples/spliceai_oracle \
    ~/.cache/permacache/orthogonal_dfa/experiments/length_oracles
```
