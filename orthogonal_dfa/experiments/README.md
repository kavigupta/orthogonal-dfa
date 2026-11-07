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

Times are roughly what the saved runs took, before dividing by `PROCESSES` or GPUs. They
leave out setup, such as fitting the neural oracles, and drawing the tables and figures.

| Paper | Results | Tables and figures | Time |
|---|---|---|---|
| Comparison to CAPAL | `data/capal/{capal_benchmarks,our_benchmarks}.json` | `figures/capal-comparison/{benchmark_grid,query_cost,success_rates}.tex` | ~1 h |
| CAPAL hyperparameter sweep | `data/capal/wall_sweep.json` | `figures/capal-comparison/hyperparameter_sweep.tex` | ~20 h |
| CAPAL with matched query budget | `data/capal/matched_budget.json` | `figures/capal-comparison/matched_budget.tex` | ~1 h |
| Random suite | `data/random_minimal/{targets,results}/` | `figures/random-minimal/{summary.tex,queries_vs_*.pdf}` | ~30 h |
| Random suite difficulty check, CAPAL | `data/random_minimal/capal_results/` | `figures/random-minimal/capal_successes.tex` | ~40 h |
| Luhn | `data/luhn/results/` | `figures/luhn/{summary.tex,examples.tex,luhn_dfa.pdf}` | ~15 min |
| Neural | `data/neural/length_oracles/` | `figures/length-oracles/` | ~50 GPU-h |

All other paper figures are illustrations not arising from experiments.

## Rerunning experiments

Run these in bash from the repository root, then rerun the experiment command from above.

```bash
# all results
rm -rf data/capal data/random_minimal data/luhn data/neural
```

To rerun part of an experiment, delete just those results, for example:

```bash
# E-L* in the comparison to CAPAL; the matched budget reruns on its own when E-L*'s spend changes
python -m orthogonal_dfa.experiments.capal_comparison.delete_elstar

# random suite at eta = 0.3, |Q| = 12
rm data/random_minimal/results/n12_*_eta0.30_seed0.json

# Luhn
rm -r data/luhn/results
```

The neural results depend on each other, so delete them together:

```bash
rm -rf data/neural
```
