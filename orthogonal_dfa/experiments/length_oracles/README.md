# Sawtooth vs per-length oracle

Round 0 of synthesis, 4 restarts, seeds 0-3, on two oracles built from the len40
gate-residual model (`len_lo=35`, `len_hi=85`) over 36-symbol strings of TAG, TAA, TGA and
two one-base wildcards:

- `saw` thresholds every string at the median for compiled length 40;
- `flat` thresholds each compiled length at its own median.

Everything else is shared.  A search ends only on success or once its held-out score has
not improved for 500 probes, and the restart kept is the best on a second held-out sample.

    orthogonal_dfa/experiments/length_oracles/reproduce.sh OUT 1 2 3 4 5 7

runs all 8 (2-7 h each) and writes `data/neural/length_oracles/length_oracles_data.json`
and `data/neural/length_oracles/random_strings.json`.  They hold counts, not rates, so they
can be re-aggregated:

- `oracle_counts`: both oracles' accepts on 200,000 fresh strings (`label.py`, which
  permacaches the labels), by half,
  number of stops and frame pattern;
- `runs`: each run's commit, hours, family size and FNR, and its kept DFA's phi on its own
  test halves;
- `restarts`: each restart's DFA, pick and search scores, ending and probes, and the
  strings ending in each of its states by half, stops and frame pattern, with their
  accepts;
- `random_strings.json`: both oracles' accepts on 20,000 uniform random strings at each
  length 35-85 (`random_strings.py`).
