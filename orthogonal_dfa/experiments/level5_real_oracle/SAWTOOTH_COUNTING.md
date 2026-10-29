# Wildcard counting on the old oracle

The current learner (PR #502) run against the old oracle learns DFAs that count
wildcards modulo 2 or 3.  This is a real regularity of that oracle, not a learner
fault.

## Setup

The old oracle is the harness default `gate_residual_oracle` (length 40, one
threshold calibrated per 5-base length bin, strings of 35-85 bases).  Symbols 0/1/2
are the 3-base stop codons TAG/TAA/TGA; symbols 3 and 4 are wildcards of 1 base.

Within a 5-base bin the single threshold leaves a sawtooth in acceptance by compiled
length.  A 36-symbol string compiles to 36 + 2 * (number of stops) bases, so the
wildcard count modulo k tracks compiled length, and a counter DFA predicts the
sawtooth.  `PerLengthOracle` (each length thresholded at its own median) was
introduced to remove this.

Runs: commit 3ed7f5a7, seeds 0-7, about 55 h per seed for 3 rounds:

    PYTHONPATH=. python -u orthogonal_dfa/experiments/level5_real_oracle/harvest_diag.py \
        --plateau --max-rounds 3 --seed $s --out DIR

## Oracle accept rate by compiled length

Measured on the held-out halves of all 8 seeds.

| compiled length | 36 | 38 | 40 | 42 | 44 | 46 | 48 |
|---|---|---|---|---|---|---|---|
| accept rate | .54 | .75 | .51 | .64 | .56 | .57 | .55 |

## Per-seed round DFAs

Each round DFA read on the seed's held-out half.  NMI is between the DFA's end state
and (a) the wildcard count modulo the best k in 2..7, (b) the frame pattern (the set of base
positions mod 3 at which a stop codon starts).  Stop self-loops is the fraction
of (state, stop symbol) pairs that stay in the same state.

| seed | round | phi | states | NMI(end, wildcards mod k) | NMI(end, frame) | stop self-loops |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.176 | 11 | 0.82 (k=3) | 0.37 | 0.97 |
| 0 | 1 | 0.181 | 27 | 0.68 (k=3) | 0.46 | 0.79 |
| 0 | 2 (best) | 0.206 | 10 | 0.48 (k=3) | 0.71 | 0.37 |
| 1 | 0 | 0.188 | 6 | 1.00 (k=2) | 0.26 | 1.00 |
| 1 | 1 | 0.169 | 3 | 1.00 (k=3) | 0.33 | 1.00 |
| 1 | 2 (best) | 0.186 | 9 | 0.79 (k=3) | 0.32 | 0.78 |
| 2 | 0 (best) | 0.234 | 7 | 0.52 (k=7) | 0.71 | 0.57 |
| 2 | 1 | 0.185 | 3 | 1.00 (k=2) | 0.26 | 1.00 |
| 2 | 2 | 0.174 | 6 | 0.81 (k=3) | 0.38 | 0.94 |
| 3 | 0 (best) | 0.234 | 9 | 0.52 (k=7) | 0.71 | 0.56 |
| 3 | 1 | 0.155 | 8 | 0.74 (k=2) | 0.21 | 0.92 |
| 3 | 2 | 0.192 | 16 | 0.57 (k=7) | 0.76 | 0.92 |
| 4 | 0 (best) | 0.162 | 8 | 1.00 (k=3) | 0.43 | 0.92 |
| 4 | 1 | 0.118 | 7 | 0.41 (k=7) | 0.26 | 0.76 |
| 4 | 2 | 0.157 | 8 | 0.58 (k=7) | 0.67 | 0.62 |
| 5 | 0 (best) | 0.222 | 13 | 0.52 (k=7) | 0.71 | 0.56 |
| 5 | 1 | 0.174 | 10 | 0.31 (k=7) | 0.38 | 0.30 |
| 5 | 2 | 0.189 | 11 | 0.76 (k=4) | 0.34 | 0.85 |
| 6 | 0 | 0.179 | 6 | 1.00 (k=2) | 0.26 | 1.00 |
| 6 | 1 | 0.175 | 6 | 0.81 (k=3) | 0.38 | 0.94 |
| 6 | 2 (best) | 0.212 | 26 | 0.48 (k=7) | 0.63 | 0.82 |
| 7 | 0 | 0.215 | 9 | 0.88 (k=3) | 0.61 | 0.78 |
| 7 | 1 (best) | 0.211 | 25 | 0.63 (k=3) | 0.53 | 0.80 |
| 7 | 2 | 0.187 | 24 | 0.46 (k=7) | 0.67 | 0.26 |

Best phi per seed (summary.json): s0 .206, s1 .186, s2 .234, s3 .234, s4 .162,
s5 .222, s6 .212, s7 .211.

## Reading

Several round DFAs are pure counters: NMI 1.00 with wildcards mod 2 or 3, stop
self-loops 0.92-1.00, frame NMI 0.26-0.43, phi .16-.19 (s1 rounds 0-1, s2 round 1,
s4 round 0, s6 round 0).  The best picks at .21-.23 (s0, s2, s3, s5) instead mix frame
structure (frame NMI about .71) with a weaker count (NMI about .5).  The other
best picks are s1 (count-dominated, frame .32), s4 (a pure counter) and s6, s7 (frame
.63 and .53).  Where k=7
is the best fit with NMI near .5, 7 is the top of the searched range and the fit is
weak, so those DFAs are not modulus-7 counters.
