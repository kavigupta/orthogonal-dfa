# Stop-codon superlanguage on the length-40 deconfounded oracle

An extension of the [len40 real-oracle experiment](../README.md).  That experiment
asks whether direct-L\* recovers a reading-frame signal from the
composition-deconfounded SpliceAI oracle, learning over the **raw ACGT** alphabet,
and finds it shatters or collapses to accept-all.

Here we change **only the alphabet**: instead of raw ACGT, the learner works over a
*superlanguage* whose symbols are the three stop codons `TAG`/`TAA`/`TGA` plus
interchangeable wildcards `X`, `Y`.  A wildcard compiles to a single base symbol
that cannot start a stop codon; a kmer symbol compiles to its codon; `compile` and
`parse` are inverses, so a wildcard never forges a stop.  The learned DFA is over
these super-symbols and is scored with **phi** (correlation) and **mutual
information**, never `est` (which rewards accept-all).

## TL;DR findings (seed 0)

1. **Why the strict gate never converges (Experiment 1).** The exact suffix-family
   FNR on this oracle is **0.085** for a pure-wildcard family and **0.065** for a
   kmer-bearing family — both far above the 0.02 limit.  ~6–9% of representative
   prefixes are intrinsically inside the ±eps decision band because the oracle's
   label flips across wildcard compilations, so *no* family drives the FNR to 0.02.
   (On the synthetic all-frames-closed oracle a wildcard tail is neutral, so the
   pure-wildcard family is decisive and its FNR is ~0 — the real oracle is
   different in kind, not degree.)

2. **What a relaxed-gate run recovers (Experiment 2).** With `fnr_limit=0.10` (just
   above the floor) the gate clears and synthesis produces a DFA.  Round 0 is an
   **8-state DFA**, accept-rate 0.78, **phi(DFA, oracle) = +0.167** (round 1 then
   drifts toward accept-all, phi +0.077).  The round-0 DFA collapses the three
   stop codons into one class and the two wildcards into another, and **rejects a
   string exactly when reading frames 0 *and* 1 are both closed** (deterministic,
   independent of frame 2).

3. **That predicate beats "all frames closed."**  `frames {0,1} both closed`
   correlates with the oracle at **phi = −0.158 / MI = 0.0191 bits**, versus
   **−0.088 / 0.0048 bits** for all-frames-closed — ~1.8× (phi) and ~4× (MI)
   stronger, and the strongest of every frame predicate tested (each single frame,
   every pair, all-three, the count).  The full 3-frame configuration holds 0.042
   bits, so the DFA's single bit captures ~45% of all frame-structure information.

So E-L\* over the stop-codon superlanguage recovers a small but **real, exact,
interpretable** signal — and the oracle's frame preference is not "all three
frames closed" (the #208 hypothesis) but specifically "frames 0 and 1 closed".

## Caveats

- Absolute signal is modest (phi ~0.16, MI ~0.02 bits) and the run is **fragile**:
  round 0 is the good one, later rounds collapse toward accept-all.
- `fnr_limit=0.10` is the knob run_len40.py deliberately holds at 0.02; relaxing it
  can manufacture shattering, so **only phi/MI claims are meaningful, never est**.
  The 8-state size and 0.78 accept-rate argue against pure shattering, but treat
  +0.167 as suggestive.
- Frames 0/1/2 are measured from the sampled middle's start; the `{0,1}` choice is
  the DFA's learned phase reference.

## Setup / dependencies

These scripts need, on one branch:
- this repo's `len40-real-oracle-experiment` base (the length-40 `gate_residual_oracle`,
  `synthesize_direct_lstar_fnr`, and `build_pst(..., fnr_limit=...)`);
- the `orthogonal_dfa/superlanguage/` package from PR #209 at commit `6eb234c`
  (`KmerVocabulary`/`SuperSampler`/`LiftedOracle`, invertible batched `compile_many`);
- one small addition: an optional `sampler=` parameter threaded through
  `build_pst`/`learn_dfa` (present on `main` via #209; port it onto this branch).

The deconfounded oracle's gate fit is `permacache`d; the first run fits it (a few
minutes on GPU).  If a cached entry was written under a different numpy and fails to
unpickle, point `XDG_CACHE_HOME` at a fresh dir to recompute it.

Design note — **no synthetic noise on the real oracle.**  `build_pst` would inject
`SymmetricBernoulli(0.5+min_signal_strength)` via its noise model; run_len40.py
avoids that with `lambda _nm,_s: oracle`.  These scripts do the same by passing
`noise_model=None` into `LiftedOracle` and ignoring `_nm` — `min_signal_strength`
only sizes the search; the compilation fiber is the sole stochasticity.

## Reproduce

```bash
# Experiment 1 — exact suffix-family FNR (a few minutes)
python experiments/len40_real_oracle/superlanguage/exp1_family_fnr.py
# -> pure-X/Y FNR = 0.0850, kmer-bearing FNR = 0.0650

# Experiment 2 — learn with the relaxed gate, dumping a DFA per round
python experiments/len40_real_oracle/superlanguage/exp2_learn.py \
    --fnr-limit 0.10 --dump-dir /tmp/sl_rounds
# -> round 0: 8 states, phi(DFA, oracle) = +0.167

# Analyze the dumps: per-round phi, round-0 DFA structure, frame-predicate phi/MI
python experiments/len40_real_oracle/superlanguage/analyze.py --dump-dir /tmp/sl_rounds
# -> DFA rejects iff frames {0,1} both closed; phi -0.158 / MI 0.019 bits
#    vs all-frames-closed -0.088 / 0.0048 bits
```

## Files

- `exp1_family_fnr.py` — exact `compute_fnr` for pure-wildcard vs kmer-bearing families.
- `exp2_learn.py` — probe-free learn run over the superlanguage, per-round dumps, final phi.
- `analyze.py` — per-round phi, best-round DFA transition table + reject-by-frame-set,
  and every frame predicate scored against the oracle by phi and MI.
