# E-L\* on the SpliceAI neural oracle — findings & idea tracker

Running E-L\* (`orthogonal_dfa/l_star`, `do_counterexample_driven_synthesis`) against
the SpliceAI model as a batched membership oracle. Oracle = model's hard exon call on
a random variable-length middle wrapped in the default_exon flanks; membership thresholded
by the model's own calibration. Oracles live in
`orthogonal_dfa/l_star/examples/spliceai_oracles.py`, the driver in
`scripts/run_elstar_spliceai.py`; figures in `figures/`.

_Last updated: 2026-07-30._

## BIGGEST UPDATE (2026-07-30): the reachability wall is NOT counter-specific
Idea 3 (composition-residual oracle, counter cleanly removed: CpG corr→0, BoW R²=0.73)
STILL hits the same wall. E-L\* round 1 on the residual: **54 states, 32 accepting, only 1
reachable**, real agreement **0.5015 = base rate**, dfa_accept 0.000. So removing the counter
did NOT fix reachability — E-L\* discovers many states but cannot wire the initial state to
the accepting ones, even on a balanced, composition-free target. ⇒ The disconnection is a
**generic failure mode of E-L\* on these high-entropy neural oracles**, not something the
counter causes. (Rounds are also ~13h each here — very slow.) This is the central negative
result so far.

## REACHABILITY MECHANISM (2026-07-30) — diagnosed
The transition resolver (`l_star/transition_resolver.py`) sets `δ(s,c)` by a MAJORITY VOTE
over the state's length-95 POOL prefixes extended by one symbol (`_resolve`, lines 125-140:
classify `{p+[c]}` for p in the state's mask). On the residual round-1 DFA (54 states), the
initial state `[]` collapses into a giant **BULK state holding 49.8% of all pool prefixes**
(the other 53 are ≤2% satellites). For ~723 length-95 bulk prefixes, +1 symbol is a ~1%
perturbation, so the majority vote rounds `δ(bulk,c)→bulk` for every c → the initial state is
an **absorbing self-loop** (only 3.7% of ALL transitions self-loop, but the initial one's do).
Satellites are small enough that +1 symbol moves them, so they interconnect into a web — but
it's only enterable FROM a satellite, never from the bulk. ⇒ Reachability failure = a
length/granularity artifact of majority-vote transitions over long prefixes; counter-independent
(so it reproduces on the composition-free residual too).

**Fix direction:** resolve transitions where 1 symbol matters — short/prefix-closed access
strings instead of pool-majority-vote (classic L\* uses access strings; the `short_prefix_closure`
core exists but is drowned by the 500 long random prefixes in the vote).

**Re-rooting the residual DFA** gives best agreement 0.5365 (vs base 0.501) — small. I first
called this "near-noise"; that was WRONG (see below).

### CORRECTION — the residual carries a small but SIGNIFICANT, GENUINELY NON-COMPOSITIONAL signal
(Files: `residual_significance.py`, `residual_structure.py`, `control_composition.py`.)
- Significance (best re-root chosen on validation, measured on disjoint TEST, len-95 where the
  residual IS clean): agreement 0.5297 vs independence-baseline 0.4972 → z=6.5; 2×2 χ² **p=4e-16**,
  φ=0.081.
- Controlling for composition: logit `truth ~ CpG + stops + DFA_pred` on the clean len-95 residual
  (target CpG corr −0.007). Adding the DFA: **LR χ²=63, p=2e-15**, DFA coef +0.40 → the DFA signal
  SURVIVES partialling out composition ⇒ **real non-compositional structure, not leaked counter.**
- Prefix-variance CEILING (E-L\*-independent): residual prefix explains **30%** of acceptance
  variance at len 190 (raw counter 38%; noise floor 0.3%) — big headroom vs the φ≈0.08 captured.
  Caveat: contaminated at len 190 (below), so the CLEAN non-compositional ceiling is TBD but >0.
⇒ SpliceAI's exon call is DOMINATED by the composition counter, but there IS a small genuine
non-compositional finite-state-ish signal, and E-L\* only captures a sliver — room to push.

### Confirmed on a CLEAN target (per-length residual, 2026-07-31)
Re-ran E-L\* on the length-robust `PerLengthResidualOracle` (generic BoW, balanced across the
query-length range) at finer eps (mss 0.06 → eps 0.032). The **round-0** DFA (30 states),
re-rooted + composition-controlled on the clean len-95 target:
- agreement 0.549 vs independence 0.504 → z=9.1; 2×2 χ² p=6e-20; **φ=0.091**.
- composition-controlled LR (DFA beyond CpG+stops): χ²=70.8, **p=3.9e-17**, DFA coef +0.34.
So on a *clean* target at *round 0* the signal is slightly STRONGER than the contaminated
round-1 result (φ 0.081→0.091), confirming it's real and that cleaning + finer eps help.
Round 1 exploded to **404 states** (finer eps on a signal-rich clean target) but the run died
mid-enrichment before saving; the states are hub-and-spoke/starved (2 bulk states + ~400
satellites with 1–7 prefixes each), so pushing further needs more prefixes to avoid starvation.

### Residual-oracle LENGTH BUG (fix before pushing)
`CompositionResidualOracle` fits at `ref_len` with FREQUENCY features → composition removal only
holds at the fit length. len 95 (fit): CpG −0.015, accept 0.50 (clean). len 140/190: CpG +0.13/+0.14,
accept 0.65 (counter leaks back, unbalanced). E-L\* queries prefix+suffix (len 95–190) so TRAINING
was contaminated — but EVAL is at len 95 (clean) and the signal survives composition control, so
the finding holds. `LengthRobustResidualOracle` (count features across 95–190 + per-length median
threshold) is balanced but over/under-corrects at the extremes (CpG +0.2/−0.2); a per-length
composition model is the clean fix.

## TL;DR

The SpliceAI exon score is dominated by a **thresholded motif-count** (≈ `#CpG − #stop-codons`;
CpG/GC-rich → accept, AT-rich / TAG·TAA·TGA → reject). That is a **bounded counter**, which is
regular but needs Θ(L) connected states. E-L\* **discovers the counter's states** but **fails to
connect them to the start state**, so every learned DFA collapses to a trivial one (accept-nothing
or accept-everything) at the base-rate agreement (~0.5–0.62). The obstruction is **reachability**,
robust across every knob tried.

## What we've established (with evidence)

### The oracle is a thresholded motif-counter
- n-gram enrichment on the continuous exon score: CpG-rich k-mers drive **accept**
  (`CG` r=+0.53; `CGA/CCG/CGG…`), AT-rich / stop-codon k-mers drive **reject**
  (`TA` r=−0.44, `TAG` r=−0.42, `TAA`; #stop-codons r=−0.51).
- Linear model on `[GC, CpG, TA, #stops]` explains **R²≈0.46** of the score (CpG alone 0.28).
- Per-prefix accept-fraction is a clean monotone/sigmoid function of the `(CpG − stops)`
  count (corr +0.67). ⇒ Myhill–Nerode state ≈ running count. Bounded ⇒ finite (a counter),
  not a continuum. (Earlier "continuum / not finite-state" framing was **wrong**.)

### E-L\* collapses to a trivial DFA — and it's a REACHABILITY failure, not resolution
Tracking **real DFA-vs-oracle agreement** per round (not the internal DFA-vs-DT estimate,
which is misleading):

| run | L | mss / eps | prefixes | result |
|---|---|---|---|---|
| A | 95 | 0.15 / .078 | 200 | 2 states, trivial reject-all, agree 0.62 |
| B | 95 | 0.10 / .053 | 200 | 2 states, trivial, agree 0.62 |
| C | 95 | 0.05 / .025 | 500 | rounds 0→1→2 gave **2→5→12 DT states** but **0 reachable accepting**, agree pinned **0.6212** every round; DFA accepts nothing |
| C round 3 | 95 | 0.05 | 500 | did **not finish in 4 days** (runaway; new states starved: 1–22 prefixes each) |

- **mss / eps is NOT the lever** (refuted): even eps=0.025 (below the ~0.05 adjacent-count gap)
  still yields only accept/reject reachable. Higher eps → trivial 2-state; lower eps → many
  starved unreachable states. No eps threads it (a counter has no faithful *coarse* DFA).
- **Counterexample rounds DO eventually connect states — at short L** (corrects an earlier
  overclaim). L=24 recalibrated run: round 0 = 46 states / **1 reachable** / agree 0.5045;
  round 1 = **194 states / 141 reachable** / agree **0.5576**. So rounds bridge the initial
  state into the machine (1→141 reachable) and agreement rises above base rate — consistent
  with shorter access strings making the count-chain assemblable. Still hampered by degenerate
  labeling (only **1 accepting state** in 194), so it rejects little (dfa_accept 0.927). At
  L=95, 3 rounds did NOT connect anything — length-dependent, as the chain-length argument predicts.

### The disconnection is specifically at the START state (L=24 run, recalibrated, round 0)
Verified from the saved DFA (`saved_mss0.1_np800_len24/round_00.pkl`), see `figures/full_dfa_L24.*`:
- **46 states, only the initial state reachable.** Initial state is a self-looping sink
  (one symbol can't move the count off ≈0), so the whole DFA behaves as its single reachable
  state → agreement = base rate.
- The other 45 states contain a **27-state strongly-connected recurrent "counter core"**
  (period 1, dense ~3.3 intra-edges/state, NOT a clean ladder or a phase grid). Its states
  carry a **monotone accept-probability gradient 0.26→0.98**, and symbols push the right way
  (C +2.8, G +2.1 rungs up; T −2.6 down) — the count signal — but with high-variance
  (±6–13 rung) wiring: a **noisy finite-sample quotient of the true counter**, fully formed
  but stranded from the start.

### Method gotchas discovered
- **Short strings need recalibration.** The L=189 threshold makes short middles ~all-reject
  (accept 0.018 at L=24). Recalibrating the threshold to the median score at the sampler
  length restores accept≈0.50 (`SpliceModelOracle(calib_len=…)`, `run_elstar.py --recalibrate`).
- **Short strings did NOT make rounds faster** (predicted wrong): recalibrated L=24 has a
  severe FNR problem (sampled 61,308 suffixes, ~100× the family) because many length-24
  strings sit right at the threshold.
- Ragged-batch oracle: pad wrapped sequences to a common length; **gather the donor output at
  position `len(middle)+3`** (varies per row) — a fixed `[-1]` index is wrong.
- **GPU hygiene:** zombie runs quietly shared a GPU earlier (throughput 3.8 vs 14.5 it/s).
  Always `pgrep -af run_elstar` before trusting timings; pin `CUDA_VISIBLE_DEVICES`.
- `SearchConfig.fnr_limit=0.02` is too strict for a real (non-Bernoulli) oracle; use ~0.05.
- **Trust the DFA, not the estimate:** always check `dfa.accepts_input` / reachability, not
  the internal DFA-vs-DT accuracy (it rose 0.58→0.81 while real agreement stayed 0.62).

## Ideas to try (tracker)

### Idea 1 — Is the START state the uniquely broken thing? [DONE — partly yes]
Swept the start state over all 46 states of the saved L=24 round-0 DFA; real held-out
agreement (oracle accept 0.504):
- Actual initial state 0: **0.5038** (= base rate; accepts everything).
- **Best re-rooting = state 40** (a LOW-count core state): **0.6402** (accept 0.669).
  The whole top of the ranking is the low-count entry region (40, 44, 43, 45, 36 ≈ 0.63–0.64).
**Conclusion:** re-rooting the DFA at a low-count core state lifts agreement from base-rate
**0.50 → 0.64** — so (a) the start state as E-L\* wired it (self-looping accept-sink) IS the
specifically broken piece, and (b) the counter core computes something real. BUT 0.64 is only
modest: the core is a *noisy* quotient, so even optimally rooted it's a weak classifier, not a
faithful counter. Implication: fixing reachability alone (e.g. seeding a low-count access
string as the start) would recover ~0.64, not ~0.9 — the remaining gap is the core's noise.
Motivates ideas 2/3 (get a cleaner, less counter-dominated target).

### Idea 2 — Set-difference oracles: SpliceAI \ FM and FM \ SpliceAI [MEASURED — premise weakened]
FM = **fixed motifs** = the trained modular_splicing model `msp-273.665a3` (BothLSSIModels +
PSAMMotifModel over 82 RBNS motifs, 3P/5P excluded), in a SEPARATE repo:
`/mnt/md0/ExpeditionsCommon/spliceai/Canonical`. Load via
`modular_splicing.utils.io.load_model(".../model/msp-273.665a3_{seed}")` (renamed-unpickler,
imports fine in current env); `cl=400`, forward is API-compatible with `compute_exon_scores`,
so `SpliceModelOracle(exon, fm, calib_len=…)` works (must set calib_len — permacached
`calibrate` can't hash FM's weakref; fixed by skipping calibrate when calib_len given).

**Comparison (both balanced 50% at L=95, 12k random middles):** agree 0.649, score corr +0.46;
disagree 35% ⇒ each set-diff target ~17.5% accept. **The set-difference does NOT remove the
counter:** SpliceAI CpG +0.54/stops −0.50; FM CpG +0.26/−0.23 (tracks it at HALF strength);
`SpliceAI−FM` still CpG +0.39/−0.36. So SpliceAI is "more of a counter" than fixed-motifs, and
differencing amplifies rather than cancels composition. ⇒ Idea 2 would likely hit the same
counter/reachability wall; **Idea 3 removes the counter far more cleanly (BoW R²=0.73)**.
Decision pending: still run E-L\* on SpliceAI\FM (17.5% accept, rebalanced) or skip in favor of Idea 3.
Files: `scratchpad/fm_compare.py`. FM load recipe in memory/this doc.

### Round-0 results (all three idea-2/3 runs), 2026-07-30 [running, watching later rounds]
All start at the trivial DFA (= L\* round 0). Real agreement vs each oracle's base rate:
- **Idea 3 (comp. residual, accept 0.498):** round 0 = 5 states, 1 reachable, agree **0.5015** (=base,
  no signal yet); round 1 building **54 states**. THE one to watch — balanced, carving structure.
- **Idea 2 plain (SpliceAI\FM, accept 0.181):** round 0 = 2 states trivial, agree 0.8185 (=base 0.819).
  TERMINATED on its own at round 0 (total 20710s) — converged to trivial reject-all (degenerate,
  as the 18% imbalance predicted).
- **Idea 2 resid (SpliceAI\FM, accept 0.229):** round 0 = 2 states trivial, agree 0.7711 (=base 0.771).
- Idea-2 runs are unbalanced (18–23% accept → base-rate attractor 0.77–0.82) AND ~2× slower
  (two model calls/query). Round 0 wall-clock 2.7h (idea3) – 6.4h (idea2-resid).

### Idea 3 — Composition-residualized SpliceAI (backwards-logistic off n-gram BoW) [RUNNING]
Fit a logistic/linear model predicting the SpliceAI exon call from n-gram (n≤4) bag-of-words
composition features, then use the **residual** as the oracle. Rationale: this directly removes
the CpG/stop counter (which is exactly the composition signal, R²≈0.46), leaving whatever
*non-compositional*, possibly genuinely finite-state, structure remains. Repo reportedly has a
writeup/implementation of this backwards-logistic deconfounding — locate and reuse.
_Depends on finding the existing implementation._

## Key files (committed)
- Oracles: `orthogonal_dfa/l_star/examples/spliceai_oracles.py`
  - `SpliceModelOracle` (raw SpliceAI/FM call; `calib_len` recalibration)
  - `CompositionResidualOracle` / `PerLengthResidualOracle` (generic BoW composition residual;
    per-length is the length-robust one)
  - `SetDifferenceOracle` + `load_fm` (SpliceAI \\ FM, FM = fixed-motif `msp-273.665a3`)
- Driver: `scripts/run_elstar_spliceai.py` (writes per-round pickles to `runs/<config>/round_NN.pkl`)
- Saved state schema: dict with `prefixes, suffixes, masks, dt (suffix classifiers), dfa,
  decision_boundary, evidence_margin, config`
- Figures: `figures/full_dfa_L24.*` (counter DFA), `figures/residual_dfa_54state.*`,
  `figures/residual_dfa_reroot16_min.*`
- Analysis (re-root sweep, significance vs independence, composition-controlled LR test,
  variance decomposition, DFA rendering) was done with throwaway scripts; the methodology
  is described inline above and reproducible from the saved pickles.
