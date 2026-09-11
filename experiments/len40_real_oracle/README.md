# Length-40 direct-L* on the composition-deconfounded SpliceAI oracle

Self-contained harness for the question: **does direct-L* / FNR-refinement DFA
synthesis recover a genuine reading-frame signal (all-frames-closed / stop codons)
from a controlled neural oracle?** This directory has everything needed to reproduce
the run and its analysis; a fresh session can start here.

## TL;DR of the finding

**No clean recovery so far.** On the length-40 deconfounded oracle the FNR driver
either shatters into many states that correlate ~0 with the oracle, or collapses to a
trivial accept-all DFA -- and the est-based round selector *rewards* the accept-all
collapse (highest est), so the algorithm's chosen output is the trivial DFA. The
reading-frame signal is real and present in the oracle (see `probe_frame_signal.py`),
but the synthesized DFA does not capture it. This is a negative / cautionary result,
not a success.

## The metric caveat (read this first)

**Use phi (correlation), not est or accuracy-vs-oracle.**

- `est` = DFA-vs-discrimination-tree agreement (`estimate_agreement_rate`). It is an
  *internal consistency* number. A trivial accept-all DFA scores **high** est (its tree
  agrees with itself). est going up across rounds does NOT mean signal was recovered.
- `phi(DFA, oracle)` = correlation between DFA acceptance and (deconfounded) oracle
  acceptance. This is the recovery metric. A shattered DFA with `phi ~ 0` recovered
  nothing; an accept-all DFA has `DFA.std()==0` so `phi==0` by construction.
- `phi(DFA, all-frames-closed)` = does the DFA correlate with the *ground-truth* frame
  predicate (does every reading frame contain a stop codon)?

Every claim about "recovery" in this experiment must be a phi claim. This was a
repeated lesson: est-based / accuracy-based framings gave false positives.

## The oracle

`gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)`

SpliceAI's exon score with its base-composition component regressed out: exon score
minus a monotonic-gate bag-of-k-mers composition model, median-thresholded. The point
of deconfounding is that a raw composition signal (e.g. CpG content) is an easy,
non-frame explanation the learner would otherwise latch onto; removing it isolates
whatever *structural* signal (like reading frame) remains.

- `length=40`: the learner samples length-40 prefixes and queries prefix+suffix
  (~40-84 nt). The gate is fit on the band `[len_lo, len_hi) = [35, 85)` so it covers
  those query lengths.
- **Band caveat:** the version of `gate_composition_residual.py` on PR #194 asserts the
  exon's own query length (189) lies in the band, which breaks this narrow-band
  length-40 use. This directory ships the pre-assertion version that fits a band around
  the operating length. If you re-derive from #194, you must relax that assertion.
- At length 40 the frame signal is a *ripple* on a dominant aperiodic positionality;
  see the caveat in "Frame signal is real but positional" below.

## The knob that must not change

**`fnr_limit = 0.02`. DO NOT loosen it.** Loosening the FNR limit (e.g. to 0.35)
manufactures artificial shattering -- the family is allowed to be sloppy, so it splits
on noise -- and invalidates the whole experiment. `run_len40.py` asserts it is 0.02.
If a run is too slow, speed it up via length / fewer probes / leaner pool, never via
fnr_limit.

## Files

| file | what it does |
| --- | --- |
| `run_len40.py` | build the oracle + run `synthesize_direct_lstar_fnr`; `--dump-dir` dumps `round_NN.pkl` per round |
| `analyze_rounds.py` | for each round dump, report phi(DFA, oracle), phi(DFA, all-frames-closed), accept-rate by #frames-closed |
| `probe_frame_signal.py` | oracle-only: is the reading-frame signal even present? (period-3 `g(TAA,L)` vs `AAA` control) |

The per-round dump hook lives in `orthogonal_dfa/l_star/counterexample_synthesis.py`
(`_dump_round`, opt-in via the `DLSTAR_DUMP_DIR` env var -- dead code otherwise).

## Reproduce

```bash
# 1. is the frame signal present in the oracle at all? (must pass before anything else)
python experiments/len40_real_oracle/probe_frame_signal.py
#    expect: TAA period-3 spread ~0.15, AAA control ~0.01  (>10x)

# 2. run the synthesis, dumping each round
python experiments/len40_real_oracle/run_len40.py --max-rounds 10 \
    --dump-dir /path/to/dumps
#    "leaned" defaults keep peak RSS ~17 GB; the heavier original config OOM'd in round 2

# 3. measure real recovery per round (phi, not est)
python experiments/len40_real_oracle/analyze_rounds.py --dump-dir /path/to/dumps
```

## What was observed (leaned, max_rounds=3)

| round | states | est | phi(DFA, oracle) | note |
| --- | --- | --- | --- | --- |
| 0 | 22 | 0.100 | (CpG-ish) | shatters |
| 1 | 34 | 0.200 | ~0 | shatters more |
| 2 | 8 | 0.816 | **+0.000** | **accept-all** (accept-rate 1.000 at every frame level) |

The driver returns the best-`est` DFA across rounds -> round 2 -> the **accept-all**
classifier. est rewarded the trivial collapse. A `max_rounds=10` continuation is the
open question: does the accept-all collapse persist, re-shatter, or oscillate?

## Frame signal is real but positional (the deconfounding subtlety)

`probe_frame_signal.py` confirms the frame signal exists: appending a stop codon `TAA`
has a **period-3** marginal effect (spread ~0.154 across `L mod 3`), while a non-stop
control `AAA` does not (~0.012). So the target is genuinely there.

But at length 40 it is a small ripple on the oracle's *dominant* structure: both `TAA`
and `AAA` also swing +-0.2 aperiodically length-to-length. The deconfounded oracle is
still ~0.53-correlated with a purely positional linear score. This is why a DFA has a
hard time: the frame period-3 is finite-memory and regular, but it is buried under
aperiodic positionality that is not. (This positionality is also what makes the
translation-invariance gate over-block -- see the note referenced below.)

## Related, elsewhere (not in this directory)

- **Synthetic shattering repro** (`InteractionOracle`) and the translation-invariance
  gate work live on PRs #192 / #197. #197 also documents that the invariance gate
  *over-blocks* on this real oracle (it refuses the frame distinguishers too, because
  every distinguisher's marginal effect is positional here).
- The gate oracle's polished form is PR #194 (with the band assertion noted above).

## Config provenance (leaned vs original)

The "leaned" defaults in `run_len40.py` (`counterexample_probes=250, per_state=12,
min_indecisive=50`) exist because the original heavier config OOM'd during round 2
(peak > available RAM). Leaning caps peak RSS ~17 GB and lets the run reach round 2+.
The leaning changes pool sizes, not `fnr_limit`. If you want the original trajectory
past round 2, the alternative is a memory fix (bytes-key table memoization) rather than
leaning -- not done here.
