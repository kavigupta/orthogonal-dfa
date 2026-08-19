# Performance: the counterexample sift re-averages the *whole* suffix family per node

## Symptom

On an expensive deterministic oracle (SpliceAI residual, `membership_query` = a batched
GPU forward pass), the counterexample-discovery phase of `synthesize_direct_lstar_fnr`
runs at roughly **8 seconds per probe**. Round 0 alone (a few hundred probes) takes ~1
hour, and a multi-round run is hours-to-days — dominated entirely by oracle calls, not by
the SpliceAI model itself.

Measured on the composition-deconfounded SpliceAI oracle at `min_signal_strength=0.05`,
`fnr_limit=0.02` (a family the FNR gate grew to **~4809 suffixes**):

```
[probe  75/400] splits=5 states=7  elapsed=628s   (~8.4 s/probe)
[probe 125/400] splits=8 states=10 elapsed=1213s
```

The states/distinguishers themselves are healthy (short, meaningful: `T`, `AT`, `TAT`,
`TGA`) — this is not a shattering problem. It is purely oracle-call volume.

## Root cause

A single membership test on the tree averages the oracle over the **entire** suffix
family, and the sift path calls it once per tree node, several times per probe.

1. **`SuffixFamily.is_accept(seq, midfix)` averages over all `vs`.**
   `is_accept` → `mean` (`suffix_family.py:53`), and

   ```python
   value = sum(self.bits(list(seq) + list(midfix))) / len(self.vs)   # line 60
   ```

   `bits` (`suffix_family.py:33`) queries the oracle on `seq + midfix + v` for **every**
   `v in self.vs`. After the FNR gate grows the family, `len(self.vs) ≈ 4809`, so **one
   `is_accept` call ≈ 4809 oracle queries ≈ 5 batched passes** (batch 1024).

2. **`tree.sift` calls `is_accept` at every node on the path.**
   `Sifter.sift_and_boundary` → `tree.sift(seq, self.family.is_accept)` (`sifting.py:24`);
   `MidfixTree.sift` walks node by node calling `decide(seq, midfix)` at each internal
   node (`midfix_tree.py:128`). For a 22-state tree that is ~5–20 `is_accept` calls per
   sift.

3. **Each probe does ~log(len) sifts.**
   `counterexample_pass` (`direct_lstar.py:287`) → `process` → `_first_disagreement`
   (`direct_lstar.py:151`) binary-searches the probe for the divergence point, sifting
   `seq[:mid]` at each step: ~`log2(len(probe))` ≈ 7 sifts per probe.

Multiplying:

```
per probe ≈ (~7 sifts) × (~10 nodes) × (4809/1024 ≈ 5 batched passes) ≈ ~350 forward passes
```

So each probe is **hundreds** of batched SpliceAI passes, not the handful one would
expect — hence ~8 s/probe.

## Why it is wasteful

`is_accept` only needs to decide which side of `accept_thresh` / `reject_thresh` the
prefix's accept-rate falls on (`_decide`, `suffix_family.py:64`). For that decision the
full 4809-suffix family is enormous overkill:

- A prefix whose true rate is far from the threshold is settled by **~10–50** suffixes; a
  sequential probability-ratio test would early-stop the vast majority of sifts long
  before exhausting the family.
- The large family exists to make the **FNR gate** cross `fnr_limit` over the whole
  representative population — a one-time, population-level statistic. Reusing it verbatim
  as the **per-string** membership estimator on every sift conflates two different jobs:
  "size the family so the population is resolvable" vs. "spend the minimum evidence to
  classify *this one* string".

There is already a batched path — `Sifter.prefill` / `MidfixTree.classify_many`
(`sifting.py:34`, `midfix_tree.py:164`) warm a whole tree level in one call — but the
probe loop cannot use it: each probe may split the tree, so probes are processed one at a
time, and the batching that would amortize the family across many strings never happens.

## Fix

**Early-stopping `is_accept`** (`suffix_family.py`). The fixed full-family mean is
replaced with a sequential binomial test: family suffixes are drawn a block at a time
and the test stops as soon as `binomial_side_of_boundary` is confident the *full* family
would land past `accept_thresh` / `reject_thresh`. The full family is spent only once a
prefix is exhausted without a decision — the genuine boundary prefixes — so an exhausted
verdict is exactly the old full-family decision, and every other verdict matches it with
probability `1 - _SIFT_ALPHA`. A prefix far from the threshold settles in the first
16-suffix block: ~300× fewer queries on the common case at `len(vs) ≈ 4809`.

Suffixes are revealed in a fixed content-independent shuffle (`_sift_order`), not the
family's own order, so each block is a representative sample even though screening admits
suffixes in correlated batches — a contiguous prefix of the raw order can be badly biased.

`Sifter.prefill` warms only that first block (`SuffixFamily.warm_sift`) rather than the
whole family; deeper blocks are pulled per-string, for the minority that need them, so a
tree level's easy sifts still settle in one batched call while the full-family warm no
longer dominates. The FNR gate and the split test (`votes`) keep the whole family — sizing
the family so the *population* is resolvable is a different job from classifying one string.

Two fixes from the original write-up were **not** taken: a fixed per-sift subsample loses
the sequential test's adaptivity for no extra saving, and cache-aware batching of the
binary-search sifts is subsumed — early-stopping removes the volume the batching was meant
to amortize.

## Repro

Run any `synthesize_direct_lstar_fnr` on an oracle whose `membership_queries` is slow,
with a family the FNR gate grows large (low `fnr_limit`), and time the discovery phase;
count oracle strings per sift to see the first-block early stop on the common case.
