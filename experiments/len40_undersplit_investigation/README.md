# len40 undersplit investigation

Why does E-L* / direct-L* produce a **misaligned** automaton in **round 0** on the real
spliceai len40 stop-codon oracle (this PR, #213), and is PR #262 (decisive-edge-routing)
a real fix or chasing an artifact?

## Setup / oracle

- `gate_residual_oracle(default_exon, load_spliceai(400, 0), length=40, len_lo=35, len_hi=85)`
- Superlanguage: `KmerVocabulary(kmers=((3,0,2),(3,0,0),(3,2,0)), base_alphabet_size=4)`
  (alphabet_size 6), `SuperSampler(vocab, 36)`, `LiftedOracle(base, vocab, seed)`.
- **Frame rule** (`sig()` in most scripts, applied to the SUPER-string): SINK = both
  reading frames f0 & f1 closed by a stop → should REJECT; live = otherwise → should ACCEPT.

## Harness

- **`scripts/instrument_multiround.py`** — runs the full
  `counterexample_driven_synthesis` and dumps a per-round pickle: `dfa`, `dt`
  (MidfixTree), thresholds/`boundary`/`evidence_margin`, `true_acc` (est), the prefix
  `prefixes` pool, harvested `indecisive`, and eval `call`/`ora`/`fpat` + phi.
- **`SPLIT_LOG` hook** in `orthogonal_dfa/l_star/transition_resolver.py`
  (`_apply_split`, env-gated): logs each split's `distinguisher` / `witness` / `sprime`.
  Inert unless `SPLIT_LOG=<path>` is set. Added for the "why is the split missing" trace.

### Run recipe

```
cd wt208   # or the repo root once run from here
XDG_CACHE_HOME=$PWD/../permacache_home CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=$PWD \
  python -u experiments/len40_undersplit_investigation/scripts/instrument_multiround.py \
  --seed 2 --rounds 3 --acc-threshold 0.98 --fnr-limit 0.10 --dump-dir <out>
```
Notes: `--fnr-limit 0.10` is required (the hardcoded 0.02 + uncapped loop OOMs on the
len40 round-0 run). Model load needs a real GPU (weights live on-device; `CUDA_VISIBLE_DEVICES=""`
fails). Analysis scripts hardcode absolute scratchpad `DUMP` paths — repoint them at
`dumps/` when re-running from the repo. Dumps present: `dumps/mr_dumps/seed{0,2}`,
`dumps/mr_multi/seed{1,3}`.

## Findings (the arc)

1. **The "collapse" is a pre-denoise label-noise artifact.** Raw phi +0.067;
   `denoise_accept_labels` recovers round0 → +0.116, round1 → +0.170 (`denoise_check.py`).
2. **`est` = DFA↔tree routing self-consistency**, NOT oracle accuracy
   (`estimate_agreement_rate`). Round0 est 0.633 > round1 0.533, so
   `do_counterexample_driven_synthesis` keeps **round 0** — the *worse-aligned* automaton.
   A mis-selection driven by round 1's higher fresh-string indecision (22.5% → 38.7%).
3. **Round 1 = clean frame automaton** (each sig → 1 pure state, SINK → 1 state);
   **round 0 = misaligned** (SINK fragmented across states incl. ACCEPT-states s4/s13).
   `dfa_structure.py`; `partition_align.py` (walk SINK/live homogeneity seed2:
   round0 0.543 vs round1 1.000).
4. **The worse labeler yields the better DFA.** Per-string family quality
   (`per_length_indecisive.py`): round0 family is the BETTER labeler (FPR 7-19%, FNR 5-8%);
   round1 WORSE (FPR 22-27%, FNR 20-24%). Yet round 1's DFA is the aligned one.
5. **Export/construction, not the family, decides alignment.** Both rounds' tree-leaf
   (sift) partitions are only ~0.4-0.5 SINK/live-homogeneous; the exported walk is
   round0 0.528 vs round1 1.000 (`sift_vs_walk2.py`). Sift/walk agree only 0.326 (round0)
   — the est self-consistency gap.
6. **Pool is the only carried state; strictly nested** pool0 (872) ⊂ pool1 (1268),
   0 dropped (`pool_compare.py`). The suffix family is **re-drawn every round and derived
   from the pool** (`sample_suffix_family` clusters suffixes by their reads across the pool
   prefixes) — so the family is NOT an independent lever; "worse family → better DFA" is
   confounded with pool growth.
7. **Refuted mechanisms (measured, not assumed):** within-sig straddle
   (`within_sig_straddle.py`), midfix straddle (`midfix_straddle.py`), length effect
   (`length_effect.py`), over-split-on-noise (contradicted by the Bernoulli tests —
   `AllFramesClosedOracle` recovers the frame rule under noise), per-string FPR-leak
   (round1 has HIGHER SINK-accept FPR yet the cleaner DFA).
8. **Reframe:** the family has two decoupled jobs — LABELING (accept/reject; round1 worse)
   and PARTITIONING (state geometry; round1 better). DFA quality tracks partitioning.

## Current question (open)

Why does round 0 **undersplit** (conflate two frame classes into one state) despite a decent
labeler, while round 1 does not despite a worse one? Method: find a specific missing split
and figure out *why* it is missing.

### `missing_split.py` — first result (seed 2), with a caveat

- Target undersplit: round-0 state **s10 (ACCEPT)** conflates **283 SINK + 183 live**;
  round 1 separates them cleanly (SINK → s9, live → s6, 60/60).
- Surprising: round-0's OWN family+band already separates s10's SINK/live at the empty root
  (SINK mean 0.47 → 67% reject; live 0.67 → 83% accept), yet round 0 conflated them. So for
  this state the undersplit is **not** a family-labeling failure — the family could split it.
- **Caveat / flaw:** tree midfixes are *conditional on the path* to their node;
  `missing_split.py` tested each midfix *unconditionally* on all s10 members, which dilutes
  round 1's conditional split. So it names the undersplit and shows round 0's family *could*
  separate it, but does **not** correctly name round 1's actual split. See next step.

### Next step (in progress)

Walk s10's SINK/live subsets through round 1's ACTUAL tree to find the node where their
paths diverge (the real split); check round 0's tree for a corresponding node; use
`SPLIT_LOG` to see why round 0's counterexample pass never created it.

## Abandoned: the (pool × family) swap

`scripts/swap_build.py` rebuilds the DFA from every (pool, family) combination. Abandoned
because (a) it is too expensive cold — `close_edges` ~20 min/config, the cold-oracle cost
the live run amortizes; and (b) it is **ill-posed** — the family is derived from the pool,
so injecting a foreign family tests a configuration the learner never produces, and its
result would not explain the real rounds. Kept for the record only.

## Logs

`logs/` holds captured run outputs (`plen_s2.out`, `sift_walk_s2.out`, `missing_split.out`,
etc.). The scratch `scripts/` set is exploratory and larger than the files named above; the
scripts referenced here are the load-bearing ones.
