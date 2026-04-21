# Findings: `test_another_countexample_poor_case` synthesis failure

Ground-truth DFA (10 states, alphabet {0,1}, initial=0, final={1}):

```
0: {0:0, 1:8}    5: {0:4, 1:8}
1: {0:1, 1:1}    6: {0:9, 1:3}   (state 1 = absorbing accept trap)
2: {0:6, 1:1}    7: {0:6, 1:8}
3: {0:2, 1:9}    8: {0:5, 1:8}
4: {0:8, 1:3}    9: {0:7, 1:3}
```

Random length-40 strings: GT accept rate ≈ 66%.

## What the learner produces

9-state DFA, initial=3, finals={0}:

```
0: {0:0, 1:0}  (absorbing accept)
1: {0:4, 1:0}   5: {0:7, 1:2}
2: {0:1, 1:5}   6: {0:3, 1:2}
3: {0:8, 1:3}   7: {0:4, 1:3}
4: {0:5, 1:2}   8: {0:6, 1:3}
```

Fresh-sample accuracy: **0.974** (26/1000 strings misclassified; all false positives).

## The structural error

Product-DFA BFS shows the learned DFA and GT disagree on **16 reachable edges**, but all but one are downstream of a single bug:

```
(gt=5, learned=1) --1--> (gt=8, learned=0)
```

Learned state 1 is a merge of GT states `{2, 5}`. In GT:
- 2 --1--> 1 (accept)
- 5 --1--> 8 (reject)

The merge picks the "accept" side, so any path through `(gt=5, learned=1) --1--> ...` wrongly accepts.

Shortest disagreeing input: `00101` (length 5), illustrating the bad merge.

## Ruled-out explanations

1. **Binary search in `locate_incorrect_point`**: Returning the longest witness rather than the binary-searched shortest didn't help (69 min run, still fails).
2. **Empty-string collapse**: `dt.classify([]) == dfa.initial_state` at every iter, so `locate_incorrect_point` starts from the right state.
3. **Reduced-predicate noise losing disagreement signal**: Diagnostic on 1000 fresh samples at iter 2 shows decisive and reduced DT both catch all 20 disagreements — reduction isn't the bottleneck.
4. **Deduplication of short prefixes**: 2000-sample replay with `x=[]` yielded 9 distinct prefixes (43 of 49 passed the decisive-DT recheck). Dedup isn't killing many.
5. **Outer reset-suffix property**: No outer `Σ*LΣ*` can have a reset suffix (absorbing-accept state blocks it). Measuring only on non-absorbing outer states: seeds 1 and 5 have worst-case continuation accept 0.737 and 0.677; not a clean predictor of synthesis outcome.
6. **Class-preservation metric `frac_strings_preserving`**: Weak correlation with synthesis success. The failing DFA scores 0.056 — barely above the natural 0.05 filter.

## The actual cause

The DT's root predicate's accept/reject distribution per GT state (suffix family clustered around the empty suffix, boundary ≈ 0.5):

| GT state | reject_frac | accept_frac | class under predicate |
|---|---:|---:|---|
| 0 | 1.000 | 0.000 | reject |
| **1** | **0.000** | **1.000** | **accept (trap)** |
| **2** | **0.344** | **0.656** | **accept** ← should be reject |
| 3 | 0.934 | 0.066 | reject |
| 4–9 | 1.000 | 0.000 | reject |

State 2's empirical accept rate on the clustered suffix family is **0.656** — above the 0.5 boundary. So state 2 routes to the DT root's **accept** subtree. The DFA construction labels every leaf in that subtree accepting, so state 2 gets merged with the accept trap.

### This is not a threshold problem

A tighter accept threshold (say 0.8) would split state 2 out, but wouldn't generalize — other benchmarks place state 2 analogues with accept_frac closer to 1.0 under their suffix families.

### It **is** a prefix-distribution problem

Prefix distribution among the initial 200-prefix PST (after PST rounds out to ~800):

| GT state | prefix count |
|---:|---:|
| 1 | 522 (accept trap) |
| 8 | 90 |
| 5 | 53 |
| 3 | 34 |
| 9 | 25 |
| 4 | 27 |
| 7 | 18 |
| 6 | 18 |
| **2** | **13** |
| 0 | — |

State 2 is hit by only ~1.5% of uniform random 40-char walks. Clustering `identify_cluster_around` picks suffixes that minimize per-prefix disagreement with the seed-suffix pattern; state 2's 13 prefixes have little weight in the aggregate, so the chosen suffix family doesn't pull hard enough on state-2's "reject" signal.

### Confirming with enrichment

Injecting 100 extra prefixes that end in GT state 2, then rerunning the suffix-family clustering:

| GT state | reject_frac (before) | reject_frac (after) |
|---|---:|---:|
| 2 | 0.344 | **1.000** |

Every non-trap state now has reject_frac ≥ 0.82. State 1 remains the sole accept-side state. The root predicate can now cleanly separate the accept trap from the rest.

## Why does a descendant predicate not fix this?

In principle, the DT could split within the accept subtree: at depth 2 the predicate `[0] + suffix` has reject_frac 1.0 from state 2 and 0.0 from state 1 — perfect discrimination. But the DFA's accepting-state labeling comes from `accepting_states = collect_states(dt.by_rejection[1])`, which includes **every** leaf in the accept subtree. Even a successful depth-2 split labels both leaves accepting, so state 2 stays in an accepting leaf regardless.

## Implication for fixes

Mis-routing at the root is the root cause. The fix must happen at (or before) the root split, either by:

- **Prefix enrichment** directed at the DT's borderline accept leaves (prefixes whose accept_frac is close to the decision boundary). More state-2-like prefixes → suffix clustering weights state-2's "reject" signal more → accept_frac crosses below the boundary → state 2 is correctly routed.
- **Re-evaluating the decision boundary** after the suffix family is chosen, using an actual bimodal-split detection on the prefix accept-frac distribution rather than the heuristic center at 0.5.
- **Partitioning prefixes by current DT leaf, then re-clustering suffixes within each partition** to find suffixes that shatter that leaf. (Closest to classical L* "distinguishing suffix" search.)

## Reproducers

- `scripts/investigate_another_poor_case.py` — hard-coded learned and GT DFAs, product-BFS, shortest disagreeing inputs.
- `scripts/check_suffix_family.py` — runs suffix-family clustering, reports reject_frac per GT state under the chosen family, and per `[c]+suffix` predicate.
- `scripts/enriched_suffix_family.py` — injects extra state-2 prefixes and shows the resulting family cleanly discriminates.
