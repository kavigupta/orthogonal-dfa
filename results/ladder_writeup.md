# The direct-L\* distinguisher *ladder*, and its two causes

## The phenomenon

On the composition-deconfounded SpliceAI oracle, direct-L\* does not converge to a
compact DFA. The discrimination tree grows a **ladder** of distinguishers, each one a
single symbol prepended to a previous one:

```
CAAGCG -> ACAAGCG -> AACAAGCG -> CAACAAGCG -> TCAACAAGCG -> GTCAACAAGCG
   GCG -> TGCG -> GTGCG                       (a second, parallel chain)
```

Every rung is exactly `symbol ++ previous_rung`; each new state differs from its parent
only by remembering one more symbol of left-context. The learner adds states without
bound and the exported DFA recovers **no** signal (held-out accuracy at chance).

This note reproduces that with ~20-line programmatic oracles
(`orthogonal_dfa/analysis/ladder_repro.py`) and — the point of the note — shows the
behavior has **two independent ingredients** that are easy to conflate.

## Cause 1 — new distinguishers can only grow by prepending

When a counterexample exposes an ambiguous edge — two strings `witness` and `sprime`
both sift to a state `s1`, but following symbol `c` should send them to different
leaves — the learner builds the separating distinguisher with
`first_disagreement(witness, sprime, prefix=[c])` (`direct_lstar._act_on_disagreement`).
That returns

```python
full = (*prefix, *midfix)          # prefix = [c]; midfix = an EXISTING tree distinguisher
if decide(s, full) != decide(sprime, full):
    return full                    # new distinguisher = c ++ existing_node_midfix
```

(`MidfixTree.first_disagreement`). So **every new distinguisher is the edge symbol
prepended to a distinguisher already in the tree** — distinguishers can only ever grow,
one symbol at a time, on the left. This is a property of the algorithm, present on every
run.

## Cause 2 — the DFA can never agree with the oracle

The prepend rule *by itself* does not force an unbounded ladder. A target with genuine
compact finite-state structure reuses short distinguishers and **closes** — the
counterexamples run out and synthesis converges. The chain grows without end only when
the learner can **never** make the DFA agree with the oracle, so the counterexample
search never runs dry and each ambiguous edge is resolved by lengthening the context one
symbol at a time.

The two causes are separable, and the two oracles below separate them.

## Control — a *regular* target ladders but converges

`FrameOracle`: accept iff `nframes(s) >= 2` (≥2 of the 3 reading frames contain a stop
codon), with a fraction `1 - SW` of strings routed to a balanced hash-coin instead.
`nframes >= 2` is a **regular** language whose minimal DFA is a chain (per frame, "have I
seen a stop yet"), and it is ~50/50 balanced so the median boundary engages.

direct-L\* **does** ladder on it — the frame chain literally is a stop-codon
prepend-ladder:

```
A -> AA -> TAA -> GTAA -> AGTAA -> AAGTAA
```

but it **converges to the correct DFA and recovers the clean signal perfectly at every
noise level**:

| SW | states | longest chain | acc vs oracle (ceiling `SW+(1-SW)/2`) | acc vs clean signal |
| --- | --- | --- | --- | --- |
| 1.00 | 14 | 6 | 1.000 (1.000) | **1.000** |
| 0.70 | 14 | 5 | 0.843 (0.850) | **1.000** |
| 0.55 | 14 | 6 | 0.768 (0.775) | **1.000** |

The ladder is present, but it is the *real, finite* frame chain, and the DFA is correct.
**So the ladder alone is not the pathology.**

## Pathology — a *non-regular* target ladders forever and recovers nothing

`PositionalScoreOracle`: accept iff `sum_i W[i, s[i]] > 0` for a fixed random,
**position-specific**, per-position-centered weight table (so ~50/50 balanced). The
score is **positional** (not a function of composition) and continuous, so there is no
compact automaton to close on — structurally what SpliceAI's deconfounded residual *is*.

direct-L\* ladders, **adds states without bound, and its accuracy thrashes below
threshold instead of converging**:

```
round 0:  4 states, est 0.433
round 1: 10 states, est 0.300      # more states, LOWER agreement
round 2: 16 states, est 0.467
```

with prepend-ladders (`T -> CT -> GCT -> GGCT`, plus `CCT`, `TCT`, …) and dozens of
"no decisive edge, falling back to a self-loop" warnings. The exported 16-state DFA has
**held-out accuracy 0.507 against a base rate of 0.509 — pure chance.** It built states
and learned nothing.

## The contrast

| oracle | regular? | states | ladders? | recovers signal? |
| --- | --- | --- | --- | --- |
| `FrameOracle` | yes (~14-state chain) | 14, stable | yes | **yes — acc vs signal 1.000** |
| `PositionalScoreOracle` | no (positional, continuous) | 4→10→16, unbounded | yes | **no — acc 0.507 ≈ chance** |

Same prepend-ladder mechanism (Cause 1, always present); opposite outcome. The
discriminator is **Cause 2** — whether the target is compactly regular in the region the
midfix varies. When it is, the ladder is the real automaton and synthesis converges; when
it is not, the ladder is the learner unrolling a shift-register against structure that
was never there, and it never terminates.

## Why this matters for SpliceAI

The composition-deconfounded SpliceAI oracle is the `PositionalScoreOracle` case, not the
`FrameOracle` case: its surviving signal is **positional and non-compositional** (per the
positional-information analysis, single-position η² ≈ 0 in the interior the midfix
varies, and a bag-of-k-mers classifier separates the learned suffix family from random at
AUC ≈ 0.54 — i.e. no compact structure there to close on). So a growing prepend-ladder of
distinguishers is the expected, diagnostic signature: the learner is reporting that the
oracle carries no compact finite-state structure it can converge on at the current noise
floor. It is not blindness to the frame signal — the `FrameOracle` control shows direct-L\*
learns a regular frame chain fine, even buried in noise.
