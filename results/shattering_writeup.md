# Why E-L\* shatters on the composition-deconfounded SpliceAI oracle

## The phenomenon

We built a composition-deconfounded oracle — SpliceAI's exon call with a monotonic
bag-of-k-mers **gate** subtracted (`gate_composition_residual.py`), so only
non-compositional structure remains — and ran E-L\* on it (`gate_elstar.py`).

It did not converge to a compact DFA. It **shattered**:

- Round 0: **92 states**, but held-out signal ≈ 0 (φ = 0.005, MI = 1.6e-5 bits vs the
  oracle). In-sample splits that do not generalize.
- The observation table's suffix family exploded to **397,219 suffixes** (round-0 pickle
  is 682 MB).
- Round 1 **OOMed** (exit 137) doing `resolve_dfa`/`deepcopy` on that table — the same
  wall the linear `composition_residual` run hit.

This is not a SpliceAI quirk. It is what E-L\* does to a **deterministic but not
compactly-regular** target.

## The frame signal is real — and regular in isolation

The deconfounded oracle carries a genuine, non-compositional signal: it **rejects no-ORF
(all-frames-closed) strings** and accepts strings with an open reading frame.

- signed φ(accept, all-frames-closed) = **−0.21**
- accept | no ORF = 0.38 vs accept | ≥1 ORF open = 0.60
- I(call; all-frames-closed | full composition) ≈ 0.016 bits (gate); ~0.03 bits as a
  thresholded oracle at length 95.

And E-L\*'s own round-0 suffixes **do track this** (`suffix_frame_analysis.py`): the more
reading frames a prefix has closed, the more the suffix family drives the oracle to
reject (correlation −0.21, matching the oracle), and it is **phase-specific** — which
frame is open changes the call, not just how many. So the probes see the frame signal.

Crucially, **all-frames-closed is itself a regular language** (~20 states) and E-L\*
learns it fine in isolation. The failure is not blindness to frame; it is that the frame
automaton is buried inside SpliceAI's non-regular residual, which shatters the learner
before it can isolate the ~20-state signal from the noise-level exact distinctions.

## Reproduction with a simple grammar

Shattering reproduces with a grammar you can write in three lines — no SpliceAI, no GPU,
no learned weights (`shattering_repro.py`). Over 300 random length-48 prefixes, count the
**distinct observation-table rows** (= states E-L\* would create by its exact
deterministic distinguishing) as the random suffix set grows:

| oracle (grammar) | 10 suf | 100 suf | 1000 suf | 4000 suf | final frac |
|---|---:|---:|---:|---:|---:|
| randomDFA (6 states, regular) | 1 | 1 | 5 | 5 | 0.02 |
| all-frames-closed (regular) | 9 | 20 | 20 | 20 | 0.07 |
| half **count**-compare (statistic) | 8 | 16 | 19 | 19 | 0.06 |
| half **lex**-compare (whole prefix) | 10 | 81 | 231 | **273** | **0.91** |

The two comparison grammars are almost identical — both ask *"is the first half bigger
than the second half?"* The only difference is **bigger how**:

```python
# saturates (~length/2 states): compares a STATISTIC
def count_compare(x):
    h = len(x) // 2
    return x[:h].count('A') > x[h:2*h].count('A')

# shatters (one state per prefix): compares the WHOLE prefix
def lex_compare(x):
    h = len(x) // 2
    return x[:h] > x[h:2*h]          # lexicographic
```

`count`-compare only needs the running A-count, so E-L\* collapses it to ~length/2 states
and it saturates. `lex`-compare needs the entire prefix: two prefixes are
E-L\*-equivalent only if identical, so **every prefix becomes its own state** and the
table runs away toward one-state-per-prefix.

## The mechanism (Myhill–Nerode, with E-L\*'s exact distinguishing)

E-L\* merges two prefixes `p1, p2` into one state iff `oracle(p1 + s) == oracle(p2 + s)`
for every distinguishing suffix `s`. For a **regular** target the number of such
equivalence classes is bounded (the DFA's states), so the table saturates. For a target
that is a fine function of the whole prefix (lex-compare; SpliceAI's residual), for
*every* pair `p1 ≠ p2` there exists a suffix that separates them — no merging is ever
final, and the family grows without bound. That unbounded growth is the shattering, and
because each state is pinned by in-sample-only distinctions, it carries no held-out
signal.

"Many states" is not the problem — the random DFA (5) and all-frames-closed (20) have
many and still converge. The problem is **unbounded, non-regular** growth.

## Implication

The lever is the **synthesis rule, not the oracle**. E-L\* distinguishes states
*exactly*: any reproducible difference forces a split. Against a deterministic non-regular
target that manufactures endless real-but-non-generalizing differences, exact
distinguishing shatters. To recover a weak regular signal (the frame automaton) from
inside a non-regular function, the merge step needs to be **noise/complexity-tolerant** —
e.g. merge prefixes whose accept-rate signatures agree to within a tolerance, or bound
model complexity — so the ~20-state frame structure can survive instead of being buried
under one-state-per-prefix shrapnel.

## Artifacts (this change)

- `orthogonal_dfa/analysis/shattering_repro.py` — the grammars (`RandomDFAOracle`,
  `HalfCountCompareOracle`, `HalfLexCompareOracle`) and `shattering_curve`; `run_round0`
  optionally runs the real synthesis end to end.
- `results/shattering_repro.md` — the generated table.

The SpliceAI composition-deconfounding pipeline that motivated this (the gate oracle, the
E-L\* run, the suffix analysis) is a separate change; the numbers cited above come from
it.
