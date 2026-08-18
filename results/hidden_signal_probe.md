# Does the ladder *hide* recoverable signal? (No — the pathology is absence, not concealment)

A natural worry about the prepend-ladder is that it *hides* signal: that a compact,
recoverable structure is present but the learner squanders itself unrolling a shift
register and exports nothing. This note tries to build that failure and finds it
doesn't happen — for regular signal the learner recovers it, and the "recovers
nothing" in `ladder_writeup.md` is genuinely **no compact signal**, not hidden signal.

## The construction: real signal you'd expect to get lost

`HiddenFrameOracle` (in `orthogonal_dfa/analysis/hidden_signal.py`):

```
label(seq) = [nframes(seq) >= 1]  XOR  [positional_score(seq) > tau]
```

- `nframes >= 1` is the PR's proven-learnable, **regular** stop-codon chain — a genuine
  compact signal.
- the positional flip is a rare (`tau` calibrated to fire on a fraction `flip` of
  strings), structured, **non-regular** perturbation, entangled by XOR so it *cannot be
  routed away as noise* — it corrupts the very strings that carry the frame signal, and
  it is what drives the ladder.

At length 16, `flip = 0.2`: base accept-rate **0.506** (so chance is 0.506 — no
majority-class shortcut), yet the compact "contains-a-stop-codon" DFA scores **0.80**
against the oracle. Real, strong, compact signal, sitting at the chance line. The
positional term alone predicts at **0.476** — useless by itself.

## Result: the signal is *not* hidden

| flip | ceiling (compact frame DFA) | learner acc vs oracle | acc vs clean frame | verdict |
| ---: | ---: | ---: | ---: | --- |
| 0.20 | 0.803 | **0.803** | 1.000 | recovered *at* the ceiling |
| 0.35 | 0.661 | **0.655** | 0.847 | recovered *at* the ceiling |

At every distractor strength tried, direct-L\* recovers the frame signal right up to the
(distractor-reduced) ceiling. Cranking `flip` only *lowers the ceiling* — it destroys
signal rather than concealing it. No hiding regime appeared.

## Why the output survives the churn

1. **L\* is complete for regular languages, and the FNR/suffix-clustering machinery is
   built to isolate exactly these chains** — so a sufficiently-present regular signal
   gets found. (Same reason the `FrameOracle` control in `ladder_writeup.md` recovers
   1.000 even under 45% *structureless* noise.)
2. **`est` falls as the DFA ladders.** The round score `est`
   (`counterexample_synthesis._estimate_accuracy`) is not accuracy against the oracle —
   it is **DFA-vs-tree fidelity** (does the exported transition function faithfully
   reproduce live sifting through the discrimination tree). Laddering spawns edges the
   family can't resolve (the "no decisive edge → self-loop" fallbacks), so the DFA tracks
   its own growing tree *worse*, and `est` drops: e.g. `0.600 → 0.567` on the mix,
   `0.433 → 0.233` on the pure positional oracle. Because `_Best` keeps the highest-`est`
   round, the drift is *penalized*, and the clean early round is what gets exported.

Note the gap this exposes: on the mix, round 0 had `est = 0.600` but true oracle
accuracy `0.803`. `est` and accuracy are different quantities; `_Best` selecting on
`est` happens to be safe here only because `est` co-moves against laddering. On a target
where an overfit round had *higher* internal fidelity than the signal-bearing round,
`_Best` would return the wrong DFA — a latent mismatch worth fixing (rank rounds by
held-out oracle agreement, which the learner can afford, not by `est`).

## Conclusion

You cannot easily *hide* genuinely-regular signal from this learner. The prepend-ladder
wastes compute, but `_Best` and L\*'s completeness preserve the recoverable output. So the
`PositionalScoreOracle` "recovers nothing" is **absence of compact structure**, not
concealment of it — which sharpens, rather than contradicts, the PR's thesis: the
discriminator is regularity of the target, and where compact regular structure exists,
it is recovered even buried in a non-regular distractor.

The one regime where regular signal is genuinely unrecoverable — states reachable only by
rare/long prefixes the fixed-length sampler underrepresents — the learner already
*detects and reports* (`uncoverable_access_strings` → "target not learnable with this
sampler, stopping"), so it is flagged, not silently hidden.

## Constructive direction: closure / merging as a regularity signal

If "avoid the valueless splits" is the goal, the split test's question ("do these members
separate?") is local and can't see value; the distinguishing property is whether the
learned automaton *closes and merges* — a finite automaton over long strings must cycle
(pigeonhole), whereas a shift-register ladder grows forward forever. Two measurable
quantities bear this out:

- **closure** `= 1 − (self-loop-fallback edges / total edges)`
- **cyclic core** `=` fraction of states inside an SCC of size ≥ 2 (self-loops excluded)

| target | regular? | states | closure | cyclic core |
| --- | --- | ---: | ---: | ---: |
| parity (count of `A` mod 2) | yes | 2 | 1.00 | 1.00 |
| mod-3 (count of `A` mod 3) | yes | 3 | 1.00 | 1.00 |
| `PositionalScoreOracle` | no | 12 | 0.79 | 0.42 |

Both quantities separate regular from non-regular, though as a *graded* score, not a
binary certificate (the forced totalisation manufactures some incidental cycles, so the
positional machine keeps a 42% core rather than 0). Because a `.*11111.*`-style chain is
itself cyclic (its reset edges make `seen-0…seen-4` one SCC), this suggests an actual
operation — **collapse the acyclic, non-merging tail into the cyclic core** (keep the
core states; redirect any core→tail edge to an absorbing sink labelled by the tail
state's accept status — at the point the ladder would start, commit to the classification
instead of unrolling).

### Core-extraction results

| target | full states | full acc | core states | core acc | note |
| --- | ---: | ---: | ---: | ---: | --- |
| parity | 2 | 1.000 | 2 | 1.000 | identity — a regular target is all core |
| `PositionalScoreOracle` | 12 | 0.540 | 6 | 0.496 | compresses; stays at chance (no signal to keep) |
| frame-XOR mix (flip 0.2) | 9 | 0.803 | **4** | **0.803** | signal preserved (frame acc still 1.000), tail dropped |

The mix is the point: core-extraction drops the 9-state machine to **4 states while keeping
the frame signal exactly** (accuracy at the 0.803 ceiling, agreement with the clean frame
signal still 1.000). The recoverable signal *is* the cyclic core; the acyclic tail was
droppable ladder. On the pure positional target the same operation compresses the machine
and it stays at chance — honest, because there is no compact signal to keep. So
closure/merging is not just a diagnostic score but an operation that separates the
finite-memory skeleton from the shift-register tail.

Reproduce with `python -m orthogonal_dfa.analysis.hidden_signal`.

### Acting on it during synthesis: the ladder gate

Core-extraction above is a post-hoc transform on the exported DFA. The same merging
signal can act *during* synthesis, so the ladder never runs unbounded in the first
place: `synthesize_direct_lstar_fnr(..., ladder_budget=k)` (off by default) ends a
discovery pass once `k` splits in a row produce a state that does **not** merge back
into the automaton — no resolved edge to an earlier state and no edge from one into it
(`DirectLStarLearner._merges_into_existing`). A regular target's splits close and merge,
so the run resets and never trips it (verified: parity still converges with the gate on,
`tests/test_ladder_gate.py`); a non-regular prepend-ladder unrolls a shift register and
does, bounding the growth. Because it can only fire when `ladder_budget` is passed, it is
inert for every existing caller.
