# direct-L\* prepend-ladder reproduction — run output

Output of `python -m orthogonal_dfa.analysis.ladder_repro` (sample length 48). See
`ladder_writeup.md` for the full explanation.

```
===== CONTROL: FrameOracle (regular) -- ladders but CONVERGES =====
base accept-rate 0.516 | states 14 | distinguishers 12 | max len 6 | longest prepend-chain 6
held-out acc vs oracle 0.768  (chance = majority class = 0.516)
   ladder: AAGTAA <- AGTAA <- GTAA <- TAA <- AA <- A
   ladder: GGTAA <- GTAA <- TAA <- AA <- A
   ladder: CTAA <- TAA <- AA <- A
   ladder: ATAA <- TAA <- AA <- A

===== PATHOLOGY: PositionalScoreOracle (non-regular) -- ladders FOREVER =====
base accept-rate 0.509
round 0:  4 states, est 0.433
round 1: 10 states, est 0.300      # more states, LOWER agreement
round 2: 16 states, est 0.467
states=16 distinguishers=14 max len=4 longest prepend-chain=4
held-out acc vs oracle 0.507  (chance = majority class = 0.509)
   ladder: GGCT <- GCT <- CT <- T
   ladder: CCT <- CT <- T
   ladder: TCT <- CT <- T

## the contrast
                      oracle  regular?  states  chain    acc  chance  recovers?
                 FrameOracle       yes      14      6  0.768   0.516        yes
       PositionalScoreOracle        no      16      4  0.507   0.509         NO
```

## Reading the result

- **Both oracles ladder** — prepend-chains of distinguishers (`TAA -> GTAA -> AGTAA -> …`
  for the frame chain; `T -> CT -> GCT -> GGCT` for the positional score). The
  prepend-only distinguisher rule is always in force.
- **The regular control converges and recovers signal**: 14 stable states, held-out
  accuracy 0.768 well above the 0.516 chance line (and at the `SW=0.55` noise ceiling of
  0.775). Across `SW ∈ {1.0, 0.7, 0.55}` — vary `FrameOracle(sw=…)` — it recovers the
  *clean* `nframes>=2` target at accuracy **1.000** every time.
- **The non-regular pathology does not**: states grow 4 → 10 → 16 while `est` thrashes
  (0.433 → 0.300 → 0.467) and never crosses threshold, the run emits repeated
  "no decisive edge, falling back to a self-loop" warnings, and the exported DFA scores
  **0.507 against a 0.509 base rate — chance.** It built states and learned nothing.

Same mechanism, opposite outcome. The discriminator is whether the target is compactly
regular in the region the midfix varies — the composition-deconfounded SpliceAI oracle is
the non-regular case.
