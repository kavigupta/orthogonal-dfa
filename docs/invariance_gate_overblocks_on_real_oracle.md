# The invariance gate over-blocks on the real (positional) oracle

The Bayes-factor gate correctly refuses the synthetic positional ladders (the
`PositionalScoreOracle`, and the `InteractionOracle` the old fixed-threshold gate
missed). But run against the **real composition-deconfounded SpliceAI oracle** it
**over-blocks**: it refuses the genuine reading-frame distinguishers too, so enabling it
yields no splits (a trivial DFA) — it throws out the signal with the ladder. The
synthetic tests do not expose this because they contain no recoverable signal to block.

This note records the evidence and the conceptual reason, because it changes what the
gate should measure.

## 1. Both gates refuse the frame distinguishers

Length-40 gate-residual SpliceAI oracle, `sample_length=40`. `distinguisher_position_dependence`
is the old residual-std score (threshold 0.04); the Bayes factor is `> 0 => refuse`.

| distinguisher | kind | old (resid-std) @0.04 | new (logBF) @0 |
| --- | --- | --- | --- |
| `CG` | CpG ladder | 0.069 REFUSE | +3.3 REFUSE |
| `CAAGCG` | CpG ladder | 0.093 REFUSE | +33.2 REFUSE |
| `TAA` | **frame** | 0.078 REFUSE | +145.8 REFUSE |
| `GTAAT` | **frame** | 0.103 REFUSE | +29.7 REFUSE |
| `AATAATTCGTAAT` | **frame** | 0.124 REFUSE | +170.5 REFUSE |

The frame distinguishers score *higher* position-dependence than the CpG ones and are
refused by both gates. (For contrast, the `InteractionOracle` scores 0.038 on the old
score — under 0.04, so the old gate missed it — while the Bayes factor refuses it; that
is the new gate's genuine improvement, but it does not help here.)

## 2. Direct evidence: the frame distinguisher's effect is period-3 positional

Not via the gate's summary — measured on the oracle directly. Define the marginal effect
of appending distinguisher `d` at absolute position `L`:

    g(d, L) = accept_rate(random length-L string + d) - accept_rate(random length-L string)

Appended after a length-`L` string, `d` sits in reading frame `L mod 3`, so a stop-codon
`d` closes *that* frame. Averaged over 3000 strings per length, `L` in 24..42:

```
TAA (stop):  mean g by L%3  ->  frame0 -0.054  frame1 -0.063  frame2 +0.091   (period-3 spread 0.154)
AAA (ctrl):  mean g by L%3  ->  frame0 +0.027  frame1 +0.020  frame2 +0.032   (period-3 spread 0.012)
```

So the stop codon's effect genuinely depends on which frame it lands in (period-3, spread
0.154), and a non-stop control has no such structure (0.012, ~13x smaller). The frame
distinguisher **is** position-dependent — confirmed from the oracle, not the gate.

But note the caveat that is the crux: **both** `TAA` and `AAA` also swing ±0.2
length-to-length. That aperiodic per-length variation is the oracle's *dominant,
non-frame positional structure* (the deconfounded oracle is ~0.53-correlated with a
positional linear score). The frame period-3 is a small ripple on top of it.

## 3. Why the gate is measuring the wrong thing

The gate scores the **marginal** effect `g(d, L)` and asks whether it varies with `L`. On
a pervasively-positional oracle *every* distinguisher's marginal effect varies with
position — the ±0.2 aperiodic swing hits `TAA` and `AAA` alike — so the gate refuses
everything. It is not even reacting mainly to the frame's period-3; it is reacting to the
oracle being positional under *any* distinguisher.

What we actually want to test is different: **is the SPLIT position-sensitive** -- does the
two-way partition of prefixes that `d` induces via `is_accept(prefix, d)` depend on
*absolute position* (a shift-register that must remember where it is, unbounded memory),
rather than on a **transportable finite-memory feature** (even a periodic one like reading
frame, which a bounded automaton tracks)?

The two are not the same:

- A finite-memory feature can have a position-dependent *marginal* effect. Reading frame
  is the example: its value depends on `pos mod 3`, so `g(TAA, L)` is period-3 -- yet the
  frame language is regular, a bounded automaton, and its split *transports* (the same
  "stop seen in frame f" feature wherever it occurs, with period 3).
- The pathology is the opposite: a split whose partition is tied to *absolute* position
  and does not transport -- the shift-register ladder, where each rung remembers one more
  symbol of where the prefix ends.

Marginal-effect position-dependence (`g(d, L)` varies with `L`) conflates the split's
position-sensitivity with the oracle's overall positionality. On the synthetic oracles the
two coincide (there is no recoverable finite-memory feature, so all positionality is the
pathology). On the real oracle they diverge: the frame split is a finite-memory feature
with a position-dependent marginal effect, and the gate wrongly refuses it.

## 4. What a correct gate would test

A signal for "the *split* is position-tied, not transportable" -- e.g., whether the
prefix partition `is_accept(., d)` is preserved under a shift of the operating length (a
finite-memory / periodic feature transports, with its period; a shift-register split does
not), or whether the split can be reproduced by a bounded-memory predictor of the prefix
rather than by its absolute end position. The current probe answers the marginal-effect
question, which is necessary but not sufficient, and on a positional oracle is dominated by
structure that has nothing to do with whether the split is finite-memory.
