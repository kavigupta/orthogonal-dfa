> Superseded by `start-at-k.md`.  Its claim that every harvest source besides the bisection is
> read whether or not a probe disagrees was wrong anyway: population re-sifts and edge resolution
> run after splits, and splits come from disagreements.

# Does a refused round force progress on main (no chains)?

Verdict: **(b)** — a zero-read fix closes it — with a **(c)**-shaped residual on main as it stands:
there is a narrow regime where the mixed harvest cannot force the next gate, and there main
progresses only if the re-clustered family happens to read the bad state differently.  It has
not been observed; a construction recipe is at the end.

## Model

Per node a string at target state q is read by a vote of n family bits, mean μ_q, sd σ ≈
√(p(1−p)/n); the band is [b − m, b + m), the gate reads at b.  With δ = |μ_q − b| / σ and
z_m = m / σ (≈ 2.6–2.9 at the shipped sizes):

    u(q) = Φ(z_m − δ) − Φ(−z_m − δ)        (undecided, per node)
    f(q) = Φ(−δ)                           (midpoint read on the wrong side, per node)

| δ | 0 | 1 | 2 | 2.5 | 3 | 3.5 | 4 |
|---|---|---|---|---|---|---|---|
| u | 0.99 | 0.95 | 0.73 | 0.54 | 0.35 | 0.18 | 0.08 |
| f | 0.5 | 0.16 | 0.023 | 6e-3 | 1.4e-3 | 2.3e-4 | 3e-5 |

So f/u ≤ 1.3e-3 once u ≤ 0.18: states read in the band rarely are also read across the middle
almost never.

**The gate's disagreement.**  Apart from decided wrong edges (below), a draw disagrees only if
some visit to some state flips at the middle:

    d ≤ Σ_q K(q)·f(q) + d_edge,     K(q) = E[visits of a draw to q] = L·V(q) ≤ L.

States with u < u* = 0.18 contribute at most f(u*)·L = 2.3e-4 · 40 ≈ 0.009 < ε/2.  So a refusal
(d > ε) needs states with u ≥ 0.18 carrying ≥ ε/2 of d, and a state at u needs
K ≥ ε / (2 f(u)) visits per draw: ≈ 0.02 at u = 0.99, 1.6 at u = 0.54, 7.4 at u = 0.35.

**The harvest's composition.**  Strings enter the round's boundary population from

- population re-sifts and final sifts: state q at rate (mass of q among length-L strings and
  leaf populations) × u(q) — blind to where the disagreement is;
- edge resolution: members·c, likewise blind;
- the bisection: only on disagreeing probes, and it reads the prefix where the walk and the
  sift part — rate ≈ N·d_q·u(q), with d_q the disagreement diverging at q.

Measured: trap round 1 (d = 0.021) — re-sift 108 + final sift 35 of 152, bisection 3; the badly
read states (c0 u 0.27, c1 0.36, c2 0.35 per sift) are the *common* ones, so the blind sources
are heavy anyway.  Generated target 20 round 1 (d = 0.67) — bisection 90 of 156.

## Where main can fail

The next gate (limit τ) is forced iff the population's indecision under a family that kept q's
reading exceeds τ.  With bisection mass B ≈ εN·u (N ≥ patience = 149) and blind contamination
C ≈ (n_pop + N)·u_c (u_c the clean per-sift rate; ≈ 0.011 at signal 0.2, ≈ 3e-4 at 0.3):

    forced  ⇐  B·u / (B + C) > τ   ⟺   εN·u² > τ(εN·u + C)

At signal 0.2 (εN ≈ 3, C ≈ 10) this needs u > 0.63.  A state with 0.18 ≤ u < 0.63 that is
visited ≥ ε/(2f(u)) times per draw **mid-string** but is rare among length-L strings and leaf
populations escapes: the blind sources don't sample it, the bisection's few strings are diluted
by C, the pass's probes read only ε, length-L strings and bisections so #407 does not halve, and
the clustering never sees it (it is absent from every population).  At signal 0.3 C ≈ 0.3 and
the same inequality holds down to u ≈ τ — the regime is empty there.

## The fix: keep the bisection's harvest apart

Put the strings the disagreement search could not place (`first_disagreeing_edge`'s undecided
reads, already harvested today) in their own population, grown by their `Walked` provenance.
Its contamination is only the clean reads *inside* bisections, R_bis = log₂L · r ≈ 15 reads per
disagreeing probe, so

    share_bad ≥ u / (u + R_bis·u_c)      forced  ⇐  u² > τ(u + R_bis·u_c)

At signal 0.2 that holds for u > 0.16 — below u* = 0.18, so **every state able to drive a refusal
forces the next gate**.  At signal 0.3 it holds for u > ≈ 0.105.  Cost: no new reads; one more
population per round (persisting like the boundary ones), grown only when the FNR gate blames it.
The family search must then find a family reading q better — which the proved clustering
guarantee supplies (no new reliance on clustering beyond `ClusteringGuarantee`).

Caveats: the population is small (≈ εN·u ≥ 0.5 strings a round) until blamed and grown, so the
"forced" step holds in expectation and needs the growth-by-provenance argument for its size;
the Gaussian per-node model and z_m ≈ 2.6 should be replaced by the binomial tails in a proof.

**Decided wrong edges** need no harvest on main: the bisection completes, and an UNDECIDED or
NO_SPLIT verdict re-adds `sprime` to the leaf's members (`add_first`), so within the round the
minority side accumulates until the split test splits; #403 additionally harvests NO_SPLIT.

## A target to test (c)

Wanted: a state with u ≈ 0.3–0.5 under typical accept-preserving families (acceptance fraction
under the family ≈ 0.25 or 0.75 at signal 0.2, i.e. its vote sits ~2.5–3σ off the boundary),
visited ≥ 2–7 times per draw mid-string, rare at length L, and with misreads persisting to the
end; signal 0.2.  Recipe: a "pre-trigger" region with self-loops, left permanently on a trigger
that almost every string meets by mid-string (e.g. first "11"), whose states' acceptance under a
random suffix is ≈ 1/4 (two independent parity bits of the suffix), feeding a common recurrent
region (e.g. a mod-3 counter) at an offset set by the pre-trigger state.  Prediction: main
refuses repeatedly with d just above ε, no halving, harvest ≈ 20% bad, and progresses only when
a re-clustered family moves those states' votes; with the fix, the bisection population is
≈ 70%+ bad and binds the next gate.

## The six-outcome statement: where it fails as specified

The specified outcomes are (1) pass, (2) the existing populations force the next gate, (3) halve,
(4) the bisection population forces the next gate at τ, (5) decided disagreements, and
(6) depth > D. The bisection population is #410 at 1f287c5:
- the bisection's first undecided read;
- an undecided final read whose middle-of-band side leaves the walk (the cheap check, with no
  read below it);
- the empty string's first undecided read, where the gate's start parts from the cut at the
  anchor.

Its growth law is `Bisected`.

**What the gate reads.** The gate reads ε's path and the draw's own path, and nothing else. Take
P the majority reading (each node read on its likelier side; it depends only on states). A
counted draw x has at least one of:
- E1: ε's middle path ≠ P(ε);
- E2(x): H walked from P(ε) along x ≠ P(x);
- E3(x): x's middle path ≠ P(x).

E2 depends on the noise only through H.

**(5) has to be E2's mass, not "bisections that complete decided".** Take a decided wrong edge
into a clean state, carrying mass m ≈ 3ε. Its bisection reads R node strings of states with
u ≈ τ/2. Then:
- At R = 40 and τ = 0.1, 87% of these bisections stop at an undecided read with u ≈ 0.05 < τ.
- So the population sits at ≈ 0.05 average indecision and is not forced.
- Completed bisections carry 0.13·m ≈ 0.4ε, under any ε₅ ≥ ε/2.
- Unchecked/reads ≈ 0.004 < τ/2, so (3) fails; the Walked harvest is low too, so (2) fails.

With (5) = D{E2} > ε₅, that round is (5).

**The cheap final-read check loses "deep" flips.** Suppose x's first undecided read is in the band
on its majority side, and a deeper bad read below it flips. The gate counts x; the pass keeps
nothing.
- Spread over many strings this costs only a constant factor of the kept bad mass.
- Concentrated on a few heavy strings it costs everything. Take one string of mass 2ε whose root
  read is in the band on its majority side with chance 0.5, and whose node below flips with
  chance 0.5:
  - with chance ≈ 0.25 the round refuses with an empty bisection contribution;
  - unchecked/reads ≈ ε/r, so (3) does not halve;
  - the Walked harvest is diluted below 2τ, so (2a) fails.

Closing it needs either an anti-concentration term, about max_x D{x}/ε, or reading on at the
middle below the first undecided node.

**ε's first undecided read can be cheap and wrong.** If ε's first undecided read has u < τ and a
deeper bad read on ε's path flips, every draw is counted. The population holds that low read
on every draw. The chance of this is up to Σ u over ε's low-u path reads, which is ≤ D·τ.

Keeping every in-band read on ε's middle path costs only ε's path reads, once per tree. Its low
reads are independent of the bad flip, so Markov turns the failure into the contamination term
D·τ²/(4(u* − τ)).

## `RoundProgress`: the provable statement, and paring it down

`RoundProgress` (`Round.lean`, proved as `round_progress`) models #410 at 2dd6e04 and #412 in `Pass.lean`:
- the walk from the gate's start, restarting from the anchor where they part;
- a NO_SPLIT verdict keeps probing as UNDECIDED does (#412), so the pass ends only through
  patience or by running out of probes;
- the bisection population: the search's first undecided read, the prefix before a placeholder
  edge the search lands on, the in-band reads of the gate's reading of an unplaceable probe that
  leaves the walk, and the in-band reads of the gate's reading of `ε`.

The gate's reads are not counted in `probeReads`. The population is stated over its distinct
strings.

Premises: `Valid`, `0 < ε`, `0 ≤ φ`, `4(N+1)φ < 1`, draws of length `L`, a suffix-free vote family,
and `MidFlipPremise uHi φ`. Bound: `(L+1)(N+1)φ/ε + passReadBound·φ`.

| outcome | status | reason |
|---|---|---|
| (1) agreement within ε | kept | |
| (2) the existing populations force the next gate | kept | |
| (3) halving, as #407 counts it | kept | |
| (4) the bisection population forces the next gate at τ | kept | the target the catch-alls are to be pared into |
| (5) some state read badly, u ≥ uHi | catch-all | `gate_flip_bound` needs every state clean; paring needs the forcing argument, (5) ⇒ (4) |
| (6) a visited edge read off more often than not | catch-all | with every state clean, a wrong edge by the majority reading. NO_SPLIT dismissals are gone (#412) and placeholder drops are held (#410); still needed for searches blocked by undecided reads within the limit, which stay quiet until #411 |
| (7) the pass runs out of probes before a patience streak | kept | the only other way a pass ends now |
