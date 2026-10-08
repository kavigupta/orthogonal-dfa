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
