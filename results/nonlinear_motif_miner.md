# Nonlinear motif harvest on the composition-controlled SpliceAI oracle

Perturbation analysis of the composition-deconfounded SpliceAI-400 oracle, run on the
`dataset_test/X8/361` exon: a **189-nt** exon body (the scored middle), flanked by 202 nt
of its real genomic context on each side (SpliceAI context `cl=400`). Position 0 is the
acceptor/exon-start, position 188 the donor/exon-end.

## Method

`context_motif_significance` measures each k-mer's **in-context** effect *against the
alternatives*: at each of many consistent-background contexts `(background, position)`,
overwrite `[p, p+k)` with *every* k-mer and record the score change `delta`. Subtracting
the per-context mean over k-mers, `rel = delta - mean_kmers(delta)`, isolates the
inserted motif's identity — how much more (or less) it moves the score than a typical
substitution at that exact spot — controlling for the removed bases and the position.
Aggregated over contexts (positions sampled across the exon) this is **position-agnostic**.

Two summaries per motif, because direction matters:

- **signed** (`effect`, `tstat`): the signed mean of `rel`. Flags consistent-direction
  motifs; a motif whose sign flips by context cancels.
- **magnitude** (`|effect| = mean|rel|`, `mag_z`): the size of the effect regardless of
  sign, and how many SDs it exceeds the average k-mer's. **This is the one to rank by**,
  because a context-dependent motif — the whole point of "in-context" — flips sign and is
  invisible to the signed mean.

## Result: the reading-frame stop codons are the dominant in-context motif

Ranked by magnitude (3-mers, 4000 contexts), the **top three motifs are the three stop
codons**, on *both* the linear- and gate-controlled oracles:

| motif | linear \|eff\| (mag_z) | linear signed t | gate \|eff\| (mag_z) | gate signed t |
| --- | --- | --- | --- | --- |
| **TAA** | 0.437 (+3.6) | −1.8 | 0.443 (+3.6) | −1.5 |
| **TAG** | 0.426 (+3.4) | −1.4 | 0.434 (+3.4) | −3.7 |
| **TGA** | 0.415 (+3.2) | −0.8 | 0.414 (+3.1) | −2.2 |

Their **magnitude is ~2× the next motif and ~3.5 SD above the average k-mer**, but their
**signed** mean is near zero (t ≈ −1 to −2) — because a stop codon's effect is
**context-dependent**: it sharply *lowers* the score when it closes the last open reading
frame (and is in-frame), and does little otherwise. Signed-averaging cancels this; the
magnitude ranking surfaces it. This is the reading-frame / ORF signal, and it is the
largest in-context effect in the oracle.

## Genuine nonlinear structure vs leftover composition (linear vs gate)

Comparing the linear residual (removes only linear composition) to the monotonic-gate
residual (also removes monotone-nonlinear composition) separates the two:

- **Stop codons are unchanged** by the gate (`TAA` 0.437 → 0.443, etc.) — the frame
  signal is genuinely **non-compositional**, surviving even the stronger deconfounding.
  The gate in fact *sharpens* its direction: `TAG`'s signed t goes −1.4 → **−3.7**.
- **The CpG motifs vanish** under the gate. On the linear residual the top *signed*
  motifs were CpG-forward (`CGA` t+5.0, `TCG` t+3.6, `CGC` t+3.3, all containing CG,
  *raising* the score); on the gate residual they drop out of the top, replaced by
  C/T-rich motifs (`CAC`, `CTC`, `TCT`). So the CpG-raising effect was **monotone-
  nonlinear composition** the linear fit left in and the gate removes — not genuine
  context structure.

## Takeaway

The dominant nonlinear, in-context, non-compositional motif structure in the
deconfounded SpliceAI oracle is the **reading-frame stop-codon signal** (`TAA`/`TAG`/`TGA`)
— context-dependent, the largest effect vs alternatives, and robust to both deconfounding
methods. The apparent CpG signal on the linear oracle is leftover composition.

Three methodological points were each necessary to see this:
1. **in-context, position-agnostic** perturbation, not position-specific saliency
   (position-specific saliency is dominated by the donor PWM at positions 183–188);
2. **magnitude, not signed** aggregation (stop codons are sign-flipping);
3. **gate vs linear** control (separates genuine nonlinear structure from leftover
   monotone composition).

## Reproduce

```
python -m orthogonal_dfa.analysis.nonlinear_motif_miner            # linear residual
USE_GATE=1 python -m orthogonal_dfa.analysis.nonlinear_motif_miner # gate residual (needs the
                                                                   # gate oracle, PR #194)
```
Env: `N_CTX` (contexts, default 3000), `MOTIF_K` (k, default 3), `EDGE_MARGIN` (drop
positions within this many of the edges).
