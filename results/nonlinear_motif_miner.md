# In-context motif significance on the composition-controlled SpliceAI oracle

Perturbation analysis of the composition-deconfounded SpliceAI-400 oracle, on the
`dataset_test/X8/361` exon: a **189-nt** exon body (the scored middle), flanked by 202 nt
of its real genomic context on each side (SpliceAI context `cl=400`). Position 0 is the
acceptor/exon-start, position 188 the donor/exon-end.

`marginal_motif_stats_adaptive` measures, for every k-mer, the **magnitude** of its effect
*in context* vs the alternatives, aggregated over contexts (position-agnostic); it is the
`magnitude` field of each `MotifRecord`.

## Method

At each of many consistent-background contexts `(background, position)`, overwrite
`[p, p+k)` with *every* k-mer and record the score change `delta`. Subtracting the
per-context mean over k-mers, `rel = delta − mean_kmers(delta)`, isolates the inserted
motif's identity — how much more (or less) it moves the score than a typical substitution
at that exact spot — controlling for the removed bases and the position. Aggregated over
contexts (positions sampled across the exon) this is **position-agnostic**.

Rank motifs by **magnitude** (`|effect| = mean|rel|`, `mag_z` = SDs above the average
k-mer): the size of the effect regardless of sign. A context-dependent motif (the whole
point of "in-context") flips sign by context and is invisible to a signed mean, so
magnitude is the summary that surfaces it.

## Result: magnitude ranking surfaces the reading-frame stop codons

Ranked by magnitude (3-mers, 4000 contexts), **the top three motifs are the three stop
codons**:

```
TAA  |eff|=0.443 (mag_z +3.6)
TAG  |eff|=0.434 (mag_z +3.4)
TGA  |eff|=0.414 (mag_z +3.1)
AGG  0.337 (+1.7)  GTA 0.336 (+1.7)  ATA 0.307 (+1.1)  ...
```

Their magnitude is **~2× the next motif and ~3.5 SD above the average k-mer**. The effect
is **context-dependent**: a stop codon sharply *lowers* the score when it closes the last
open reading frame (and is in-frame), and does little otherwise — so a signed average would
cancel it, but the magnitude ranking surfaces it. This is the reading-frame / ORF signal,
the largest in-context effect in the oracle.

The signal is **non-compositional**: it survives the monotonic-gate deconfounding, which
removes both first-order and monotone-nonlinear composition. (This is consistent, from a
different model class, with the direct-L\* result that the recoverable non-compositional
signal in this oracle is reading-frame closure.)

## Reproduce

```python
from orthogonal_dfa.analysis.nonlinear_motif_miner import marginal_motif_stats_adaptive

records, info = marginal_motif_stats_adaptive(max_k=3, target_error=0.04, batch=200)
for r in sorted(records, key=lambda r: -r.magnitude)[:10]:
    print(r)
```
`notebooks/nonlinear_motif_miner.ipynb` runs this (and the marginal-benefit ranking the
records are sorted by) for `max_k=4` and `max_k=6`. Sampling is adaptive, so `target_error`
replaces a fixed context count; `edge_margin` drops positions within that many of the edges
— use it to exclude the donor edge and confirm what is context- vs edge-driven.

## Open follow-ups (for the next session)

- **Phase-aware contexts.** The frame signal is phase-locked (a stop codon only closes a
  frame when it is in-frame, position ≡ start mod 3). Sampling positions uniformly *dilutes*
  it. Fixing / stratifying position mod 3 should sharpen the frame signal.
- **Wider scans:** larger `max_k`; tighter `target_error`; `edge_margin` to isolate
  context- from edge-driven effects.
