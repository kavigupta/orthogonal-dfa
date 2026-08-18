# Nonlinear motif harvest on the composition-controlled SpliceAI oracle

Perturbation analysis of the composition-deconfounded SpliceAI-400 oracle, on the
`dataset_test/X8/361` exon: a **189-nt** exon body (the scored middle), flanked by 202 nt
of its real genomic context on each side (SpliceAI context `cl=400`). Position 0 is the
acceptor/exon-start, position 188 the donor/exon-end.

The module offers **two views** of the residual's non-compositional structure, and the
scientific story is the progression from the first to the second:

1. `harvest` / `single_base_ism` / `motif_effects` — **position-specific** saliency and
   epistasis (in-silico mutagenesis at fixed positions).
2. `context_motif_significance` — **position-agnostic, in-context** significance of a
   motif's effect vs the alternatives, aggregated over contexts.

---

## 1. Position-specific saliency (`harvest`)

In-silico mutagenesis, 200 backgrounds, 3-mers. The saliency (per-position range of the
single-base ISM effect) concentrates hard at the **donor site**:

```
top saliency positions:  185 (range 0.36, favours A), 183 (0.29, T), 184 (0.28, A),
                         then a long drop-off (86, 22, 28, 7, ...) around 0.14
```

Separating additive from epistatic there (`nonlinear = motif effect − sum of single-base
effects`):

- **Strongest perturbations are mostly additive** — the donor positional PWM (positional,
  so linear deconfounding cannot remove it): `GGG@183` effect −0.46 / nonlinear +0.01,
  `TAA@183` +0.457 / nonlinear −0.008.
- **Genuinely epistatic donor motifs**: `GGT@184` effect −0.77 / **nonlinear −0.36**,
  `GTA@185` **−0.31** — specific donor-edge 3-mers worse than their bases predict.

Position-specific saliency thus localises to real splice biology, but it is **dominated by
the donor PWM**, which is position-locked. To find structure that acts *in context anywhere*
we go position-agnostic.

## 2. In-context motif significance (`context_motif_significance`)

At each of many consistent-background contexts `(background, position)`, overwrite
`[p, p+k)` with *every* k-mer and record the score change `delta`. Subtracting the
per-context mean over k-mers, `rel = delta − mean_kmers(delta)`, isolates the inserted
motif's identity — how much more (or less) it moves the score than a typical substitution
at that exact spot — controlling for the removed bases and the position. Aggregated over
contexts (positions sampled across the exon) this is **position-agnostic**.

Two summaries per motif, and the difference between them *is* the main lesson:

- **signed** (`effect`, `tstat`): the signed mean of `rel`. A motif whose sign flips by
  context cancels.
- **magnitude** (`|effect| = mean|rel|`, `mag_z`): the size of the effect regardless of
  sign. **Rank by this** — a context-dependent motif (the whole point of "in-context")
  flips sign and is invisible to the signed mean.

### 2a. Signed ranking is misleading — it surfaces CpG (leftover composition)

Top *signed* motifs on the **linear** oracle are CpG-forward, all *raising* the score:

```
CGA t+5.0   ATC t+4.0   GGC t−3.9   GCC t−3.8   TCG t+3.6   TCA t+3.6   CGC t+3.3
```

But their magnitude is unremarkable (mag_z ≈ 0). They are a weak, consistent-direction
effect — and (§3) they vanish under the gate, i.e. they are leftover composition.

### 2b. Magnitude ranking surfaces the reading-frame stop codons

Ranked by magnitude (3-mers, 4000 contexts), **the top three motifs are the three stop
codons**:

```
TAA  |eff|=0.437 (mag_z +3.6)  signed −0.018 (t −1.8)
TAG  |eff|=0.426 (mag_z +3.4)  signed −0.013 (t −1.4)
TGA  |eff|=0.415 (mag_z +3.2)  signed −0.008 (t −0.8)
AGG  0.337 (+1.7)  GTA 0.336 (+1.7)  ATA 0.307 (+1.1)  ...
```

Their magnitude is **~2× the next motif and ~3.5 SD above the average k-mer**, but their
signed mean is near zero (t ≈ −1 to −2) — because a stop codon's effect is
**context-dependent**: it sharply *lowers* the score when it closes the last open reading
frame (and is in-frame), and does little otherwise. Signed-averaging cancels this; the
magnitude ranking surfaces it. This is the reading-frame / ORF signal, the largest
in-context effect in the oracle.

## 3. Genuine nonlinear structure vs leftover composition (linear vs gate)

Comparing the linear residual (removes only linear composition) to the monotonic-gate
residual (also removes monotone-nonlinear composition):

| motif | linear \|eff\| (mag_z) | linear signed t | gate \|eff\| (mag_z) | gate signed t |
| --- | --- | --- | --- | --- |
| **TAA** | 0.437 (+3.6) | −1.8 | 0.443 (+3.6) | −1.5 |
| **TAG** | 0.426 (+3.4) | −1.4 | 0.434 (+3.4) | **−3.7** |
| **TGA** | 0.415 (+3.2) | −0.8 | 0.414 (+3.1) | −2.2 |

- **Stop codons are unchanged** by the gate — the frame signal is genuinely
  **non-compositional**, surviving even the stronger deconfounding. The gate in fact
  *sharpens* its direction (`TAG` signed t −1.4 → **−3.7**): once composition is cleanly
  removed, stop codons consistently *lower* the score.
- **The CpG motifs vanish** under the gate. On the linear residual the top *signed* motifs
  were CpG-forward (`CGA`/`TCG`/`CGC`); on the gate residual they drop out of the top,
  replaced by C/T-rich motifs (`CAC`/`CTC`/`TCT`). So the CpG-raising effect was
  **monotone-nonlinear composition** the linear fit left in and the gate removes.

## Takeaway

The dominant nonlinear, in-context, non-compositional motif structure in the deconfounded
SpliceAI oracle is the **reading-frame stop-codon signal** (`TAA`/`TAG`/`TGA`) —
context-dependent, the largest effect vs alternatives, and robust to both deconfounding
methods. The apparent CpG signal on the linear oracle is leftover composition. (This is
consistent, from a different model class, with the direct-L\* result that the recoverable
non-compositional signal in this oracle is reading-frame closure.)

Three methodological points were each necessary to see this:
1. **in-context, position-agnostic** perturbation, not position-specific saliency
   (position-specific saliency is dominated by the donor PWM at positions 183–188);
2. **magnitude, not signed** aggregation (stop codons are sign-flipping);
3. **gate vs linear** control (separates genuine nonlinear structure from leftover
   monotone composition).

## Reproduce

```
python -m orthogonal_dfa.analysis.nonlinear_motif_miner            # in-context, linear residual
USE_GATE=1 python -m orthogonal_dfa.analysis.nonlinear_motif_miner # gate residual (needs the
                                                                   # gate oracle, PR #194)
```
Env: `N_CTX` (contexts, default 3000), `MOTIF_K` (k, default 3), `EDGE_MARGIN` (drop
positions within this many of the edges — use it to exclude the donor edge and confirm
what is context- vs edge-driven).

## Open follow-ups (for the next session)

- **Phase-aware contexts.** The frame signal is phase-locked (a stop codon only closes a
  frame when it is in-frame, position ≡ start mod 3). Sampling positions uniformly *dilutes*
  it (`TAA` signed t only −1.8). Fixing / stratifying position mod 3 should sharpen the
  frame signal and let the *signed* ranking recover it too.
- **`USE_GATE` depends on PR #194** (the monotonic-gate oracle). The gate import is lazy
  and pylint-guarded; the linear path works on `main` alone. Rebase/merge order:
  #194 → #196 to make the gate path importable on `main`.
- **Wider scans:** `MOTIF_K=2,4`; larger `N_CTX`; `EDGE_MARGIN` to isolate context- from
  edge-driven effects; and follow up the position-specific epistatic donor motifs
  (`GGT`/`GTA`@184–185).
