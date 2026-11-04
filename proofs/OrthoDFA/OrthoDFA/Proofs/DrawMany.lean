import OrthoDFA.Code.Level3DrawMany
import OrthoDFA.Proofs.HarvestCode

/-! # Decoupling the `m`-draw sampler from the harvest law

The three facts that make `drawMany >>= clustering` a drop-in, with clustering oblivious to the
draws:

* `drawManyLaw_eq_pi` — (AGR) conditioned on all `m` draws hitting, the output law is exactly
  the `m`-fold product of `popLaw (harvestLaw …)`.  Exact: the draws are i.i.d. and each coordinate
  is `harvestCodeLaw`, whose normalised law is `popLaw (harvestLaw …)` by
  `harvestCodeLaw_popLaw_eq`.
* `drawManyMiss_le` — (FAIL) the chance any draw misses is `≤ δₛ'`, by a union bound over the `m`
  draws with each `harvestMiss ≤ δₛ'/m` at `N' = harvestN y (δₛ'/m)`.
* `drawManyReads_card_le` — (COST) the sampler reads only what its `m` blocks of `N'` probes read,
  `≤ m · N' · ((3 + clog₂ L) · |mids T|)`. -/

namespace OrthoDFA

namespace Random

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α]
variable (read : FreeMonoid α → ARU) (D : Measure (FreeMonoid α)) (k : ℕ) (H : HypothesisDFA α)
  (site : Level3Site) {m N' : ℕ}

open scoped Classical in
/-- The strings the `m`-draw sampler reads: the union of `harvestFirstReads` over its `m` blocks. -/
noncomputable def drawManyReads (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α) (k : ℕ)
    {m N' : ℕ} (yss : Fin m → (Fin N' → FreeMonoid α)) : Finset (FreeMonoid α) :=
  Finset.univ.biUnion fun i => harvestFirstReads read T E k (List.ofFn (yss i))

/-- (AGR) Agreement — exact.  Conditioned on all `m` draws hitting, the sampler's output law is the
`m`-fold product of the harvest law (each coordinate normalised by `popLaw`), for every cap
`N' ≥ 1`.  Proof: the raw (unnormalised) law factors as `Measure.pi (fun _ => harvestCodeLaw …)`
(the all-hit event and the coordinatewise map both factor over `Fin m`); normalising commutes with
the i.i.d. product (popLaw of a product is the product of popLaws), and `harvestCodeLaw_popLaw_eq`
rewrites each factor. -/
theorem drawManyLaw_eq_pi [IsProbabilityMeasure D] (hN : 0 < N') :
    drawManyLaw read D k H site N' m
      = Measure.pi (fun _ : Fin m => popLaw (harvestLaw read D k H site)) := by
  sorry

/-- (FAIL) Failure mass.  With the (round-good) yield bound `y ≤ (harvestLaw …).real univ` and
`N' = harvestN y (δₛ'/m)`, the chance any of the `m` draws misses is `≤ δₛ'`.  Proof: the miss event
is `⋃ i, {draw i misses}`; a union bound gives `≤ ∑ i, harvestMiss … N'`, and each term is
`≤ δₛ'/m` by `harvestCodeLaw_miss_le`, so the total is `≤ m · (δₛ'/m) = δₛ'`. -/
theorem drawManyMiss_le [IsProbabilityMeasure D] {y δₛ' : ℝ}
    (hy0 : 0 < y) (hy1 : y ≤ 1) (hδ : 0 < δₛ') (hm : 0 < m)
    (hyield : y ≤ (harvestLaw read D k H site).real Set.univ) :
    drawManyMiss read D k H site (harvestN y (δₛ' / m)) m ≤ ENNReal.ofReal δₛ' := by
  sorry

/-- (COST) Cost.  When every probe in every block is at most `L` long, the sampler reads
`≤ m · N' · ((3 + clog₂ L) · |mids T|)` strings: `m` blocks, each `harvestCode_reads_card_le` over
its `N'` probes.  (At the loop's `read = familyRead … F`, each read is `|F|` raw oracle queries;
that factor is applied where the sampler is wired in.) -/
theorem drawManyReads_card_le {L : ℕ} (yss : Fin m → (Fin N' → FreeMonoid α))
    (hyss : ∀ i, ∀ x ∈ List.ofFn (yss i), x.toList.length ≤ L) :
    (drawManyReads read H.tree H.edges k yss).card
      ≤ m * (N' * ((3 + Nat.clog 2 L) * (mids H.tree).card)) := by
  sorry

end Random

end OrthoDFA
