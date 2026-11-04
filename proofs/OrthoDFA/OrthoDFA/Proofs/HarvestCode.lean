import OrthoDFA.Code.Level3HarvestCode
import OrthoDFA.Code.LearnerLoop
import OrthoDFA.Proofs.RandomHarvest
import OrthoDFA.Proofs.KappaReads
import OrthoDFA.Proofs.RandomSearch

/-! # Decoupling the harvest sampler from the harvest law

The three facts that make `harvestCode` a drop-in for `harvestLaw` in the loop:

* `harvestCodeLaw_popLaw_eq` — (A1) conditioned on a hit, its output law is exactly
  `popLaw (harvestLaw …)`.  No `δ`, no good-event hypothesis: rejection sampling reproduces the
  conditional law exactly.
* `harvestCodeLaw_miss_le` — (A2) the all-miss mass is `(1 - hitProb) ^ N`; with the round's yield
  bound `y ≤ hitProb` and `N = harvestN y δₛ`, it is `≤ δₛ`.
* `harvestCode_reads_card_le` — (B) the sampler reads only what its (at most `|xs|`) walked probes
  read, `≤ |xs|` probe-walks each of `≤ 2 + clog₂ L` reads. -/

namespace OrthoDFA

namespace Random

open MeasureTheory

variable {α : Type*} [Fintype α]
variable (read : FreeMonoid α → ARU) (D : Measure (FreeMonoid α)) (k : ℕ) (H : HypothesisDFA α)
  (site : Level3Site) (N : ℕ) (xs : List (FreeMonoid α))

open scoped Classical in
/-- The strings the bounded-rejection sampler reads: the union of `probeReads` over the probes it
walks (all of `xs` in the worst case). -/
noncomputable def harvestFirstReads (read : FreeMonoid α → ARU) (T : DTree α)
    (E : Edges α) (k : ℕ) (xs : List (FreeMonoid α)) : Finset (FreeMonoid α) :=
  xs.foldr (fun x acc => Kappa.probeReads read T E k x ∪ acc) ∅

/-- (A1) Agreement — exact.  Conditioned on a hit, the sampler's output law is the harvest law
(both normalised by `popLaw`), for every cap `N`. -/
theorem harvestCodeLaw_popLaw_eq [IsProbabilityMeasure D] :
    popLaw (harvestCodeLaw read D k H site N) = popLaw (harvestLaw read D k H site) := by
  sorry

/-- (A2) Failure mass.  The chance all `N` probes miss is `(1 - hitProb) ^ N`; with the (round-good)
yield bound `y ≤ hitProb = (harvestLaw …).real univ` and `N = harvestN y δₛ`, it is `≤ δₛ`. -/
theorem harvestCodeLaw_miss_le [IsProbabilityMeasure D] {y δₛ : ℝ}
    (hy0 : 0 < y) (hy1 : y ≤ 1) (hδ : 0 < δₛ)
    (hyield : y ≤ (harvestLaw read D k H site).real Set.univ) :
    harvestMiss read D k H site (harvestN y δₛ) ≤ ENNReal.ofReal δₛ := by
  sorry

/-- (B) Cost.  The sampler reads only the strings its walked probes read; when every probe is at
most `L` long this is `≤ |xs| · (2 + clog₂ L)`.  (At the loop's `read = familyRead … F`, each such
read is `|F|` raw oracle queries; that factor is applied where the sampler is wired in.) -/
theorem harvestCode_reads_card_le {L : ℕ} (hxs : ∀ x ∈ xs, x.toList.length ≤ L) :
    (harvestFirstReads read H.tree H.edges k xs).card ≤ xs.length * (2 + Nat.clog 2 L) := by
  sorry

end Random

end OrthoDFA
